"""Opt-in text-first QQ voice replies, with durable fail-closed attempt claims."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from contextvars import copy_context
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import time
import tempfile
import threading

from .meme_native import _matches

logger = logging.getLogger(__name__)
FALLBACK = '语音暂时发送失败，请先看文字回复。'
_AUDIO = {'.wav', '.mp3', '.ogg', '.opus', '.silk', '.amr', '.m4a', '.flac'}

# Module-generation local, but profile/transport/canonical-session partitioned.
# Eviction fails closed for an old pending reply, never makes it current again.
_CURRENT_TURNS = OrderedDict()
_STARTED_TURNS = OrderedDict()
_INPUT_LANES = set()
_ACTIVE_SENDS = {}
_ACTIVE_REPLIES = {}
# Optional slow providers must not exhaust asyncio's shared executor (admission,
# database and ordinary tools use it too). Cancelled queued work never starts.
_SYNTHESIS_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix='qq-auto-tts')


def _turn_key(adapter, session_key):
    return (str(adapter._receipt_home.resolve()), adapter._app_id,
            getattr(adapter, '_owner_profile', None) or 'default', session_key)


def begin_turn(adapter, event, session_key, *, incoming=False):
    """Fallback for restored turns; an older queued turn cannot undo newer intake."""
    turn = getattr(event, '_qq_meme_turn', None)
    if (not _matches(turn, event) or event.internal or event.inherit_channel_pin
            or session_key != adapter._source_session_key(event.source)):
        return
    key = _turn_key(adapter, session_key)
    # The native bridge shares this trusted turn object with the delivery waiter.
    # Re-entering preparation must not resurrect an older executing turn.
    if getattr(turn, '_voice_generation', None) is not None:
        return
    identity = (key, event.message_id)
    if identity in _STARTED_TURNS:
        turn._voice_generation = _STARTED_TURNS[identity]
        return
    # Live intake registers first. Recovery without intake uses the execution hook.
    # A fresh object prevents an old in-flight generation reviving after eviction.
    generation = (getattr(event, '_gateway_active_turn_token', None) or event.message_id, object())
    turn._voice_generation = generation
    if not incoming and key in _INPUT_LANES:
        # Unknown restored executions cannot supersede a known live input.
        return
    _STARTED_TURNS[identity] = generation
    if incoming:
        _INPUT_LANES.add(key)
    _CURRENT_TURNS[key] = generation
    _CURRENT_TURNS.move_to_end(key)
    for task, (message_id, _) in tuple(_ACTIVE_REPLIES.get(key, {}).items()):
        if message_id != event.message_id:
            task.cancel()
    # Stop an awaited codec/upload before it can post the old message. A POST
    # already submitted to QQ remains unknown, never recalled or retried.
    for task, message_id in tuple(_ACTIVE_SENDS.get(key, {}).items()):
        if message_id != event.message_id:
            task.cancel()
    while len(_CURRENT_TURNS) > 2048:
        expired, _ = _CURRENT_TURNS.popitem(last=False)
        _INPUT_LANES.discard(expired)
    while len(_STARTED_TURNS) > 4096:
        _STARTED_TURNS.popitem(last=False)


def receive_input(adapter, event):
    """Notify before admission can wait, after canonical route and authorization checks."""
    from gateway.request_security import policy_for_adapter
    from gateway.platforms.base import coerce_plaintext_gateway_command
    if (not _matches(getattr(event, '_qq_meme_turn', None), event)
            or event.internal or event.inherit_channel_pin):
        return
    if not adapter._message_handler or adapter._drop_unresolved(event):
        return
    if event.allow_gateway_control:
        coerce_plaintext_gateway_command(event)
    source = event.source
    if adapter._is_sender_authorized(source.user_id, source.chat_type, source.chat_id,
                                     thread_id=source.thread_id) is not True:
        return
    policy = policy_for_adapter(adapter, source)
    command = event.get_command()
    if policy is not None and command and not policy.allows_command(command):
        return
    key = adapter._source_session_key(source)
    expected = str((event.metadata or {}).get('gateway_session_key') or '').strip()
    if expected and expected != key:
        return
    begin_turn(adapter, event, key, incoming=True)


def schedule(adapter, event, session_key, text, metadata=None, *, ephemeral=False):
    """The text ACK owns completion. Track optional audio outside the reply lane."""
    if not _eligible(adapter, event, text, ephemeral, metadata) or not _is_current(adapter, event, session_key):
        return
    key = _turn_key(adapter, session_key)
    active = _ACTIVE_REPLIES.setdefault(key, {})
    if any(message_id == event.message_id for message_id, _ in active.values()):
        return
    tasks = getattr(adapter, '_qq_voice_tasks', None)
    if tasks is None:
        tasks = adapter._qq_voice_tasks = set()

    async def deliver():
        from .delivery_binding import bind_delivery, turn_context
        # Re-enter the captured source's scope even after the foreground turn ends.
        with adapter._media_delivery_scope(event.source), bind_delivery(
                event.source.chat_id, turn_context(event, session_key)):
            await after_text(adapter, event, session_key, text, metadata, ephemeral=ephemeral)

    task = asyncio.create_task(deliver(), name='qq-optional-voice')
    tasks.add(task)
    active[task] = (event.message_id, adapter)

    def done(completed):
        tasks.discard(completed)
        active.pop(completed, None)
        if not active and _ACTIVE_REPLIES.get(key) is active:
            _ACTIVE_REPLIES.pop(key, None)
        if not completed.cancelled() and completed.exception() is not None:
            logger.warning('QQ optional voice task failed: %s', type(completed.exception()).__name__)

    task.add_done_callback(done)


async def shutdown(adapter):
    """Cancel optional delivery before this transport's HTTP client is closed."""
    adapter._qq_voice_stopping = True
    tasks = tuple(getattr(adapter, '_qq_voice_tasks', ()))
    for task in tasks:
        task.cancel()
    if tasks:
        _, pending = await asyncio.wait(tasks, timeout=5)
        if pending:
            logger.warning('QQ optional voice shutdown: %d tasks still unwinding', len(pending))


def _is_current(adapter, event, session_key):
    if getattr(adapter, '_qq_voice_stopping', False):
        return False
    turn = getattr(event, '_qq_meme_turn', None)
    generation = getattr(turn, '_voice_generation', None)
    key = _turn_key(adapter, session_key)
    current = _CURRENT_TURNS.get(key)
    if generation is None and _matches(turn, event):
        # A native waiter may retain its original ingress stamp rather than the
        # executing object's dynamic attributes. Resolve the executed generation
        # by its trusted identity, never by the newest turn or response text.
        generation = _STARTED_TURNS.get((key, event.message_id))
    # Standalone final callers have no execution hook. They are compatible only
    # when no newer prepared turn is known; never promote a final to current.
    if generation is not None:
        return current is generation
    return current is None


async def _send_current_voice(adapter, event, session_key, path, metadata):
    """Structured child: inherits delivery scope and is always awaited/reaped."""
    key = _turn_key(adapter, session_key)
    task = asyncio.create_task(adapter.send_voice(chat_id=event.source.chat_id,
        audio_path=path, reply_to=event.message_id, metadata=metadata))
    active = _ACTIVE_SENDS.setdefault(key, {})
    active[task] = event.message_id
    try:
        return await task
    except asyncio.CancelledError:
        if not asyncio.current_task().cancelling() and not _is_current(adapter, event, session_key):
            return None  # A newer turn cancelled only this old optional send.
        raise
    finally:
        active.pop(task, None)
        if not active:
            _ACTIVE_SENDS.pop(key, None)


class _AudioPaths(list):
    """Only granted files in this call's exclusive directory are disposable."""
    def __init__(self, paths, directory, policy):
        super().__init__(paths)
        self.directory = directory
        self.policy = policy
        self.owned = {}
        for raw in paths:
            path = Path(raw).resolve()
            if path.parent == directory.resolve() and policy.allows_media(path):
                self.owned[path] = path.stat()

    def discard(self):
        for path, original in self.owned.items():
            try:
                if not self.policy.allows_media(path):
                    continue
                now = path.stat()
                if (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns) != (
                        original.st_dev, original.st_ino, original.st_size, original.st_mtime_ns):
                    continue  # A replacement belongs to someone else.
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning('QQ unused voice cleanup failed: %s', type(exc).__name__)
        try:
            self.directory.rmdir()  # Never recursively remove provider/user files.
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug('QQ unused voice directory retained (not empty or locked)')


class _SynthesisJob:
    """A cancelled asyncio waiter cannot cancel the provider thread's cleanup."""
    def __init__(self):
        self.lock = threading.Lock()
        self.abandoned = False
        self.paths = None

    def run(self, is_current, *args):
        if not is_current():
            return []
        paths = _synthesize(*args)
        with self.lock:
            self.paths = paths
            if self.abandoned and isinstance(paths, _AudioPaths):
                paths.discard()
        return paths

    def discard(self):
        with self.lock:
            self.abandoned = True
            if isinstance(self.paths, _AudioPaths):
                self.paths.discard()


def enabled(adapter):
    options = (adapter.config.extra or {}).get('voice_reply', {})
    return isinstance(options, dict) and options.get('enabled') is True


def mark_response(event, extracted):
    # Inspect the extracted attachments, not arbitrary MEDIA strings in prose.
    event._qq_explicit_voice = any(
        voice or Path(path).suffix.lower() in _AUDIO for path, voice in extracted.media_files)


def _eligible(adapter, event, text, ephemeral, metadata):
    turn = getattr(event, '_qq_meme_turn', None)
    if not enabled(adapter) or ephemeral or not text.strip() or not _matches(turn, event):
        return False
    if (event.internal or event.inherit_channel_pin or event.is_command()
            or not turn.prepared or getattr(event, '_qq_explicit_voice', False)
            or getattr(event, '_adapter_chained_response', False)
            or getattr(event, '_heartbeat_session_id', None)
            or (metadata or {}).get('_interim_send')
            or adapter._strip_at_mention(turn.input_text).lstrip().startswith('/')
            or getattr(event, 'ledger_message_id', None) not in (None, event.message_id)):
        return False
    source = event.source
    if source.chat_type == 'group':
        return adapter._guess_chat_type(source.chat_id) == 'group'
    return (source.chat_type in {'dm', 'c2c'} and source.chat_id == source.user_id
            and adapter._guess_chat_type(source.chat_id) == 'c2c')


def _claim(home, identity):
    path = home / 'cache' / 'qqbot' / 'voice_replies.sqlite3'
    path.parent.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(identity, ensure_ascii=True).encode()).hexdigest()
    with closing(sqlite3.connect(path, timeout=5)) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, state TEXT NOT NULL)')
        # Never reclaim unknown/failed attempts: a lost ACK is not evidence of non-delivery.
        return db.execute("INSERT OR IGNORE INTO attempts VALUES (?, 'unknown')", (key,)).rowcount == 1


def _limits(adapter):
    """Operator limits: 1..1000 spoken characters and 1..60 seconds.

    Invalid values use conservative defaults; bools are not numeric settings.
    """
    options = (adapter.config.extra or {}).get('voice_reply', {})
    chars = options.get('max_chars', 180)
    seconds = options.get('max_seconds', 60)
    if type(chars) is not int or not 1 <= chars <= 1000:
        chars = 180
    if type(seconds) not in (int, float) or not 1 <= seconds <= 60:
        seconds = 60
    return chars, seconds


def _synthesize(text, policy, max_chars=180, max_seconds=60):
    from tools.tts_tool import text_to_speech_tool
    from tools.tts_text_normalize import prepare_spoken_text
    spoken = prepare_spoken_text(text, max_chars=None)
    if not spoken:
        raise ValueError('Empty spoken response')
    if sum(not char.isspace() for char in spoken) > max_chars:
        return []  # Intentional text-only reply, not a provider failure.
    if not policy.allows_tool('text_to_speech', {'text': spoken}):
        raise PermissionError('TTS denied by original requester policy')
    # An exclusive, internally chosen cache namespace proves ownership for stale
    # result cleanup. No user path or provider override is accepted here.
    from hermes_constants import get_hermes_dir
    cache = get_hermes_dir('cache/audio', 'audio_cache')
    Path(cache).mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='qq-auto-', dir=cache))
    audio = _AudioPaths([], directory, policy)
    try:
        data = json.loads(text_to_speech_tool(text=text, output_path=str(directory / 'speech.mp3')))
        paths = data.get('file_paths') or [data.get('file_path')]
        if data.get('success') is not True or not paths or not all(isinstance(p, str) and p for p in paths):
            raise ValueError('TTS returned no complete audio')
        for path in paths:
            if not policy.grant_artifact(path):
                raise PermissionError('Generated audio grant failed')
        audio = _AudioPaths(paths, directory, policy)
        from .voice_duration import audio_duration
        # Inspect ALL parts before the first send: never deliver a partial long reply.
        durations = [audio_duration(Path(path)) for path in audio]
        if sum(durations) > max_seconds:
            audio.discard()
            return []
        return audio
    except BaseException:
        audio.discard()
        raise
    finally:
        # A provider may return a shared/external file; it is never ours to delete.
        # In that case our empty reservation can still be removed.
        try:
            directory.rmdir()
        except OSError:
            pass


async def after_text(adapter, event, session_key, text, metadata=None, *, ephemeral=False):
    """Never let optional audio turn an acknowledged text into a replayable failure."""
    from agent.request_security import current_policy, policy_scope
    from gateway.request_security import policy_for_adapter
    from hermes_constants import get_hermes_home
    from gateway.session_identity import identity_of

    claimed = False
    job = _SynthesisJob()
    started = time.monotonic()
    # Correlate slow background synthesis without logging text, paths or platform IDs.
    trace = hashlib.sha256(str(event.message_id).encode()).hexdigest()[:16]
    try:
        if not _eligible(adapter, event, text, ephemeral, metadata):
            return
        if not _is_current(adapter, event, session_key):
            return
        with adapter._media_delivery_scope(event.source):
            source = event.source
            home = adapter._receipt_home
            if not isinstance(home, Path) or get_hermes_home().resolve() != home.resolve():
                return
            identity = identity_of(source)
            owner = getattr(adapter, '_owner_profile', None) or 'default'
            transport = identity.transport_profile if identity else (source.profile or owner)
            if transport != owner or session_key != adapter._source_session_key(source):
                return
            ref = getattr(source, '_transport_adapter_ref', None)
            original = ref() if callable(ref) else None
            if original is not None and original._app_id != adapter._app_id:
                return
            policy = policy_for_adapter(adapter, source)
            active = current_policy()
            if (policy is None or (policy.platform, policy.chat_id, policy.subject_id, policy.scope_id)
                    != ('qqbot', source.chat_id, source.user_id, owner)
                    or (active is not None and active.cache_key() != policy.cache_key())):
                return
            with policy_scope(policy):
                claimed = await asyncio.to_thread(_claim, home, [adapter._app_id, owner,
                    source.chat_type, source.chat_id, source.user_id, source.thread_id,
                    source.scope_id, session_key, event.message_id])
                if not claimed:
                    logger.info('QQ optional voice skipped event=%s reason=already_claimed', trace)
                    return
                logger.info('QQ optional voice started event=%s', trace)
                if not _is_current(adapter, event, session_key):
                    return
                # Media-only scope intentionally has no provider secrets. Hydrate
                # the captured source's runtime scope for this background synthesis.
                runner = getattr(adapter, 'gateway_runner', None)
                scope = (runner._async_profile_scope_for_source(source)
                         if runner is not None else nullcontext())
                async with scope:
                    context = copy_context()
                    paths = await asyncio.get_running_loop().run_in_executor(
                        _SYNTHESIS_EXECUTOR, context.run, job.run,
                        lambda: _is_current(adapter, event, session_key), text, policy, *_limits(adapter))
                if not _is_current(adapter, event, session_key):
                    job.discard()
                    return
                if not paths:
                    logger.info('QQ optional voice skipped event=%s reason=length_limit elapsed=%.2fs',
                                trace, time.monotonic() - started)
                    return
                for path in paths:
                    if not _is_current(adapter, event, session_key):
                        job.discard()
                        return
                    # The transport owns this part now. An unknown ACK or task
                    # cancellation must not delete an already-submitted artifact.
                    paths.owned.pop(Path(path).resolve(), None)
                    result = await _send_current_voice(adapter, event, session_key, path, metadata)
                    if result is None:
                        job.discard()
                        return
                    if not result.success:
                        raise RuntimeError(f'Native voice delivery failed or unknown: {result.error}')
                logger.info('QQ optional voice acknowledged event=%s parts=%d elapsed=%.2fs',
                            trace, len(paths), time.monotonic() - started)
    except asyncio.CancelledError:
        job.discard()
        raise
    except Exception as exc:
        if not _is_current(adapter, event, session_key):
            job.discard()
            return
        logger.warning('QQ optional voice unavailable: %s', type(exc).__name__, exc_info=True)
        if claimed and _is_current(adapter, event, session_key):
            try:
                with adapter._media_delivery_scope(event.source), policy_scope(policy):
                    # Plain send, not final delivery: never recurse or replay the original text.
                    await adapter.send(event.source.chat_id, FALLBACK, reply_to=event.message_id,
                                       metadata={'_interim_send': True})
            except Exception:
                logger.warning('QQ voice failure notice unavailable', exc_info=True)
