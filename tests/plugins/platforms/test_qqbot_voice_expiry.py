"""Automatic audio expires at execution start, not at an unrelated input arrival."""
import asyncio
import json
from pathlib import Path
import sqlite3
import threading
from unittest.mock import AsyncMock

import httpx
import pytest

from agent.request_security import current_policy
from gateway.platforms.base import SendResult
from plugins.platforms.qqbot.memes import capture_turn
from tests.plugins.platforms.test_qqbot_voice_reply import bot, event_for, final, wire
from tests.waiting import async_wait_until


async def start(bot, event):
    key = bot._source_session_key(event.source)
    await bot.prepare_turn_notes(event, 'session', key, [])
    return key


@pytest.mark.asyncio
@pytest.mark.parametrize('next_turn', ['same', 'other', 'arrived_only'])
@pytest.mark.parametrize('provider_failure', [False, True])
async def test_new_execution_expires_old_audio_but_other_session_or_arrival_does_not(
        bot, wire, monkeypatch, next_turn, provider_failure):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    files = []

    def synth(*, text, output_path):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(10), 'test never released provider'
        if provider_failure:
            raise RuntimeError('synthetic provider failure')
        path = Path(output_path).with_suffix('.silk')
        path.write_bytes(b'\x02#!SILK_V3\x02\x00xx')
        files.append(path)
        return json.dumps({'success': True, 'file_paths': [str(path)]})

    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', synth)
    calls, transport = wire
    first = event_for(bot, 'A')
    await start(bot, first)
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        task = asyncio.create_task(final(bot, first))
        try:
            await asyncio.wait_for(started.wait(), 10)
            second = event_for(bot, 'B')
            if next_turn == 'other':
                second.source = bot.build_source(chat_id='other', user_id='guest', chat_type='group')
                bot._chat_type_map['other'] = 'group'
                capture_turn(second)
            if next_turn != 'arrived_only':
                await start(bot, second)
                # Preparation reentry of the first executed event cannot revive it.
                await start(bot, first)
                await start(bot, event_for(bot, 'A'))
        finally:
            release.set()
        result, _ = await asyncio.wait_for(task, 10)
    assert result.success
    stale = next_turn == 'same'
    assert len(calls) == (1 if stale else 2)
    if not provider_failure:
        assert files[0].exists() is not stale
    if stale:
        assert bot._upload_local_file.await_count == 0
    with sqlite3.connect(bot._receipt_home / 'cache/qqbot/voice_replies.sqlite3') as db:
        assert db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == 1


@pytest.mark.asyncio
async def test_already_stale_final_never_starts_synthesis(bot, wire, monkeypatch):
    synth = AsyncMock(side_effect=AssertionError('stale audio reached provider'))
    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', synth)
    first, second = event_for(bot, 'A'), event_for(bot, 'B')
    await start(bot, first)
    await start(bot, second)
    calls, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        result, _ = await final(bot, first)
    assert result.success and len(calls) == 1
    synth.assert_not_called()


@pytest.mark.asyncio
async def test_new_turn_audio_is_not_deleted_when_old_provider_finishes(bot, wire, monkeypatch):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    files = {}
    def synth(*, text, output_path):
        if text == 'A':
            loop.call_soon_threadsafe(started.set)
            assert release.wait(10)
        path = Path(output_path).with_suffix('.silk')
        path.write_bytes(b'\x02#!SILK_V3\x02\x00xx')
        files[text] = path
        return json.dumps({'success': True, 'file_paths': [str(path)]})
    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', synth)
    first, second = event_for(bot, 'A'), event_for(bot, 'B')
    await start(bot, first)
    calls, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        old = asyncio.create_task(final(bot, first, 'A'))
        try:
            await asyncio.wait_for(started.wait(), 10)
            await start(bot, second)
            result, _ = await final(bot, second, 'B')
            assert result.success and files['B'].exists()
        finally:
            release.set()
        await asyncio.wait_for(old, 10)
    assert not files['A'].exists() and files['B'].exists()
    assert files['A'].parent != files['B'].parent
    assert [body['msg_type'] for _, body in calls] == [2, 2, 7]


@pytest.mark.asyncio
async def test_each_part_rechecks_generation_and_keeps_submitted_part(bot, wire, monkeypatch):
    files = []
    def synth(*, text, output_path):
        for number in range(2):
            path = Path(output_path).with_name(f'part-{number}.silk')
            path.write_bytes(b'\x02#!SILK_V3\x02\x00xx')
            files.append(path)
        return json.dumps({'success': True, 'file_paths': list(map(str, files))})
    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', synth)
    sent = []
    async def send_voice(**kwargs):
        sent.append(kwargs['audio_path'])
        await start(bot, event_for(bot, 'B'))
        return SendResult(success=True, message_id='submitted')
    monkeypatch.setattr(bot, 'send_voice', send_voice)
    _, transport = wire
    first = event_for(bot, 'A')
    await start(bot, first)
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        await final(bot, first)
    assert sent == [str(files[0])]
    assert files[0].exists() and not files[1].exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['upload', 'post'])
async def test_new_turn_cancels_awaited_old_transport_without_replay(bot, monkeypatch, stage):
    entered, reaped = asyncio.Event(), asyncio.Event()
    requests = []
    async def block():
        entered.set()
        try:
            await asyncio.Future()
        finally:
            reaped.set()
    async def upload(*args, **kwargs):
        assert current_policy().subject_id == 'guest'
        await block()
    if stage == 'upload':
        monkeypatch.setattr(bot, '_upload_local_file', upload)
    def synth(*, text, output_path):
        path = Path(output_path).with_suffix('.silk')
        path.write_bytes(b'\x02#!SILK_V3\x02\x00xx')
        return json.dumps({'success': True, 'file_paths': [str(path)]})
    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', synth)
    async def request(req):
        body = json.loads(req.content)
        requests.append(body)
        if body['msg_type'] == 7:
            await block()
        return httpx.Response(200, json={'id': 'text-ack'})
    first = event_for(bot, 'A')
    await start(bot, first)
    async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
        bot._http_client = client
        task = asyncio.create_task(final(bot, first))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            await start(bot, event_for(bot, 'B'))
            result, _ = await asyncio.wait_for(task, 10)
            assert result.success and reaped.is_set()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    # A posted request cannot be revoked, but there is no retry or fallback.
    assert [body['msg_type'] for body in requests] == ([2, 7] if stage == 'post' else [2])
    assert current_policy() is None


@pytest.mark.asyncio
async def test_cancelled_waiter_cleans_late_thread_result_only(bot, wire, monkeypatch, tmp_path):
    started, finished = asyncio.Event(), threading.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    files = []
    user_file = tmp_path / 'explicit.silk'
    user_file.write_bytes(b'user-owned')
    def synth(*, text, output_path):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(10)
        path = Path(output_path).with_suffix('.silk')
        path.write_bytes(b'\x02#!SILK_V3\x02\x00xx')
        files.append(path)
        return json.dumps({'success': True, 'file_paths': [str(path)]})
    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', synth)
    from plugins.platforms.qqbot import voice_reply
    original_run = voice_reply._SynthesisJob.run
    def run(job, *args):
        try:
            return original_run(job, *args)
        finally:
            finished.set()
    monkeypatch.setattr(voice_reply._SynthesisJob, 'run', run)
    calls, transport = wire
    first = event_for(bot, 'A')
    await start(bot, first)
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        task = asyncio.create_task(final(bot, first))
        try:
            await asyncio.wait_for(started.wait(), 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
        await async_wait_until(finished.is_set, timeout=10, what='cancelled synthesis cleanup')
    assert not files[0].exists() and user_file.read_bytes() == b'user-owned'
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('external', [False, True])
async def test_long_audio_discards_only_exclusive_generated_files(bot, wire, monkeypatch, tmp_path, external):
    from tests.plugins.platforms.test_qqbot_voice_reply import write_wav
    files = []
    def synth(*, text, output_path):
        path = tmp_path / 'shared.wav' if external else Path(output_path).with_suffix('.wav')
        write_wav(path, 61)
        files.append(path)
        return json.dumps({'success': True, 'file_paths': [str(path)]})
    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', synth)
    calls, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        await final(bot, event_for(bot))
    assert files[0].exists() is external
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_locked_unused_audio_logs_cleanup_failure_without_fallback(bot, wire, monkeypatch, caplog):
    from tests.plugins.platforms.test_qqbot_voice_reply import write_wav
    files = []
    def synth(*, text, output_path):
        path = Path(output_path).with_suffix('.wav')
        write_wav(path, 61)
        files.append(path.resolve())
        return json.dumps({'success': True, 'file_paths': [str(path)]})
    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', synth)
    original_unlink = Path.unlink
    def unlink(path, *args, **kwargs):
        if path.resolve() in files:
            raise PermissionError('synthetic file lock')
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', unlink)
    calls, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        await final(bot, event_for(bot))
    assert 'QQ unused voice cleanup failed: PermissionError' in caplog.text
    assert len(calls) == 1 and files[0].exists()


@pytest.mark.asyncio
async def test_real_fifo_starts_next_turn_while_previous_tts_waits(monkeypatch):
    from tests.gateway.test_qqbot_busy_fifo import make_pair, message
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    runner, adapter = make_pair(monkeypatch)
    del adapter.send  # Exercise native text/voice HTTP delivery, not the fixture's send stub.
    adapter.config.extra['voice_reply'] = {'enabled': True}
    adapter._chat_type_map['room'] = 'group'
    adapter._ensure_connected = AsyncMock(return_value=True)
    adapter._auth_headers = AsyncMock(return_value={})
    adapter._upload_local_file = AsyncMock(return_value={'file_info': 'voice'})
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='voice-expiry')
    runner._wire_adapter_handlers(adapter, busy_text_mode='queue')
    tts_started, second_started = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    files, requests = [], []

    def synth(*, text, output_path):
        loop.call_soon_threadsafe(tts_started.set)
        assert release.wait(15), 'FIFO could not start the next turn during TTS'
        path = Path(output_path).with_suffix('.silk')
        path.write_bytes(b'\x02#!SILK_V3\x02\x00xx')
        files.append(path)
        return json.dumps({'success': True, 'file_paths': [str(path)]})

    async def model(event, source, key, generation):
        # The real preparation hook is invoked at the model execution seam.
        await runner._hmwa_adapter_turn_notes(event, 'session', key, [])
        if event.text == 'B':
            second_started.set()
        response = '回复。' if event.text == 'A' else '长' * 181
        execution_result.get()['result'] = {'final_response': response, 'messages': []}
        return response

    def request(req):
        body = json.loads(req.content)
        requests.append(body)
        return httpx.Response(200, json={'id': f'ack-{len(requests)}'})

    monkeypatch.setattr(runner, '_handle_message_with_agent', model)
    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', synth)
    async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
        adapter._http_client = client
        try:
            first, second = message(adapter, 'A'), message(adapter, 'B', user='other')
            for event in (first, second):
                event.raw_message = {'id': event.message_id, 'content': event.text}
                capture_turn(event)
            await adapter.handle_message(first)
            await asyncio.wait_for(tts_started.wait(), 10)
            await adapter.handle_message(second)
            await asyncio.wait_for(second_started.wait(), 10)
            assert not release.is_set()
            release.set()
            await async_wait_until(lambda: not adapter._background_tasks, timeout=15, what='FIFO audio completion')
        finally:
            release.set()
            tasks = list(adapter._background_tasks) + [live.task for live in authority.sessions.values() if live.task]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    assert all(body['msg_type'] != 7 for body in requests)
    assert len(requests) == 2 and len(files) == 1 and not files[0].exists()
