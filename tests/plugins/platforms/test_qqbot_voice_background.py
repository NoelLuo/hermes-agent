"""Optional QQ voice must neither own the reply lane nor outlive a newer input."""
import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from gateway.session_authority import initialize_session_authority
from gateway.session_results import execution_result
from gateway.run_adapter_turns import finish_adapter_followup
from plugins.platforms.qqbot import voice_reply
from plugins.platforms.qqbot.memes import capture_turn
from tests.gateway.test_qqbot_busy_fifo import make_pair, message
from tests.plugins.platforms.test_qqbot_voice_reply import bot, event_for, wire
from tests.waiting import async_wait_until


async def submit(bot, event, text='回复。'):
    key = bot._source_session_key(event.source)
    extracted = await bot._extract_response_content(text, event, key, is_ephemeral_response=False)
    return await bot.send_final_ledgered(event, key, extracted.text_content, {}, reply_to=event.message_id)


@pytest_asyncio.fixture
async def blocked_synthesis(monkeypatch):
    loop = asyncio.get_running_loop()
    started, finished = asyncio.Event(), threading.Event()
    release = threading.Event()
    files = []

    def synth(*, text, output_path):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(20), 'test did not release synthesis'
        path = Path(output_path).with_suffix('.silk')
        path.write_bytes(b'\x02#!SILK_V3\x02\x00xx')
        files.append(path)
        return json.dumps({'success': True, 'file_paths': [str(path)]})

    original = voice_reply._SynthesisJob.run

    def run(job, *args):
        try:
            return original(job, *args)
        finally:
            finished.set()

    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', synth)
    monkeypatch.setattr(voice_reply._SynthesisJob, 'run', run)
    yield started, release, finished, files
    release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize('lane', ['final', 'queued', 'followup'])
async def test_final_ack_returns_before_slow_synthesis(bot, wire, blocked_synthesis, lane):
    started, release, finished, _ = blocked_synthesis
    calls, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        event = event_for(bot)
        key = bot._source_session_key(event.source)

        async def deliver():
            if lane == 'final':
                result, _ = await submit(bot, event)
                return result.success
            if lane == 'queued':
                return await bot.deliver_queued_response('回复。', event, key)
            from types import SimpleNamespace
            runner = SimpleNamespace(_delivery_adapter_for=lambda source: bot)
            result = await finish_adapter_followup(runner, {'final_response': '回复。'},
                                                   event, event.source, key, {})
            return result['already_sent']

        task = asyncio.create_task(deliver())
        try:
            await asyncio.wait_for(started.wait(), 10)
            success = await asyncio.wait_for(asyncio.shield(task), 2)
            assert success and not release.is_set()
            assert len(calls) == 1
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await async_wait_until(finished.is_set, timeout=10, what='provider exit')
            await asyncio.gather(*getattr(bot, '_qq_voice_tasks', ()), return_exceptions=True)
    assert [body['msg_type'] for _, body in calls] == [2, 7]


@pytest.mark.asyncio
async def test_real_intake_expires_old_voice_before_next_execution_prepares(monkeypatch, blocked_synthesis):
    runner, adapter = make_pair(monkeypatch)
    del adapter.send
    adapter.config.extra['voice_reply'] = {'enabled': True}
    adapter._chat_type_map['room'] = 'group'
    adapter._ensure_connected = AsyncMock(return_value=True)
    adapter._auth_headers = AsyncMock(return_value={})
    adapter._upload_local_file = AsyncMock(return_value={'file_info': 'voice'})
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='voice-intake')
    runner._wire_adapter_handlers(adapter, busy_text_mode='queue')
    started, release, finished, files = blocked_synthesis
    second_entered, prepare_second = asyncio.Event(), asyncio.Event()
    requests = []

    async def model(event, source, key, generation):
        if event.text == 'B':
            second_entered.set()
            await prepare_second.wait()
        await runner._hmwa_adapter_turn_notes(event, 'session', key, [])
        response = '回复。' if event.text == 'A' else '长' * 181
        execution_result.get()['result'] = {'final_response': response, 'messages': []}
        return response

    def request(req):
        body = json.loads(req.content)
        requests.append(body)
        return httpx.Response(200, json={'id': f'ack-{len(requests)}'})

    monkeypatch.setattr(runner, '_handle_message_with_agent', model)
    async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
        adapter._http_client = client
        try:
            first, second = message(adapter, 'A'), message(adapter, 'B', user='other')
            for event in (first, second):
                event.raw_message = {'id': event.message_id, 'content': event.text}
                capture_turn(event)
            await adapter.handle_message(first)
            await asyncio.wait_for(started.wait(), 10)
            await async_wait_until(lambda: not adapter._active_sessions, timeout=2, what='text lane release')
            await adapter.handle_message(second)
            await asyncio.wait_for(second_entered.wait(), 10)
            release.set()  # B has arrived, but has NOT run prepare_turn_notes.
            await async_wait_until(finished.is_set, timeout=10, what='old provider completion')
            await asyncio.gather(*getattr(adapter, '_qq_voice_tasks', ()), return_exceptions=True)
            assert files and not files[0].exists()
            assert all(body['msg_type'] != 7 for body in requests)
            adapter._upload_local_file.assert_not_awaited()
        finally:
            release.set()
            prepare_second.set()
            tasks = list(adapter._background_tasks) + [live.task for live in authority.sessions.values() if live.task]
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(*getattr(adapter, '_qq_voice_tasks', ()), return_exceptions=True)
    assert [body['msg_type'] for body in requests] == [2, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['same', 'other', 'unauthorized', 'unstamped', 'internal',
                                 'duplicate', 'no-handler', 'denied-command', 'wrong-route'])
async def test_actual_handle_message_only_invalidates_same_authorized_lane(
        bot, wire, blocked_synthesis, case):
    started, release, finished, files = blocked_synthesis
    first = event_for(bot, 'A')
    key = bot._source_session_key(first.source)
    await bot.prepare_turn_notes(first, 'session', key, [])
    bot.set_message_handler(AsyncMock(return_value=None))
    bot.set_authorization_check(lambda *args, **kwargs: case != 'unauthorized')
    incoming = event_for(bot, 'A' if case == 'duplicate' else 'B')
    if case == 'other':
        incoming.source = bot.build_source(chat_id='other', user_id='guest', chat_type='group')
        bot._chat_type_map['other'] = 'group'
        capture_turn(incoming)
    elif case == 'unstamped':
        del incoming._qq_meme_turn
    elif case == 'internal':
        incoming.internal = True
    elif case == 'no-handler':
        bot.set_message_handler(None)
    elif case == 'denied-command':
        incoming.text = '/stop'
    elif case == 'wrong-route':
        incoming.metadata = {'gateway_session_key': 'not-this-conversation'}
    calls, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        try:
            await submit(bot, first)
            await asyncio.wait_for(started.wait(), 10)
            await bot.handle_message(incoming)
        finally:
            release.set()
            await async_wait_until(finished.is_set, timeout=10, what='scoped provider completion')
            await asyncio.gather(*bot._background_tasks, return_exceptions=True)
            await asyncio.gather(*getattr(bot, '_qq_voice_tasks', ()), return_exceptions=True)
    stale = case == 'same'
    assert files[0].exists() is not stale
    assert sum(body['msg_type'] == 7 for _, body in calls) == (0 if stale else 1)


@pytest.mark.asyncio
async def test_disconnect_reaps_audio_waiter_without_waiting_for_provider(bot, wire, blocked_synthesis):
    started, release, finished, files = blocked_synthesis
    calls, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        await submit(bot, event_for(bot))
        await asyncio.wait_for(started.wait(), 10)
        try:
            await asyncio.wait_for(bot.disconnect(), 2)
            assert not bot._qq_voice_tasks
            assert not finished.is_set()
        finally:
            release.set()
            await async_wait_until(finished.is_set, timeout=10, what='disconnected provider cleanup')
    assert files and not files[0].exists()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_intake_cancels_even_while_busy_admission_is_waiting(bot, wire, blocked_synthesis):
    started, release, finished, files = blocked_synthesis
    first = event_for(bot, 'A')
    key = bot._source_session_key(first.source)
    await bot.prepare_turn_notes(first, 'session', key, [])
    bot.set_message_handler(AsyncMock(return_value=None))
    bot.set_authorization_check(lambda *args, **kwargs: True)
    entered, unblock = asyncio.Event(), asyncio.Event()

    async def busy(event, session_key):
        entered.set()
        await unblock.wait()
        return True

    bot._busy_session_handler = busy
    bot._active_sessions[key] = asyncio.Event()
    _, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        await submit(bot, first)
        await asyncio.wait_for(started.wait(), 10)
        intake = asyncio.create_task(bot.handle_message(event_for(bot, 'B')))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            await async_wait_until(lambda: not bot._qq_voice_tasks, timeout=2, what='immediate voice cancellation')
            assert not intake.done() and not release.is_set()
        finally:
            unblock.set()
            release.set()
            await asyncio.gather(intake, return_exceptions=True)
            await async_wait_until(finished.is_set, timeout=10, what='busy admission cleanup')
    assert files and not files[0].exists()
    bot._upload_local_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_known_intake_cannot_be_superseded_by_old_or_restored_preparation(bot, wire, monkeypatch):
    from tests.plugins.platforms.test_qqbot_voice_reply import drain_voice
    bot.set_message_handler(AsyncMock(return_value=None))
    bot.set_authorization_check(lambda *args, **kwargs: True)
    events = [event_for(bot, identifier) for identifier in ('A', 'B', 'C')]
    for event in events:
        await bot.handle_message(event)
    await asyncio.gather(*bot._background_tasks, return_exceptions=True)
    synthesized = []

    def synth(*, text, output_path):
        synthesized.append(text)
        path = Path(output_path).with_suffix('.silk')
        path.write_bytes(b'\x02#!SILK_V3\x02\x00xx')
        return json.dumps({'success': True, 'file_paths': [str(path)]})

    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', synth)
    calls, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        for event in [event_for(bot, 'restored-old'), *events]:
            key = bot._source_session_key(event.source)
            await bot.prepare_turn_notes(event, 'session', key, [])
            await submit(bot, event, event.message_id)
            await drain_voice(bot)
    assert synthesized == ['C']
    assert [body['msg_type'] for _, body in calls] == [2, 2, 2, 2, 7]
