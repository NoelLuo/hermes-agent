"""Text-first voice uses the real QQ adapter, policy and temporary durable journal."""
import asyncio
import hashlib
import json
import sqlite3
import threading
import wave
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from agent.request_security import current_policy, policy_scope, RequestSecurityPolicy
from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.request_security import policy_for_adapter
from gateway.run import GatewayRunner
from gateway.run_adapter_turns import finish_adapter_followup
from plugins.platforms.qqbot.adapter import QQAdapter, MEDIA_TYPE_VOICE
from plugins.platforms.qqbot.memes import capture_turn
from plugins.platforms.qqbot.voice_reply import FALLBACK


@pytest.fixture
def bot():
    adapter = QQAdapter(PlatformConfig(enabled=True, extra={
        'app_id': 'voice-test', 'markdown_support': True,
        'role_policy': {'enabled': True}, 'voice_reply': {'enabled': True}}))
    adapter._chat_type_map['room'] = 'group'
    adapter._ensure_connected = AsyncMock(return_value=True)
    adapter._auth_headers = AsyncMock(return_value={})
    adapter._upload_local_file = AsyncMock(return_value={'file_info': 'test-upload'})
    return adapter


def event_for(bot, identifier='input-1'):
    event = MessageEvent(source=bot.build_source(chat_id='room', user_id='guest', chat_type='group'),
        text='你好', message_id=identifier, raw_message={'id': identifier, 'content': '你好'})
    capture_turn(event)
    return event


@pytest.fixture
def wire():
    calls = []
    def handle(req):
        calls.append((req.url.path, json.loads(req.content)))
        return httpx.Response(200, json={'id': f'ack-{len(calls)}'})
    return calls, httpx.MockTransport(handle)


@pytest.fixture
def tts(monkeypatch, tmp_path):
    calls = []
    def synth(*, text, output_path):
        policy = current_policy()
        calls.append((text, policy, threading.get_ident()))
        path = Path(output_path).with_suffix('.silk')
        path.write_bytes(b'\x02#!SILK_V3\x02\x00xx')
        return json.dumps({'success': True, 'file_paths': [str(path)]})
    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', synth)
    return calls


async def final(bot, event, text='**你好**，这是回复。', *, ephemeral=False, metadata=None):
    key = bot._source_session_key(event.source)
    extracted = await bot._extract_response_content(text, event, key, is_ephemeral_response=ephemeral)
    return await bot.send_final_ledgered(event, key, extracted.text_content, metadata or {},
        reply_to=event.message_id, is_ephemeral_response=ephemeral)


@pytest.mark.asyncio
async def test_guest_text_before_voice_scoped_grant_and_current_chat(bot, wire, tts, tmp_path):
    calls, transport = wire
    event = event_for(bot)
    policy = policy_for_adapter(bot, event.source)
    private = tmp_path / 'private.silk'
    private.write_bytes(b'private')
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        result, owner = await final(bot, event)
        generated = bot._upload_local_file.call_args.args[2]
        with policy_scope(policy):
            denied = await bot.send_voice('other-room', generated)
    assert result.success and owner is bot
    assert [body['msg_type'] for _, body in calls] == [2, 7]
    assert all(path == '/v2/groups/room/messages' for path, _ in calls)
    assert bot._upload_local_file.call_args.args[3] == MEDIA_TYPE_VOICE
    assert tts[0][0] == '**你好**，这是回复。'
    assert tts[0][1] is policy and not policy.is_admin
    assert tts[0][2] != threading.get_ident()
    assert policy.allows_media(generated) and not policy.allows_media(private)
    assert not denied.success and current_policy() is None


@pytest.mark.asyncio
async def test_full_text_and_every_generated_part(bot, wire, monkeypatch, tmp_path):
    seen = []
    paths = [tmp_path / f'part-{n}.silk' for n in range(2)]
    for path in paths:
        path.write_bytes(b'\x02#!SILK_V3\x02\x00xx')
    def synth(*, text, output_path):
        seen.append(text)
        return json.dumps({'success': True, 'file_paths': [str(p) for p in paths]})
    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', synth)
    text = '完整内容。' * 20 + '结尾仍在。'
    calls, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        await final(bot, event_for(bot), text)
    assert seen == [text]
    assert bot._upload_local_file.await_count == 2
    assert [body['msg_type'] for _, body in calls][-2:] == [7, 7]


@pytest.mark.asyncio
async def test_durable_repeat_and_new_actual_turn(bot, wire, tts):
    _, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        await final(bot, event_for(bot))
        # Reconstructed event has no in-memory delivery state.
        await final(bot, event_for(bot))
        await final(bot, event_for(bot, 'input-2'))
    assert len(tts) == 2
    with sqlite3.connect(bot._receipt_home / 'cache/qqbot/voice_replies.sqlite3') as db:
        assert db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == 2


@pytest.mark.asyncio
async def test_restored_final_logs_voice_ack_without_disclosing_input(bot, wire, tts, caplog):
    # Production restores native admissions and delivers under the request policy.
    event = event_for(bot, 'private-platform-event-id')
    stamp = bot.snapshot_native_turn(event)
    del event._qq_meme_turn
    bot.restore_native_turn(event, stamp)
    policy = policy_for_adapter(bot, event.source)
    _, transport = wire
    caplog.set_level('INFO', logger='plugins.platforms.qqbot.voice_reply')
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        with policy_scope(policy):
            result, _ = await final(bot, event, '仅供本轮朗读的正文')
            await final(bot, event, '仅供本轮朗读的正文')
    assert result.success and len(tts) == 1
    messages = [r.getMessage() for r in caplog.records
                if r.name == 'plugins.platforms.qqbot.voice_reply']
    trace = hashlib.sha256(event.message_id.encode()).hexdigest()[:16]
    assert any(f'started event={trace}' in m for m in messages)
    assert any(f'acknowledged event={trace} parts=1 elapsed=' in m for m in messages)
    assert any('reason=already_claimed' in m for m in messages)
    assert all(event.message_id not in m and '仅供本轮朗读的正文' not in m for m in messages)


@pytest.mark.parametrize('case', ['disabled', 'internal', 'ephemeral', 'slash', 'raw-slash',
    'interim', 'unstamped', 'chained', 'target', 'profile', 'policy', 'explicit-audio'])
@pytest.mark.asyncio
async def test_excluded_finals_remain_text_only(bot, wire, tts, tmp_path, case):
    event = event_for(bot)
    text, metadata = '正常文字', {}
    if case == 'disabled':
        bot.config.extra['voice_reply']['enabled'] = False
    elif case == 'internal':
        event.internal = True
    elif case == 'slash':
        event.text = '/status'
    elif case == 'raw-slash':
        event._qq_meme_turn.input_text = '/status'
    elif case == 'interim':
        metadata['_interim_send'] = True
    elif case == 'unstamped':
        del event._qq_meme_turn
    elif case == 'chained':
        event._adapter_chained_response = True
    elif case == 'target':
        event.source.user_id = 'different-guest'
    elif case == 'profile':
        bot._receipt_home = tmp_path / 'different-home'
    elif case == 'explicit-audio':
        path = tmp_path / 'explicit.silk'
        path.write_bytes(b'\x02#!SILK_V3\x02\x00xx')
        policy_for_adapter(bot, event.source).grant_artifact(str(path))
        text += f'\n[[audio_as_voice]]\nMEDIA:{path}'
    calls, transport = wire
    other = RequestSecurityPolicy('other', 'qqbot', 'elsewhere') if case == 'policy' else None
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        with policy_scope(other):
            await final(bot, event, text, ephemeral=case == 'ephemeral', metadata=metadata)
    assert not tts
    assert not any(body['msg_type'] == 7 for _, body in calls)


@pytest.mark.parametrize('failure', ['synth', 'upload'])
@pytest.mark.asyncio
async def test_optional_failure_preserves_text_and_never_retries_voice(bot, wire, tts, monkeypatch, failure):
    if failure == 'synth':
        def fail(**kwargs):
            raise RuntimeError('synthetic provider error')
        monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', fail)
    else:
        bot._upload_local_file.side_effect = RuntimeError('lost upload ACK')
    calls, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        result, _ = await final(bot, event_for(bot))
        assert result.success and result.message_id == 'ack-1'
        await final(bot, event_for(bot))
    text = [body.get('markdown', {}).get('content', body.get('content', '')) for _, body in calls]
    assert text.count(FALLBACK) == 1
    assert bot._upload_local_file.await_count == (1 if failure == 'upload' else 0)


@pytest.mark.asyncio
async def test_normal_queued_first_and_followup_use_same_hook(bot, wire, tts):
    calls, transport = wire
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: bot
    event = event_for(bot)
    key = bot._source_session_key(event.source)
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        handled = await runner._deliver_queued_first_response('排队第一条', event.source, bot,
            session_key=key, inbound_event=event)
        followup = event_for(bot, 'input-followup')
        result = await finish_adapter_followup(runner, {'final_response': '后续一条'},
            followup, followup.source, key, {})
    assert handled and result['already_sent'] and result['media_already_delivered']
    assert [body['msg_type'] for _, body in calls] == [2, 7, 2, 7]
    assert len(tts) == 2


@pytest.mark.asyncio
async def test_queued_ephemeral_and_acknowledged_finals_no_voice(bot, wire, tts):
    calls, transport = wire
    event = event_for(bot)
    key = bot._source_session_key(event.source)
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        assert await bot.deliver_queued_response('已发送', event, key, text_already_delivered=True)
        assert await bot.deliver_queued_response('临时状态', event, key, deliver_media=False)
    assert len(calls) == 1 and not tts


@pytest.mark.asyncio
async def test_concurrent_claim_only_one_voice(bot, wire, tts):
    _, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        await asyncio.gather(final(bot, event_for(bot)), final(bot, event_for(bot)))
    assert len(tts) == 1


@pytest.mark.asyncio
async def test_text_network_failure_does_not_synthesize(bot, tts):
    def refused(req):
        return httpx.Response(403, json={'message': 'synthetic refusal'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(refused)) as client:
        bot._http_client = client
        result, _ = await final(bot, event_for(bot))
    assert not result.success and not tts


@pytest.mark.asyncio
async def test_journal_unavailable_fails_closed_without_text_replay(bot, wire, tts):
    directory = bot._receipt_home / 'cache/qqbot'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'voice_replies.sqlite3').mkdir()
    calls, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        result, _ = await final(bot, event_for(bot))
    assert result.success and len(calls) == 1 and not tts


@pytest.mark.asyncio
async def test_restored_native_stamp_and_guest_dm(bot, wire, tts):
    bot._chat_type_map['guest'] = 'c2c'
    event = MessageEvent(source=bot.build_source(chat_id='guest', user_id='guest', chat_type='dm'),
        text='你好', message_id='dm-input', raw_message={'id': 'dm-input', 'content': '你好'})
    capture_turn(event)
    state = bot.snapshot_native_turn(event)
    del event._qq_meme_turn
    bot.restore_native_turn(event, state)
    calls, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        result, _ = await final(bot, event)
    assert result.success and len(tts) == 1
    assert [body['msg_type'] for _, body in calls] == [2, 7]
    assert all(path == '/v2/users/guest/messages' for path, _ in calls)


@pytest.mark.asyncio
async def test_base_voice_first_is_disabled_only_when_qq_voice_enabled(bot):
    from gateway.platforms.event import MessageType
    event = event_for(bot)
    event.message_type = MessageType.VOICE
    bot._auto_tts_default = True
    key = bot._source_session_key(event.source)
    assert not bot._wants_auto_tts(event, key, asyncio.Event(), '回复', [])
    bot.config.extra['voice_reply']['enabled'] = False
    assert bot._wants_auto_tts(event, key, asyncio.Event(), '回复', [])


@pytest.mark.parametrize('count, expected', [(180, 1), (181, 0), (8000, 0)])
@pytest.mark.asyncio
async def test_normalized_character_limit_is_inclusive_without_truncation(bot, wire, tts, count, expected):
    calls, transport = wire
    text = '**' + '你 ' * count + '**'
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        result, _ = await final(bot, event_for(bot), text)
    assert result.success
    assert len(tts) == expected
    if expected:
        assert tts[0][0] == text  # The tool owns the single normalization pass.
    assert bot._upload_local_file.await_count == expected
    assert not any(body.get('markdown', {}).get('content', body.get('content', '')) == FALLBACK for _, body in calls)


def write_wav(path, seconds):
    with wave.open(str(path), 'wb') as audio:
        audio.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
        audio.writeframes(b'\x00\x00' * round(seconds * 8000))


@pytest.mark.parametrize('durations, limit, expected', [
    ([60], 60, 1), ([60.001], 60, 0), ([30, 30], 60, 2),
    ([30, 30.001], 60, 0), ([10, 11], 20, 0), ([20], 20, 1)])
@pytest.mark.asyncio
async def test_complete_actual_wav_duration_checked_before_any_send(
        bot, wire, monkeypatch, tmp_path, durations, limit, expected):
    from gateway.platforms.base import SendResult
    paths = [tmp_path / f'part-{n}.wav' for n in range(len(durations))]
    for path, duration in zip(paths, durations):
        write_wav(path, duration)
    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', lambda **kwargs:
        json.dumps({'success': True, 'file_paths': list(map(str, paths))}))
    # Send boundary only: normalization, policy/grants, journal and duration are real.
    bot.send_voice = AsyncMock(return_value=SendResult(success=True))
    bot.config.extra['voice_reply']['max_seconds'] = limit
    calls, transport = wire
    event = event_for(bot)
    policy = policy_for_adapter(bot, event.source)
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        result, _ = await final(bot, event)
    assert result.success and bot.send_voice.await_count == expected
    assert len(calls) == 1  # No failure notice for an intentional skip.
    assert all(policy.allows_media(str(path)) for path in paths)


@pytest.mark.parametrize('kind', ['invalid', 'truncated', 'empty', 'unknown'])
@pytest.mark.asyncio
async def test_invalid_duration_fails_closed_before_any_chunk(bot, wire, monkeypatch, tmp_path, kind):
    good, bad = tmp_path / 'good.wav', tmp_path / 'bad.wav'
    write_wav(good, 1)
    if kind == 'invalid':
        bad.write_bytes(b'not audio')
    elif kind == 'truncated':
        write_wav(bad, 1)
        bad.write_bytes(bad.read_bytes()[:-2])
    elif kind == 'empty':
        write_wav(bad, 0)
    else:
        bad = bad.with_suffix('.mp3')
        bad.write_bytes(b'not audio')
        monkeypatch.setattr('tools.transcription_audio._probe_audio_duration', lambda _: None)
    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', lambda **kwargs:
        json.dumps({'success': True, 'file_paths': [str(good), str(bad)]}))
    bot.send_voice = AsyncMock()
    calls, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        result, _ = await final(bot, event_for(bot))
        await final(bot, event_for(bot))
    assert result.success and not bot.send_voice.called
    assert sum(body.get('markdown', {}).get('content', body.get('content', '')) == FALLBACK for _, body in calls) == 1


@pytest.mark.parametrize('options, expected', [
    ({}, (180, 60)), ({'max_chars': 10, 'max_seconds': 1.5}, (10, 1.5)),
    ({'max_chars': True, 'max_seconds': True}, (180, 60)),
    ({'max_chars': 0, 'max_seconds': 0}, (180, 60)),
    ({'max_chars': 1001, 'max_seconds': 61}, (180, 60)),
    ({'max_chars': '50', 'max_seconds': '20'}, (180, 60)),
    ({'max_chars': 5.5, 'max_seconds': float('nan')}, (180, 60)),
    ({'max_seconds': float('inf')}, (180, 60))])
def test_operator_limits_are_bounded(bot, options, expected):
    from plugins.platforms.qqbot.voice_reply import _limits
    bot.config.extra['voice_reply'].update(options)
    assert _limits(bot) == expected


@pytest.mark.asyncio
async def test_custom_character_limit_and_skip_remain_durable(bot, wire, tts):
    bot.config.extra['voice_reply']['max_chars'] = 2
    calls, transport = wire
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        await final(bot, event_for(bot), '三个字')
        bot.config.extra['voice_reply']['max_chars'] = 180
        await final(bot, event_for(bot), '三个字')
        await final(bot, event_for(bot, 'next'), '两个')
    assert len(tts) == 1 and tts[0][0] == '两个'
    assert not any(body.get('markdown', {}).get('content', body.get('content', '')) == FALLBACK for _, body in calls)
