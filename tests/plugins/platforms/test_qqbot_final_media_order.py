"""Final delivery ordering through real QQ hooks, policy, journals and mock HTTP/TTS."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agent.request_security import current_policy, policy_scope
from gateway.request_security import policy_for_adapter
from gateway.run_adapter_turns import finish_adapter_followup
from plugins.platforms.qqbot.memes import controller_for
from tests.plugins.platforms.test_qqbot_memes import bot, make_event
from tests.plugins.platforms.test_qqbot_meme_expression import emit, prepare


@pytest.fixture
def ordered(bot, monkeypatch, tmp_path):
    bot.config.extra['voice_reply'] = {'enabled': True}
    events = []

    async def upload(*args, **kwargs):
        path = Path(args[2])
        kind = 'voice' if path.suffix == '.silk' else 'meme'
        assert current_policy().allows_media(path)
        events.append(kind + '-upload')
        return {'file_info': kind}

    def synth(*, text, output_path):
        events.append('tts')
        assert current_policy().subject_id == 'visitor'
        path = Path(output_path).with_suffix('.silk')
        path.write_bytes(b'\x02#!SILK_V3\x02\x00xx')
        return json.dumps({'success': True, 'file_paths': [str(path)]})

    def request(req):
        body = json.loads(req.content)
        kind = body.get('media', {}).get('file_info', 'text')
        events.append(kind)
        return httpx.Response(200, json={'id': f'ack-{len(events)}'})

    monkeypatch.setattr(bot, '_upload_local_file', upload)
    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', synth)
    return bot, events, httpx.MockTransport(request)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['expression', 'auto'])
@pytest.mark.parametrize('lane', ['normal', 'queued', 'followup'])
async def test_text_meme_then_synthesis_and_voice(ordered, mode, lane):
    bot, events, transport = ordered
    ctrl = controller_for(bot)
    ctrl.settings = replace(ctrl.settings, mode=mode)
    event = make_event(bot)
    text = '你好。' + ('\n[[qq_meme_optional:yes]]' if mode == 'expression' else '')
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        if lane == 'normal':
            _, results = await emit(bot, event, text)
            assert results[0].success
        else:
            key, _ = await prepare(bot, event)
            if lane == 'queued':
                assert await bot.deliver_queued_response(text, event, key)
            else:
                runner = SimpleNamespace(_delivery_adapter_for=lambda source: bot)
                result = await finish_adapter_followup(runner, {'final_response': text},
                    event, event.source, key, {})
                assert result['already_sent'] and result['media_already_delivered']
    assert events == ['text', 'meme-upload', 'meme', 'tts', 'voice-upload', 'voice']
    assert current_policy() is None


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['expression', 'auto'])
async def test_meme_upload_failure_still_sends_voice_without_replaying_text(ordered, monkeypatch, mode):
    bot, events, transport = ordered
    ctrl = controller_for(bot)
    ctrl.settings = replace(ctrl.settings, mode=mode)
    upload = bot._upload_local_file

    async def fail_image(*args, **kwargs):
        if Path(args[2]).suffix == '.gif':
            events.append('meme-failed')
            raise OSError('synthetic upload failure')
        return await upload(*args, **kwargs)

    monkeypatch.setattr(bot, '_upload_local_file', fail_image)
    text = '你好。' + ('\n[[qq_meme_optional:yes]]' if mode == 'expression' else '')
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        _, results = await emit(bot, make_event(bot), text)
    assert results[0].success
    assert events == ['text', 'meme-failed', 'tts', 'voice-upload', 'voice']


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['expression', 'auto'])
async def test_disabled_meme_does_not_disable_voice(ordered, mode):
    bot, events, transport = ordered
    ctrl = controller_for(bot)
    ctrl.settings = replace(ctrl.settings, mode=mode, enabled=False)
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        await emit(bot, make_event(bot), '你好。\n[[qq_meme_optional:yes]]' if mode == 'expression' else '你好。')
    assert events == ['text', 'tts', 'voice-upload', 'voice']


@pytest.mark.asyncio
async def test_pure_expression_has_no_auto_voice(ordered):
    bot, events, transport = ordered
    ctrl = controller_for(bot)
    ctrl.settings = replace(ctrl.settings, mode='expression')
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        await emit(bot, make_event(bot), '[[qq_meme_optional:yes]]')
    assert events == ['meme-upload', 'meme']


@pytest.mark.asyncio
async def test_attachment_and_extraction_reentry_do_not_repeat_meme_or_voice(ordered):
    bot, events, transport = ordered
    ctrl = controller_for(bot)
    ctrl.settings = replace(ctrl.settings, mode='expression')
    event = make_event(bot)
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        await emit(bot, event, '你好。\n[[qq_meme_optional:yes]]')
        before = list(events)
        key = bot._source_session_key(event.source)
        for _ in range(2):
            extracted = await bot._extract_response_content('你好。\n[[qq_meme_optional:yes]]', event, key,
                is_ephemeral_response=False)
            await bot._deliver_attachments(event, extracted, {}, anything_sent=True, record_delivery=lambda r: None)
        assert events == before
        # Reconstructed retry remains claimed by the existing durable guards.
        await emit(bot, make_event(bot), '你好。\n[[qq_meme_optional:yes]]')
    assert events.count('meme') == events.count('tts') == events.count('voice') == 1


@pytest.mark.asyncio
async def test_explicit_voice_attachment_follows_expression_without_auto_tts(ordered, tmp_path):
    bot, events, transport = ordered
    ctrl = controller_for(bot)
    ctrl.settings = replace(ctrl.settings, mode='expression')
    event = make_event(bot)
    audio = tmp_path / 'explicit.silk'
    audio.write_bytes(b'\x02#!SILK_V3\x02\x00xx')
    policy = policy_for_adapter(bot, event.source)
    with policy_scope(policy):
        assert policy.grant_artifact(audio)
        async with httpx.AsyncClient(transport=transport) as client:
            bot._http_client = client
            await emit(bot, event, f'你好。\nMEDIA:{audio}\n[[qq_meme_optional:yes]]')
    # Explicit audio currently makes a decorated response ineligible for a GIF;
    # it must never synthesize a duplicate automatic voice.
    assert 'tts' not in events
    assert events[-1] == 'voice'
    if 'meme' in events:
        assert events.index('meme') < events.index('voice')


@pytest.mark.asyncio
async def test_unexpected_expression_failure_cleans_snapshot_and_does_not_retry(ordered, monkeypatch):
    from plugins.platforms.qqbot.meme_expression_delivery import engine_for
    bot, events, transport = ordered
    ctrl = controller_for(bot)
    ctrl.settings = replace(ctrl.settings, mode='expression')
    async def fail(*args):
        events.append('meme-failed')
        raise OSError('synthetic journal failure')
    monkeypatch.setattr(engine_for(bot), 'deliver', fail)
    async with httpx.AsyncClient(transport=transport) as client:
        bot._http_client = client
        event = make_event(bot)
        _, results = await emit(bot, event, '你好。\n[[qq_meme_optional:yes]]')
    assert results[0].success
    assert events == ['text', 'meme-failed', 'tts', 'voice-upload', 'voice']
    assert event._qq_expression_delivery.snapshot is None


def test_auto_voice_passes_original_code_to_real_tool_and_limits_spoken_script(monkeypatch):
    import wave
    from agent.request_security import RequestSecurityPolicy
    from plugins.platforms.qqbot.voice_reply import _synthesize
    from tools import tts_tool
    from tools.tts_text_normalize import prepare_spoken_text

    raw = '请读代码：`**literal**`。'
    spoken = prepare_spoken_text(raw, max_chars=None)
    seen = []
    config = {'provider': 'test-speech', 'providers': {'test-speech': {
        'type': 'command', 'command': 'unused', 'output_format': 'wav',
        'voice_compatible': True, 'preserve_source_audio': True}}}
    monkeypatch.setattr(tts_tool, '_load_tts_config', lambda: config)
    monkeypatch.setattr(tts_tool, '_session_platform', lambda: ('qqbot', False))
    def generate(text, output_path, *args):
        seen.append(text)
        with wave.open(output_path, 'wb') as audio:
            audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            audio.writeframes(b'\x00\x00' * 2400)
        return output_path
    monkeypatch.setattr(tts_tool, '_generate_command_tts', generate)
    policy = RequestSecurityPolicy(subject_id='guest', platform='qqbot', chat_id='room', scope_id='test')
    limit = sum(not c.isspace() for c in spoken)
    with policy_scope(policy):
        paths = _synthesize(raw, policy, max_chars=limit)
        assert paths
        assert _synthesize(raw, policy, max_chars=limit - 1) == []
    assert seen == [spoken]
    assert '**literal**' in seen[0]
