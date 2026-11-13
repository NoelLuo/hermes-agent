"""Configured command voices/languages reach the real TTS registry without config mutation."""
import copy
import json
import wave
from pathlib import Path
from unittest.mock import Mock

import pytest

from tools import tts_tool
from tools.registry import registry
from tools.tts_command_options import apply_options
from agent.request_security import RequestSecurityPolicy


@pytest.fixture
def configured(monkeypatch):
    config = {'provider': 'local-speech', 'providers': {'local-speech': {
        'type': 'command', 'command': 'synth --voice {voice} --language {language}',
        'voice': 'arona-cn', 'language': 'auto', 'output_format': 'wav', 'voice_compatible': True,
        'preserve_source_audio': True,
        'voices': {'arona-cn': 'arona-cn', '阿洛娜': 'arona-cn', '普拉娜': 'plana-jp'},
        'languages': ['auto', 'all_zh', 'zh', 'en', 'all_ja']}}}
    monkeypatch.setattr(tts_tool, '_load_tts_config', lambda: config)
    monkeypatch.setattr(tts_tool, '_session_platform', lambda: ('qqbot', False))
    return config


def wav_file(path):
    with wave.open(str(path), 'wb') as audio:
        audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        audio.writeframes(b'\x01\x00' * 2400)


def test_registry_voice_and_language_reach_command_without_mutation(configured, monkeypatch, tmp_path):
    before = copy.deepcopy(configured)
    seen = []

    def generate(text, output_path, provider, command, config):
        seen.append((text, command['voice'], command['language']))
        wav_file(output_path)
        return output_path

    monkeypatch.setattr(tts_tool, '_generate_command_tts', generate)
    result = json.loads(registry.dispatch('text_to_speech', {
        'text': 'こんにちは。', 'voice_id': '普拉娜', 'language': 'all_ja',
        'output_path': str(tmp_path / 'voice.wav')}))
    assert result['success'] and result['voice_compatible']
    assert seen == [('こんにちは。', 'plana-jp', 'all_ja')]
    assert configured == before
    assert result['media_tag'].startswith('[[audio_as_voice]]')
    with wave.open(result['file_path']) as audio:
        assert audio.getnframes() == 2400


@pytest.mark.parametrize('option', [{'voice_id': '../secret'}, {'voice_id': 'x & calc'}, {'language': 'invalid'}])
def test_unconfigured_selection_never_runs_command(configured, monkeypatch, option):
    generate = Mock()
    monkeypatch.setattr(tts_tool, '_generate_command_tts', generate)
    result = json.loads(registry.dispatch('text_to_speech', {'text': 'hello', **option}))
    assert not result['success']
    generate.assert_not_called()


def test_schema_advertises_only_configured_choices(configured):
    props = tts_tool._tts_schema_overrides()['parameters']['properties']
    assert '普拉娜' in props['voice_id']['enum']
    assert 'all_ja' in props['language']['enum']
    assert 'find_voices' not in props


def test_guest_can_select_voice_but_not_output_path():
    policy = RequestSecurityPolicy(subject_id='visitor', platform='qqbot', chat_id='room', scope_id='test')
    assert policy.allows_tool('text_to_speech', {'text': 'hello', 'voice_id': 'arona-cn', 'language': 'en'})
    assert not policy.allows_tool('text_to_speech', {'text': 'hello', 'output_path': 'C:/outside.wav'})
    assert not policy.allows_tool('terminal', {'command': 'anything'})


def test_default_voice_comes_from_config(configured):
    command = configured['providers']['local-speech']
    result = apply_options(configured, 'local-speech', command)
    assert result['providers']['local-speech']['voice'] == 'arona-cn'


def test_voice_compatible_wav_concat_stays_pcm(tmp_path):
    from tools.tts_tool_delivery import _concat_audio_files
    paths = [tmp_path / 'one.wav', tmp_path / 'two.wav']
    for path in paths:
        wav_file(path)
    destination = tmp_path / 'combined.wav'
    result = _concat_audio_files([str(path) for path in paths], str(destination), voice_compatible=True)
    assert result == str(destination)
    with wave.open(str(destination)) as audio:
        assert audio.getnframes() == 4800
        assert audio.getcomptype() == 'NONE'
