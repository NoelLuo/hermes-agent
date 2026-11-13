"""Speech cleanup must preserve content and avoid creating extra CJK clauses."""
import json
import wave

import pytest

from tools.tts_text_normalize import prepare_spoken_text
from tools import tts_tool
from tools.registry import registry


@pytest.mark.parametrize(('raw', 'expected'), [
    ('第一段。\n\n第二段！\n第三段？', '第一段。 第二段！ 第三段？'),
    ('First!\n\nNext?', 'First! Next?'),
    ('* one\n* two\n* three', 'one. two. three.'),
    ('***\n## 标题\n\n- [x] **完成**\n- [ ] *待办*\n---', '标题, 完成. 待办.'),
    ('变量 foo_bar_baz 和 x_y，2 * 3 * 4，a | b。', '变量 foo_bar_baz 和 x_y，2 * 3 * 4，a | b。'),
    ('```python\nx_y = 2 * 3\nprint(x_y)\n```', 'x_y = 2 * 3. print(x_y).'),
    ('~~旧~~，***新***，__加粗__，_斜体_。', '旧，新，加粗，斜体。'),
    ('你好。 [[qq_meme_optional:sample_id]] [[qq_meme_none]] [[audio_as_voice]]\nMEDIA:/tmp/demo.wav', '你好。'),
    ('| 项目 | 数值 |\n| :--- | ---: |\n| 苹果 | 三个 |', '项目; 数值. 苹果; 三个.'),
])
def test_spoken_markdown_boundaries(raw, expected):
    assert prepare_spoken_text(raw, max_chars=None) == expected


def test_registry_and_qq_auto_use_same_complete_script(monkeypatch, tmp_path):
    from agent.request_security import RequestSecurityPolicy, policy_scope
    from plugins.platforms.qqbot.voice_reply import _synthesize

    raw = '## 提醒\n\n* **喝水**。\n* 休息！\n\n晚安。'
    seen = []
    config = {'provider': 'local-speech', 'providers': {'local-speech': {
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
    result = json.loads(registry.dispatch('text_to_speech', {
        'text': raw, 'output_path': str(tmp_path / 'explicit.wav')}))
    assert result['success'] and result['chunk_count'] == 1
    policy = RequestSecurityPolicy(subject_id='guest', platform='qqbot', chat_id='room', scope_id='test')
    with policy_scope(policy):
        paths = _synthesize(raw, policy)
        assert paths and all(policy.allows_media(p) for p in paths)
    assert seen == ['提醒, 喝水。 休息！ 晚安。'] * 2
    assert _synthesize('长' * 181, policy) == []
    assert len(seen) == 2

def test_cleanup_failure_never_falls_back_to_raw_markup(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('broken cleaner')
    monkeypatch.setattr('tools.tts_text_normalize.prepare_spoken_text', fail)
    result = json.loads(registry.dispatch('text_to_speech', {'text': '**not spoken**'}))
    assert not result['success']
    assert 'cleanup failed' in result['error']
