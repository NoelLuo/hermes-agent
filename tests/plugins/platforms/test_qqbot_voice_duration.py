"""Duration probing uses real WAV/SILK parsing and the existing ffprobe seam."""
import pytest

from plugins.platforms.qqbot.voice_duration import audio_duration


@pytest.mark.parametrize('duration', [None, float('nan'), float('inf'), 0, -1])
def test_unknown_or_invalid_probe_duration_rejected(tmp_path, monkeypatch, duration):
    path = tmp_path / 'audio.mp3'
    path.write_bytes(b'probe fixture')
    monkeypatch.setattr('tools.transcription_audio._probe_audio_duration', lambda _: duration)
    with pytest.raises(ValueError, match='unknown or empty'):
        audio_duration(path)


def test_other_formats_use_existing_probe(tmp_path, monkeypatch):
    path = tmp_path / 'audio.ogg'
    path.write_bytes(b'probe fixture')
    seen = []
    def probe(name):
        seen.append(name)
        return 60.001
    monkeypatch.setattr('tools.transcription_audio._probe_audio_duration', probe)
    assert audio_duration(path) == 60.001
    assert seen == [str(path.resolve())]


@pytest.mark.parametrize('packets, expected', [(3000, 60), (3001, 60.02)])
def test_silk_packet_duration_without_rounding(tmp_path, packets, expected):
    path = tmp_path / 'audio.silk'
    path.write_bytes(b'\x02#!SILK_V3' + b'\x02\x00xx' * packets)
    assert audio_duration(path) == pytest.approx(expected)


def test_invalid_silk_never_falls_back_to_probe(tmp_path, monkeypatch):
    path = tmp_path / 'audio.silk'
    path.write_bytes(b'\x02#!SILK_V3\x03\x00xx')
    def unexpected(_):
        pytest.fail('Invalid SILK must not use permissive probe fallback')
    monkeypatch.setattr('tools.transcription_audio._probe_audio_duration', unexpected)
    with pytest.raises(ValueError):
        audio_duration(path)
