"""Fail-closed duration checks for complete local automatic QQ voice replies."""
from __future__ import annotations

import math
from pathlib import Path
import wave


def audio_duration(path: Path) -> float:
    """Measure authorized local audio, rejecting unknown/empty/truncated input."""
    from .voice_codec import silk_duration, TENCENT_MAGIC

    with path.open('rb') as stream:
        header = stream.read(12)
    if header.startswith((TENCENT_MAGIC, b'#!SILK_V3')):
        duration = silk_duration(path)
    elif path.suffix.lower() == '.wav' or header[:4] == b'RIFF':
        with wave.open(str(path), 'rb') as audio:
            frames = audio.getnframes()
            frame_size = audio.getnchannels() * audio.getsampwidth()
            # Header duration alone can accept a truncated synthesis. Read the
            # complete PCM in bounded chunks before accepting its frame count.
            remaining = frames
            while remaining:
                count = min(remaining, 65536)
                if len(audio.readframes(count)) != count * frame_size:
                    raise ValueError('Truncated WAV audio')
                remaining -= count
            duration = frames / audio.getframerate()
    else:
        from tools.transcription_audio import _probe_audio_duration
        duration = _probe_audio_duration(str(path.resolve()))
    if duration is None or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Audio duration is unknown or empty')
    return duration
