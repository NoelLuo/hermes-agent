"""Local audio -> Tencent SILK, with request-bound native QQ delivery.

Operator-only PlatformConfig.extra keys:
  voice_codec_command: [absolute executable, fixed args...] (PCM, SILK appended)
  voice_ffmpeg: optional ffmpeg executable path
The codec must encode mono 24000Hz s16le PCM into Tencent SILK, 20ms packets.
No shell, network fetch, caller-supplied command, or document fallback is used.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import struct
import tempfile

from hermes_constants import get_hermes_home
from hermes_platform.resolver.core import locate_command

TENCENT_MAGIC = b"\x02#!SILK_V3"
MAX_SECONDS = 300  # Local resource cap, not a claim about QQ's duration limit.
MAX_SILK_BYTES = 20 * 1024 * 1024  # Stay below QQ's voice -> file downgrade threshold.


class VoiceCodecError(ValueError):
    """A voice cannot be safely delivered as native SILK."""


def silk_duration(path: Path) -> float:
    """Validate Tencent prefix and complete nonempty 20ms packet framing."""
    if not 12 < path.stat().st_size < MAX_SILK_BYTES:
        raise VoiceCodecError("SILK is empty or exceeds the native voice size limit")
    data = path.read_bytes()
    if not data.startswith(TENCENT_MAGIC):
        raise VoiceCodecError("QQ voice requires Tencent SILK magic (02 23 21 53 49 4c 4b 5f 56 33)")
    offset, packets = len(TENCENT_MAGIC), 0
    while offset < len(data):
        if offset + 2 > len(data):
            raise VoiceCodecError("Truncated SILK packet length")
        size = struct.unpack_from("<h", data, offset)[0]
        offset += 2
        if size <= 0 or offset + size > len(data):
            raise VoiceCodecError("Invalid or truncated SILK packet")
        offset += size
        packets += 1
    duration = packets * 0.02
    if not 0 < duration <= MAX_SECONDS:
        raise VoiceCodecError("SILK duration is empty or exceeds 300 seconds")
    return duration


async def _run(argv: list[str]) -> None:
    """Reap children before temporary files are removed, including cancellation."""
    process = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    try:
        code = await asyncio.wait_for(process.wait(), timeout=120)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    if code:
        raise VoiceCodecError(f"Voice codec process failed (exit {code})")


@asynccontextmanager
async def native_voice(source: str, extra: dict):
    """Caller MUST authorize the original source before entering this context."""
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise VoiceCodecError("Voice source is not a local file")
    with path.open("rb") as stream:
        header = stream.read(10)
    if header.startswith((TENCENT_MAGIC, b"#!SILK_V3")):
        silk_duration(path)
        yield path
        return
    command = extra.get("voice_codec_command")
    if (not isinstance(command, list) or not command
            or any(not isinstance(part, str) or not part or "\x00" in part for part in command)
            or not Path(command[0]).is_absolute()):
        raise VoiceCodecError("Configure QQ voice_codec_command as an absolute executable argv list")
    ffmpeg = locate_command(str(extra.get("voice_ffmpeg") or "ffmpeg")).command
    if not ffmpeg:
        raise VoiceCodecError("ffmpeg is required for QQ native voice conversion")
    cache = get_hermes_home() / "cache" / "qq-voice"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="voice-", dir=cache) as temp:
        pcm, silk = Path(temp) / "audio.pcm", Path(temp) / "audio.silk"
        await _run([*ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-protocol_whitelist", "file", "-format_whitelist",
                    "wav,mp3,flac,ogg,aac,mov,matroska,webm", "-i", str(path),
                    "-map", "0:a:0", "-vn", "-t", str(MAX_SECONDS + 1),
                    "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", "-f", "s16le", str(pcm)])
        size = pcm.stat().st_size
        duration = size / 48000
        if size % 2 or not 0.02 <= duration <= MAX_SECONDS:
            raise VoiceCodecError("Audio must contain 20ms to 300s of nonempty PCM")
        await _run([*command, str(pcm), str(silk)])
        encoded_duration = silk_duration(silk)
        if abs(encoded_duration - duration) > 0.1:
            raise VoiceCodecError("SILK duration does not match source PCM")
        yield silk


async def send_native_voice(adapter, chat_id, source, caption=None, reply_to=None):
    from agent.request_security import current_policy, policy_scope
    from gateway.platforms.base import SendResult
    from .adapter import MEDIA_TYPE_VOICE

    policy = current_policy()
    if policy is None:
        from gateway.request_security import policy_for_adapter
        policy = policy_for_adapter(adapter, adapter.build_source(
            chat_id=chat_id, chat_type=adapter._guess_chat_type(chat_id)))
    if policy is not None:
        if not policy.is_admin and (policy.platform != "qqbot" or policy.chat_id != chat_id):
            return SendResult(success=False, error="Request policy denied attachment destination")
        if not policy.allows_media(source):
            return SendResult(success=False, error="Request policy denied attachment")
    if adapter._is_url(source):
        return SendResult(success=False, error="Remote QQ voice URLs are unsupported; provide authorized local audio")
    try:
        with policy_scope(policy):
            async with native_voice(source, adapter.config.extra or {}) as output:
                generated = output != Path(source).expanduser().resolve()
                if policy is not None and generated and not policy.grant_artifact(str(output)):
                    raise VoiceCodecError("Request policy could not authorize converted voice")
                return await adapter._send_media(
                    chat_id, str(output), MEDIA_TYPE_VOICE, "voice", caption, reply_to)
    except (OSError, ValueError) as exc:
        return SendResult(success=False, retryable=False,
                          error=f"QQ native voice conversion failed: {exc or type(exc).__name__}")
