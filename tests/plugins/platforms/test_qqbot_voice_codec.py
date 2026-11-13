"""QQ voice conversion and real request-policy/upload boundaries (no QQ traffic)."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.request_security import RequestSecurityPolicy, current_policy, policy_scope
from gateway.platforms.base import SendResult
from plugins.platforms.qqbot.adapter import QQAdapter, MEDIA_TYPE_VOICE
from plugins.platforms.qqbot import voice_codec as codec


def packet_stream(count=50):
    return codec.TENCENT_MAGIC + b"\x03\x00abc" * count


def adapter():
    obj = QQAdapter.__new__(QQAdapter)
    obj.config = SimpleNamespace(extra={"voice_codec_command": [str(Path.cwd() / "codec.exe")]})
    obj._chat_type_map = {"room": "group"}
    obj._ensure_connected = AsyncMock(return_value=True)
    obj._upload_local_file = AsyncMock(return_value={"file_info": "silk-upload"})
    obj._post_message = AsyncMock(return_value=SendResult(success=True, message_id="voice-id"))
    obj._next_msg_seq = lambda _: 1
    return obj


@pytest.fixture
def conversion(tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    source.write_bytes(b"RIFF audio input")
    calls = []
    async def run(argv):
        calls.append(argv)
        if argv[-1].endswith(".pcm"):
            Path(argv[-1]).write_bytes(b"\0" * 48000)
        else:
            Path(argv[-1]).write_bytes(packet_stream())
    monkeypatch.setattr(codec, "_run", run)
    monkeypatch.setattr(codec, "locate_command", lambda _: SimpleNamespace(command=("ffmpeg",)))
    monkeypatch.setattr(codec, "get_hermes_home", lambda: tmp_path / "profile")
    return source, calls


@pytest.mark.parametrize("data", [b"", codec.TENCENT_MAGIC, b"#!SILK_V3" + b"\x03\x00abc",
    codec.TENCENT_MAGIC + b"\x03\x00ab", codec.TENCENT_MAGIC + b"\0\0abc",
    codec.TENCENT_MAGIC + b"\x01", codec.TENCENT_MAGIC + b"\xff\xffabc"])
def test_rejects_non_tencent_empty_and_truncated_silk(tmp_path, data):
    path = tmp_path / "bad.silk"
    path.write_bytes(data)
    with pytest.raises(codec.VoiceCodecError):
        codec.silk_duration(path)


def test_valid_packets_duration(tmp_path):
    path = tmp_path / "voice.silk"
    path.write_bytes(packet_stream())
    assert codec.silk_duration(path) == 1.0


def test_native_upload_uses_type_three_and_scoped_generated_artifact(conversion):
    source, calls = conversion
    obj = adapter()
    policy = RequestSecurityPolicy("visitor", "qqbot", "room")
    other = RequestSecurityPolicy("other", "qqbot", "room")
    policy.grant_artifact(str(source))
    outputs = []
    async def upload(chat_type, chat_id, path, file_type, file_name):
        outputs.append(Path(path))
        assert file_type == MEDIA_TYPE_VOICE == 3
        assert current_policy() is policy
        assert policy.allows_media(path) and not other.allows_media(path)
        assert codec.silk_duration(Path(path)) == 1.0
        return {"file_info": "silk-upload"}
    obj._upload_local_file.side_effect = upload
    with policy_scope(policy):
        result = asyncio.run(obj.send_voice("room", str(source), caption="voice", reply_to="inbound"))
    assert result.success
    assert len(calls) == 2
    assert calls[0][calls[0].index("-ac") + 1] == "1"
    assert calls[0][calls[0].index("-ar") + 1] == "24000"
    assert all(not output.exists() and not output.parent.exists() for output in outputs)
    body = obj._post_message.call_args.args[1]
    assert body["media"] == {"file_info": "silk-upload"}
    assert body["msg_id"] == "inbound"
    assert source.exists()


@pytest.mark.parametrize("destination,grant", [("room", False), ("other-room", True)])
def test_authorizes_before_conversion(conversion, destination, grant):
    source, calls = conversion
    obj = adapter()
    policy = RequestSecurityPolicy("visitor", "qqbot", "room")
    if grant:
        policy.grant_artifact(str(source))
    with policy_scope(policy):
        result = asyncio.run(obj.send_voice(destination, str(source)))
    assert not result.success and "denied" in result.error
    assert not calls
    obj._upload_local_file.assert_not_awaited()


def test_remote_urls_never_upload_as_native_voice(conversion):
    _, calls = conversion
    obj = adapter()
    with policy_scope(RequestSecurityPolicy("visitor", "qqbot", "room")):
        result = asyncio.run(obj.send_voice("room", "https://example.org/voice.silk"))
    assert not result.success and "Remote" in result.error
    assert not calls
    obj._upload_local_file.assert_not_awaited()


def test_existing_tencent_silk_passes_without_codec(conversion):
    source, calls = conversion
    source.write_bytes(packet_stream())
    obj = adapter()
    obj.config.extra = {}
    policy = RequestSecurityPolicy("visitor", "qqbot", "room")
    policy.grant_artifact(str(source))
    with policy_scope(policy):
        result = asyncio.run(obj.send_voice("room", str(source)))
    assert result.success and not calls and source.exists()


@pytest.mark.parametrize("command", [None, "shell command", ["relative.exe"], []])
def test_missing_or_invalid_codec_configuration_is_clear(conversion, command):
    source, _ = conversion
    obj = adapter()
    obj.config.extra["voice_codec_command"] = command
    with policy_scope(RequestSecurityPolicy("owner", "qqbot", "room", is_admin=True)):
        result = asyncio.run(obj.send_voice("room", str(source)))
    assert not result.success and "voice_codec_command" in result.error
    obj._upload_local_file.assert_not_awaited()


@pytest.mark.parametrize("failure", ["encode", "upload", "cancel"])
def test_cleanup_on_failure_and_cancellation(conversion, monkeypatch, failure):
    source, _ = conversion
    obj = adapter()
    if failure == "encode":
        original = codec._run
        async def fail(argv):
            if argv[-1].endswith(".silk"):
                raise codec.VoiceCodecError("encoder unavailable")
            await original(argv)
        monkeypatch.setattr(codec, "_run", fail)
    else:
        obj._upload_local_file.side_effect = (asyncio.CancelledError() if failure == "cancel"
                                              else OSError("upload unavailable"))

    with policy_scope(RequestSecurityPolicy("owner", "qqbot", "room", is_admin=True)):
        if failure == "cancel":
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(obj.send_voice("room", str(source)))
        else:
            assert not asyncio.run(obj.send_voice("room", str(source))).success
    assert not list((source.parent / "profile" / "cache" / "qq-voice").glob("voice-*"))
    assert source.exists()


def test_wrong_encoded_duration_fails_closed(conversion, monkeypatch):
    source, _ = conversion
    original = codec._run
    async def wrong_duration(argv):
        await original(argv)
        if argv[-1].endswith(".silk"):
            Path(argv[-1]).write_bytes(packet_stream(5))
    monkeypatch.setattr(codec, "_run", wrong_duration)
    obj = adapter()
    with policy_scope(RequestSecurityPolicy("owner", "qqbot", "room", is_admin=True)):
        result = asyncio.run(obj.send_voice("room", str(source)))
    assert not result.success and "duration" in result.error
    obj._upload_local_file.assert_not_awaited()


def test_adapter_derived_policy_is_bound_through_upload(conversion, monkeypatch):
    source, _ = conversion
    obj = adapter()
    obj.build_source = lambda **kw: SimpleNamespace(**kw)
    policy = RequestSecurityPolicy("visitor", "qqbot", "room")
    policy.grant_artifact(str(source))
    monkeypatch.setattr("gateway.request_security.policy_for_adapter", lambda *_: policy)
    async def upload(*args):
        assert current_policy() is policy
        assert policy.allows_media(args[2])
        return {"file_info": "silk-upload"}
    obj._upload_local_file.side_effect = upload
    with policy_scope(None):
        assert asyncio.run(obj.send_voice("room", str(source))).success
        assert current_policy() is None


@pytest.mark.parametrize("pcm_size", [0, 959, 48000 * 301])
def test_empty_short_or_oversized_pcm_is_not_encoded(conversion, monkeypatch, pcm_size):
    source, _ = conversion
    calls = []
    async def invalid_pcm(argv):
        calls.append(argv)
        Path(argv[-1]).write_bytes(b"\0" * pcm_size)
    monkeypatch.setattr(codec, "_run", invalid_pcm)
    obj = adapter()
    with policy_scope(RequestSecurityPolicy("owner", "qqbot", "room", is_admin=True)):
        result = asyncio.run(obj.send_voice("room", str(source)))
    assert not result.success and "PCM" in result.error
    assert len(calls) == 1
    obj._upload_local_file.assert_not_awaited()
