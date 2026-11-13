"""Command TTS buffers complete replies; cancelled POST workers own cleanup."""
import asyncio
import base64
import threading

import pytest

from hermes_cli.web_routers import audio
from hermes_cli.web_models import TTSSpeakRequest
from hermes_cli import web_server
from tests.hermes_cli.test_web_server_speak_stream import stream_client, _url, _FakeStreamer


@pytest.fixture
def command_config(monkeypatch):
    cfg = {"provider": "local-command", "providers": {
        "local-command": {"type": "command", "command": "tts {input_path} {output_path}"}}}
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: cfg)
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: None)


def test_command_requests_complete_reply_post_without_sentence_synthesis(
    stream_client, command_config, monkeypatch, tmp_path
):
    calls = []
    path = tmp_path / "response.wav"
    def synth(text, **kwargs):
        calls.append(text)
        path.write_bytes(b"complete-audio")
        return {"success": True, "file_path": str(path), "provider": "local-command"}
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", synth)
    with stream_client.websocket_connect(_url()) as conn:
        monkeypatch.setattr(audio, "_audio_file_to_pcm", lambda path: (b"\x01\x00", 24000))
        conn.send_json({"text": "First entire sentence. Second entire paragraph.", "done": True})
        assert conn.receive_json() == {"type": "fallback"}
    assert calls == []
    text = "First entire sentence.\n\nSecond entire paragraph. End."
    result = stream_client.post("/api/audio/speak", json={"text": text}, headers={"Authorization": f"Bearer {web_server._SESSION_TOKEN}"})
    assert result.status_code == 200
    assert calls == [text]
    assert base64.b64decode(result.json()["data_url"].split(",", 1)[1]) == b"complete-audio"
    assert not path.exists()


def test_native_streamer_takes_priority_even_for_command_config(stream_client, command_config, monkeypatch):
    streamer = _FakeStreamer([b"\x01\x00"])
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: streamer)
    with stream_client.websocket_connect(_url()) as conn:
        conn.send_json({"text": "A complete spoken sentence.", "done": True})
        assert conn.receive_json()["type"] == "start"
        assert conn.receive_bytes() == b"\x01\x00"
        assert conn.receive_json() == {"type": "end"}
    assert streamer.requests == ["A complete spoken sentence."]


def test_multiple_delivery_files_never_silently_play_only_first(stream_client, monkeypatch, tmp_path):
    paths = [tmp_path / "first.wav", tmp_path / "last.wav"]
    def synth(text):
        for path in paths:
            path.write_bytes(b"audio")
        return {"success": True, "file_path": str(paths[0]), "file_paths": list(map(str, paths))}
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", synth)
    result = stream_client.post("/api/audio/speak", json={"text": "full reply"}, headers={"Authorization": f"Bearer {web_server._SESSION_TOKEN}"})
    assert result.status_code == 413
    assert all(not path.exists() for path in paths)


@pytest.mark.asyncio
async def test_cancel_during_synthesis_still_cleans_late_file(monkeypatch, tmp_path):
    entered, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    path = tmp_path / "late.wav"
    def synth(text, **kwargs):
        entered.set()
        assert release.wait(10)
        path.write_bytes(b"late-result")
        return {"success": True, "file_path": str(path)}
    unlink = audio._unlink_quietly
    def observe_unlink(value):
        unlink(value)
        cleaned.set()
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", synth)
    monkeypatch.setattr(audio, "_unlink_quietly", observe_unlink)
    task = asyncio.create_task(audio.speak_text(TTSSpeakRequest(text="Complete reply.")))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
    assert await asyncio.to_thread(cleaned.wait, 10)
    assert not path.exists()


def test_oversized_audio_rejected_not_truncated_and_cleaned(stream_client, monkeypatch, tmp_path):
    path = tmp_path / "large.wav"
    def synth(text, **kwargs):
        path.write_bytes(b"123456789")
        return {"success": True, "file_path": str(path)}
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", synth)
    monkeypatch.setattr(audio, "_MAX_SPEAK_AUDIO_BYTES", 8)
    result = stream_client.post("/api/audio/speak", json={"text": "reply"}, headers={"Authorization": f"Bearer {web_server._SESSION_TOKEN}"})
    assert result.status_code == 413
    assert not path.exists()
