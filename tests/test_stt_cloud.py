"""Cloud speech-to-text (OpenAI-compatible /audio/transcriptions) and the backend switch; no network, no models."""

import httpx
import numpy as np
import pytest

from wvt.cli import make_stt
from wvt.config import Config
from wvt.stt import SpeechToText
from wvt.stt_cloud import CloudSTT

CFG = {
    "base_url": "https://api.groq.com/openai/v1",
    "api_key_env": "GROQ_API_KEY",
    "model": "whisper-large-v3-turbo",
    "language": "pl",
    "timeout_s": 20,
    "send_prompt": True,
}


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _audio():
    return (np.sin(np.linspace(0, 2 * np.pi * 440, 16000)) * 0.5).astype(np.float32)


def _loaded(handler, cfg=None):
    """A CloudSTT with the key present and the model marked as loaded."""
    stt = CloudSTT(cfg or CFG, client=_client(handler))
    stt.load()
    return stt


def test_request_shape_and_text(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.read()
        return httpx.Response(200, json={"text": "dzień dobry"})

    stt = _loaded(handler)
    text = stt.transcribe(_audio(), prompt="Xardas, Lester")
    assert text == "dzień dobry"
    assert seen["url"].endswith("/audio/transcriptions")
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert b'name="model"' in body and b"whisper-large-v3-turbo" in body
    assert b'name="language"' in body and b"pl" in body
    assert b'name="response_format"' in body and b"json" in body
    assert b'name="prompt"' in body and b"Xardas, Lester" in body
    assert b'name="file"' in body and b'filename="audio.wav"' in body and b"audio/wav" in body
    assert b"RIFF" in body and b"WAVE" in body


def test_no_prompt_when_disabled(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    seen = {}

    def handler(request):
        seen["body"] = request.read()
        return httpx.Response(200, json={"text": "ok"})

    stt = _loaded(handler, {**CFG, "send_prompt": False})
    assert stt.prompt_ok is False
    stt.transcribe(_audio(), prompt="Xardas")
    assert b'name="prompt"' not in seen["body"]


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    stt = CloudSTT(CFG, client=_client(lambda r: httpx.Response(200, json={"text": ""})))
    with pytest.raises(RuntimeError) as exc:
        stt.load()
    assert "GROQ_API_KEY" in str(exc.value)


def test_non_200_raises_and_has_no_text(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def handler(request):
        return httpx.Response(401, text="invalid api key")

    stt = _loaded(handler)
    with pytest.raises(RuntimeError) as exc:
        stt.transcribe(_audio())
    assert "401" in str(exc.value)


def test_empty_audio_is_empty(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def handler(request):  # must not be reached
        raise AssertionError("no request for empty audio")

    stt = _loaded(handler)
    assert stt.transcribe(np.zeros(0, dtype=np.float32)) == ""


def test_endpoint_normalization():
    from wvt.stt_cloud import endpoint

    assert endpoint("https://api.groq.com/openai/v1/") == "https://api.groq.com/openai/v1/audio/transcriptions"
    full = "https://x/audio/transcriptions"
    assert endpoint(full) == full


def test_make_stt_cloud(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    cfg = Config({"stt": {"backend": "cloud", "cloud": dict(CFG)}})
    stt = make_stt(cfg)
    assert isinstance(stt, CloudSTT)
    assert stt.device == "cloud"
    assert stt.model_label == "whisper-large-v3-turbo"


def test_make_stt_defaults_to_local():
    cfg = Config({"stt": {"model_dir": "models/whisper-large-v3-turbo-onnx", "model_name": "whisper"}})
    stt = make_stt(cfg)
    assert isinstance(stt, SpeechToText)
    assert stt.model is None  # not loaded by the factory
