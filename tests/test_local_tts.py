"""Local TTS (Piper/SAPI): resampling, 24 kHz s16le streaming, fallback, availability."""

import importlib.util
import threading

import numpy as np
import pytest

from wvt.voice import local


def test_resample_lengths():
    x = np.zeros(16000, dtype=np.float32)
    assert len(local.resample_to_24k(x, 16000)) == 24000
    y = np.zeros(22050, dtype=np.float32)
    assert len(local.resample_to_24k(y, 22050)) == 24000
    assert len(local.resample_to_24k(np.zeros(24000, dtype=np.float32), 24000)) == 24000


def test_resample_short_and_empty_are_safe():
    assert local.resample_to_24k(np.zeros(0, dtype=np.float32), 16000).size == 0
    one = local.resample_to_24k(np.array([0.5], dtype=np.float32), 16000)
    assert one.size > 0 and np.allclose(one, 0.5)
    assert local.resample_to_24k(np.zeros(2, dtype=np.float32), 0).size == 2  # bad rate: unchanged


def test_pcm16_chunks_are_16bit_and_chunked():
    chunks = list(local.pcm16_chunks(np.zeros(5000, dtype=np.float32), chunk=2400))
    assert [len(c) for c in chunks] == [4800, 4800, 400]
    pcm = np.frombuffer(b"".join(chunks), dtype="<i2")
    assert pcm.size == 5000
    assert list(local.pcm16_chunks(np.zeros(0, dtype=np.float32))) == []


def _fake_piper(self, text):
    return np.full(24000, 0.25, dtype=np.float32)  # 1 s at 24 kHz


def _fake_sapi(self, text):
    return np.full(12000, -0.5, dtype=np.float32)  # 0.5 s at 24 kHz


def test_stream_tts_piper_yields_24k_pcm_and_stats(monkeypatch):
    monkeypatch.setattr(local.LocalTTS, "_piper_pcm", _fake_piper)
    tts = local.LocalTTS({"engine": "piper", "piper_voice": "pl_PL-gosia-medium"})
    assert tts.model_id == "piper-pl_PL-gosia-medium"
    assert tts.output_format == "pcm_24000"
    stats = []
    chunks = list(tts.stream_tts("local:pl", "Cześć, tu Iskra.", cancel=threading.Event(), on_stats=stats.append))
    pcm = np.frombuffer(b"".join(chunks), dtype="<i2")
    assert pcm.size == 24000  # 1 s at 24 kHz, no WAV header
    assert np.allclose(pcm[0], int(0.25 * 32767))
    assert len(stats) == 1
    assert stats[0].chars == len("Cześć, tu Iskra.")
    assert stats[0].cost == 0.0


def test_stream_tts_falls_back_to_sapi_when_piper_raises(monkeypatch):
    def boom(self, text):
        raise RuntimeError("piper broke")

    monkeypatch.setattr(local.LocalTTS, "_piper_pcm", boom)
    monkeypatch.setattr(local.LocalTTS, "_sapi_pcm", _fake_sapi)
    tts = local.LocalTTS({"engine": "auto"})
    tts._piper_ready = True  # pretend Piper is installed so "auto" selects it first
    assert tts._resolved_engine() == "piper"
    stats = []
    pcm = np.frombuffer(b"".join(tts.stream_tts("local:pl", "test", on_stats=stats.append)), dtype="<i2")
    assert pcm.size == 12000  # SAPI audio, resampled already by the fake
    assert len(stats) == 1


def test_stream_tts_piper_raises_when_engine_is_explicit(monkeypatch):
    def boom(self, text):
        raise RuntimeError("piper broke")

    monkeypatch.setattr(local.LocalTTS, "_piper_pcm", boom)
    tts = local.LocalTTS({"engine": "piper"})
    with pytest.raises(RuntimeError):
        list(tts.stream_tts("local:pl", "test"))


def test_stream_tts_cancel_stops_chunks_but_still_reports_stats(monkeypatch):
    monkeypatch.setattr(local.LocalTTS, "_piper_pcm", _fake_piper)
    tts = local.LocalTTS({"engine": "piper"})
    cancel = threading.Event()
    cancel.set()
    stats = []
    assert list(tts.stream_tts("local:pl", "test", cancel=cancel, on_stats=stats.append)) == []
    assert len(stats) == 1


def test_available_with_fake_module_and_model(monkeypatch, tmp_path):
    model = tmp_path / "pl_PL-gosia-medium.onnx"
    model.write_bytes(b"x")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "piper" else None)
    monkeypatch.setattr(local.shutil, "which", lambda name: None)
    tts = local.LocalTTS({"engine": "piper", "piper_model": str(model)})
    assert tts.available() is True
    assert tts.model_id == "piper-pl_PL-gosia-medium"


def test_missing_model_or_module_marks_piper_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(local.shutil, "which", lambda name: None)
    tts = local.LocalTTS({"engine": "piper", "piper_model": str(tmp_path / "missing.onnx")})
    assert tts.available() is False
    assert tts._piper_ready is False


def test_auto_engine_uses_sapi_when_piper_missing(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(local.shutil, "which", lambda name: None)
    tts = local.LocalTTS({"engine": "auto", "sapi_voice": "pl-PL"})
    assert tts._resolved_engine() == "sapi"
    assert tts.model_id == "sapi-pl-PL"
    assert tts.available() == (local.sys.platform == "win32")


def test_close_is_harmless():
    local.LocalTTS({"engine": "sapi"}).close()
