"""Optional cloud speech-to-text: any OpenAI-compatible ``/audio/transcriptions`` API (Groq, OpenAI, OpenRouter...).

Local Whisper stays the default (``stt.backend: "local"``); this is chosen on purpose, e.g. on a machine without a
GPU or while the local model is busy.  The recording (WAV, PCM 16-bit mono 16 kHz) is sent to the provider named in
``stt.cloud.base_url`` - that is the trade-off the setting makes.

The class has the same surface as :class:`wvt.stt.SpeechToText` (``load``, ``warmup``, ``transcribe``, ``describe``,
``model``, ``model_label``, ``device``, ``load_seconds``, ``prompt_ok``, ``dropped``, ``prompt_dropped``), so the app,
the pipeline and :mod:`wvt.stt_server` do not need to know which one they hold.  Nothing about the audio or the text
is ever logged; only the provider status and the model name.
"""

from __future__ import annotations

import io
import logging
import time
import wave
from typing import Optional

import httpx
import numpy as np

from .envfile import secret

log = logging.getLogger(__name__)


def endpoint(base_url: str) -> str:
    """``https://api.groq.com/openai/v1`` -> ``.../v1/audio/transcriptions`` (already-complete URLs kept)."""
    url = (base_url or "").strip().rstrip("/")
    if url.endswith("/audio/transcriptions"):
        return url
    return url + "/audio/transcriptions"


def encode_wav(audio: np.ndarray, sample_rate: int = 16000) -> io.BytesIO:
    """float32 samples in [-1, 1] -> a WAV file (PCM 16-bit, mono) ready to upload."""
    pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    data = (pcm * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(data)
    buf.seek(0)
    return buf


class CloudSTT:
    """Speech-to-text answered by a cloud provider instead of the local Whisper."""

    def __init__(self, cfg: dict, client: Optional[httpx.Client] = None):
        cfg = cfg or {}
        self.url = endpoint(str(cfg.get("base_url") or "https://api.groq.com/openai/v1"))
        self.api_key_env = str(cfg.get("api_key_env") or "GROQ_API_KEY")
        self._model = str(cfg.get("model") or "whisper-large-v3-turbo")
        self.language = str(cfg.get("language") or "")
        self.timeout_s = float(cfg.get("timeout_s") or 20)
        self.send_prompt = bool(cfg.get("send_prompt", True))
        self.model = None  # set by load(), like SpeechToText
        self.device = "cloud"
        self.load_seconds = 0.0
        self.prompt_ok = self.send_prompt
        self.dropped = 0
        self.prompt_dropped = 0
        self._key = ""
        self._client = client
        self._owns_client = client is None

    # ------------------------------------------------------------------ what is loaded
    @property
    def model_label(self) -> str:
        return self._model

    def describe(self) -> str:
        return f"{self.model_label} (chmura)"

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        key = secret(self.api_key_env)
        if not key:
            raise RuntimeError(f"brak klucza {self.api_key_env} w .env")
        self._key = key
        self.model = self._model
        log.info("STT ready on cloud: %s", self._model)

    def warmup(self) -> None:
        return None

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
        self._client = None

    # ------------------------------------------------------------------ recognition
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000, prompt: str = "") -> str:
        """Speech -> text through the provider's ``/audio/transcriptions``.  ``prompt`` is sent when enabled."""
        if audio is None or np.size(audio) == 0:
            return ""
        if self.model is None:
            raise RuntimeError("STT not loaded")
        form: dict[str, str] = {"model": self._model, "response_format": "json"}
        if self.language:
            form["language"] = self.language
        if self.send_prompt and prompt:
            form["prompt"] = prompt
        wav = encode_wav(audio, sample_rate)
        t0 = time.perf_counter()
        response = self._http().post(
            self.url,
            headers={"Authorization": f"Bearer {self._key}"},
            data=form,
            files={"file": ("audio.wav", wav, "audio/wav")},
        )
        if response.status_code != 200:  # status and a short body only: never the audio or the transcript
            snippet = " ".join((response.text or "").split())[:200]
            raise RuntimeError(f"STT cloud {response.status_code}: {snippet}")
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError("STT cloud: the provider did not return JSON") from None
        text = str(data.get("text") or "").strip() if isinstance(data, dict) else ""
        log.debug("STT cloud answered in %.0f ms", (time.perf_counter() - t0) * 1000)
        return text
