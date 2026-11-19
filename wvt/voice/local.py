"""Local text to speech for Iskra: Piper first, the Windows SAPI voice (Microsoft Paulina) as fallback.

The vocal output is 24 kHz mono s16le raw PCM, the same format the ElevenLabs client streams
and the mixer expects.  Only the assistant speaks through here; character voices stay text-only.

Piper (GPL) is optional: when the module or the model file is missing, SAPI is used instead.
Nothing here logs the spoken text - only lengths and engine names.
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from ..config import resolve_path
from .elevenlabs import TTSStats

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
CHUNK_SAMPLES = SAMPLE_RATE // 10  # 0.1 s per yielded chunk
PIPER_TIMEOUT_S = 30.0
SAPI_TIMEOUT_S = 30.0
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

_SAPI_SCRIPT = r"""
& {
Add-Type -AssemblyName System.Speech
$ErrorActionPreference = 'Stop'
$out = $args[0]
$src = $args[1]
$text = [System.IO.File]::ReadAllText($src, [System.Text.Encoding]::UTF8)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = @($synth.GetInstalledVoices() | Where-Object { $_.Enabled })
$voice = $voices | Where-Object { $_.VoiceInfo.Culture.Name -eq 'pl-PL' } | Select-Object -First 1
if (-not $voice) {
    $voice = $voices | Where-Object { $_.VoiceInfo.Culture.Name -like 'pl*' } | Select-Object -First 1
}
if (-not $voice) {
    $synth.Dispose()
    throw 'Brak polskiego glosu TTS (Microsoft Paulina) w systemie.'
}
$synth.SelectVoice($voice.VoiceInfo.Name)
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(24000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth.SetOutputToWaveFile($out, $fmt)
$synth.Speak($text)
$synth.SetOutputToNull()
$synth.Dispose()
}
""".strip()  # no trailing newline: powershell would treat the following arguments as a new command


def resample_to_24k(samples: np.ndarray, sample_rate: int, target: int = SAMPLE_RATE) -> np.ndarray:
    """Linear interpolation to ``target`` Hz (float32).  Empty or 1-sample input is safe."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size == 0 or sample_rate <= 0 or sample_rate == target:
        return x
    n = int(round(len(x) * target / sample_rate))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(x) == 1:
        return np.full(n, float(x[0]), dtype=np.float32)
    pos = np.linspace(0.0, len(x) - 1, n, dtype=np.float64)
    idx = np.clip(np.floor(pos).astype(np.int64), 0, len(x) - 2)
    frac = (pos - idx).astype(np.float32)
    return (x[idx] * (1.0 - frac) + x[idx + 1] * frac).astype(np.float32)


def pcm16_chunks(samples: np.ndarray, chunk: int = CHUNK_SAMPLES) -> Iterator[bytes]:
    """Float32 [-1, 1] -> s16le bytes, ``chunk`` samples each."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2").tobytes()
    step = max(1, int(chunk)) * 2  # bytes (16-bit mono)
    for i in range(0, len(pcm), step):
        yield pcm[i:i + step]


class LocalTTS:
    """Speech synthesis without ElevenLabs.  Piper (optional) with the SAPI voice as fallback."""

    output_format = "pcm_24000"

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})
        self.enabled = bool(self.cfg.get("enabled", True))
        self.engine = str(self.cfg.get("engine", "auto") or "auto").lower()
        self._volume = float(self.cfg.get("volume", 1.0) or 1.0)
        self._sapi_voice = str(self.cfg.get("sapi_voice", "pl-PL") or "pl-PL")
        self._piper_voice = str(self.cfg.get("piper_voice", "pl_PL-gosia-medium") or "pl_PL-gosia-medium")
        self.piper_model = resolve_path(self.cfg.get("piper_model", "models/piper/pl_PL-gosia-medium.onnx"))
        self._piper_module = importlib.util.find_spec("piper") is not None
        self._piper_exe = shutil.which("piper") if not self._piper_module else None
        self._piper_ready = (self._piper_module or bool(self._piper_exe)) and self.piper_model.exists()
        self._sapi_ready = sys.platform == "win32"
        self.model_id = self._engine_id()

    # ------------------------------------------------------------------ engine choice
    def _engine_id(self) -> str:
        engine = self._resolved_engine()
        return f"piper-{self._piper_voice}" if engine == "piper" else f"sapi-{self._sapi_voice}"

    def _resolved_engine(self) -> str:
        if self.engine in ("piper", "sapi"):
            return self.engine
        return "piper" if self._piper_ready else "sapi"

    def available(self) -> bool:
        """A synthesis attempt is possible (SAPI is only checked by running it, never at init)."""
        if self.engine == "piper":
            return self._piper_ready
        if self.engine == "sapi":
            return self._sapi_ready
        return self._piper_ready or self._sapi_ready

    def describe(self) -> str:
        engine = self._resolved_engine()
        if engine == "piper":
            return f"Piper ({self._piper_voice})"
        return f"SAPI ({self._sapi_voice})"

    # ------------------------------------------------------------------ streaming
    def stream_tts(self, voice_id: str, text: str, language_code: Optional[str] = None,
                   cancel=None, on_stats=None) -> Iterator[bytes]:
        """Yield 24 kHz mono s16le chunks; calls ``on_stats`` once (cost 0.0)."""
        text = " ".join((text or "").split())
        t0 = time.monotonic()
        samples = self._render(text, cancel)
        if self._volume != 1.0:
            samples = np.clip(samples * self._volume, -1.0, 1.0).astype(np.float32)
        ttfa: Optional[float] = None
        try:
            for chunk in pcm16_chunks(samples):
                if cancel is not None and cancel.is_set():
                    break
                if ttfa is None:
                    ttfa = (time.monotonic() - t0) * 1000.0
                yield chunk
        finally:
            if on_stats is not None:
                total_ms = (time.monotonic() - t0) * 1000.0
                on_stats(TTSStats(ttfa if ttfa is not None else total_ms, total_ms, len(text), 0.0))

    def _render(self, text: str, cancel=None) -> np.ndarray:
        """Float32 samples at 24 kHz from the chosen engine, with a Piper -> SAPI fallback in auto."""
        engine = self._resolved_engine()
        if engine == "piper":
            try:
                return self._piper_pcm(text)
            except Exception as exc:
                if self.engine != "auto" or not self._sapi_ready:
                    raise
                log.warning("piper synthesis failed (%s) - falling back to SAPI", type(exc).__name__)
                return self._sapi_pcm(text)
        return self._sapi_pcm(text)

    # ------------------------------------------------------------------ Piper
    def _piper_cmd(self) -> list[str]:
        if self._piper_module:
            return [sys.executable, "-m", "piper"]
        return [self._piper_exe or "piper"]

    def _piper_pcm(self, text: str) -> np.ndarray:
        if not self._piper_ready:
            raise RuntimeError("Piper niedostępny (brak modułu piper albo modelu)")
        with tempfile.TemporaryDirectory(prefix="wvt_tts_") as td:
            wav = Path(td) / "piper.wav"
            cmd = self._piper_cmd() + ["-m", str(self.piper_model), "-f", str(wav), "--", text]
            try:
                res = subprocess.run(cmd, capture_output=True, timeout=PIPER_TIMEOUT_S,
                                     creationflags=_CREATE_NO_WINDOW)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f"Piper nie uruchomił się: {type(exc).__name__}") from exc
            if res.returncode != 0 or not wav.exists():
                detail = (res.stderr or b"").decode("utf-8", errors="replace")[:200].strip()
                raise RuntimeError(f"Piper zwrócił błąd ({res.returncode}){': ' + detail if detail else ''}")
            return self._read_wav(wav)

    # ------------------------------------------------------------------ SAPI
    def _sapi_pcm(self, text: str) -> np.ndarray:
        if not self._sapi_ready:
            raise RuntimeError("SAPI dostępny tylko na Windows")
        with tempfile.TemporaryDirectory(prefix="wvt_tts_") as td:
            wav = Path(td) / "sapi.wav"
            src = Path(td) / "text.txt"
            src.write_text(text, encoding="utf-8")
            cmd = ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                   "-Command", _SAPI_SCRIPT, str(wav), str(src)]
            try:
                res = subprocess.run(cmd, capture_output=True, timeout=SAPI_TIMEOUT_S,
                                     creationflags=_CREATE_NO_WINDOW)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f"PowerShell/SAPI nie uruchomił się: {type(exc).__name__}") from exc
            if res.returncode != 0 or not wav.exists():
                detail = (res.stderr or b"").decode("utf-8", errors="replace")[:200].strip()
                raise RuntimeError(f"SAPI nie zsyntezował mowy{': ' + detail if detail else ''}")
            return self._read_wav(wav)

    # ------------------------------------------------------------------ WAV -> 24 kHz float
    @staticmethod
    def _read_wav(path: Path) -> np.ndarray:
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width != 2:
            raise RuntimeError("tylko 16-bit PCM WAV")
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        return resample_to_24k(x, sr)

    def close(self) -> None:
        """Nothing persistent to release (every synthesis uses a temporary directory)."""
