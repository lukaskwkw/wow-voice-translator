"""Console prototype (spec stage 1): STT / translation / benchmark / addon install.

    python -m wvt.cli check
    python -m wvt.cli stt tests/audio/pl_01.wav
    python -m wvt.cli translate "Siema, idziemy na Deadmines?" --dir pl-en
    python -m wvt.cli mic --seconds 5 --channel party
    python -m wvt.cli bench [--model models/llm/xxx.gguf] [--no-llm] [--no-stt]
    python -m wvt.cli install-addon [--wow "D:/Games/World of Warcraft"]
    python -m wvt.cli router "weź go doproś do party" [--json]
    python -m wvt.cli cloud "Odpowiedz: działa"
    python -m wvt.cli tts --download
    python -m wvt.cli tts --test "Cześć, tu Iskra." [--engine sapi]
    python -m wvt.cli signal [--delay 3] [--out logs/signal_debug.png] [--width 320] [--height 120]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

from .config import ROOT, Config, resolve_path, wow_root_from
from .envfile import load_env
from .logging_setup import setup_logging


# --------------------------------------------------------------------------- helpers

def load_wav(path: str | Path) -> np.ndarray:
    from .audio import resample

    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("only 16-bit PCM WAV supported")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return resample(x, sr)


def normalize_words(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def wer(ref: str, hyp: str) -> float:
    r, h = normalize_words(ref), normalize_words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def add_noise(x: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Pink-ish noise + a low hum, roughly like game music/ambience behind a voice."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(len(x)).astype(np.float32)
    spec = np.fft.rfft(white)
    f = np.arange(len(spec), dtype=np.float32)
    f[0] = 1.0
    pink = np.fft.irfft(spec / np.sqrt(f), n=len(x)).astype(np.float32)
    t = np.arange(len(x)) / 16000.0
    noise = pink / (np.std(pink) + 1e-9) + 0.5 * np.sin(2 * np.pi * 110 * t).astype(np.float32)
    sig_p = np.mean(x**2) + 1e-12
    noise *= np.sqrt(sig_p / (10 ** (snr_db / 10)) / (np.mean(noise**2) + 1e-12))
    return np.clip(x + noise, -1, 1).astype(np.float32)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[k]


def _vram() -> dict | None:
    from .gpu import vram_used_mb

    return vram_used_mb()


def make_llm(cfg: Config, model: str | None = None):
    from .llm import LocalLLM

    llm_cfg = dict(cfg["llm"])
    if model:
        llm_cfg["model_path"] = model
    return LocalLLM(llm_cfg)


def make_stt(cfg: Config):
    from .stt import SpeechToText

    return SpeechToText.from_config(cfg["stt"])


# --------------------------------------------------------------------------- commands

def cmd_check(cfg: Config, _args) -> int:
    print("Python:", sys.version.split()[0])
    v = _vram()
    print("GPU:", v["name"] if v else "brak NVML / GPU", f"| VRAM użyte {v['used']:.0f}/{v['total']:.0f} MB" if v else "")
    try:
        import onnxruntime as ort
        from .gpu import preload_cuda_dlls

        preload_cuda_dlls()
        print("onnxruntime:", ort.__version__, ort.get_available_providers())
    except Exception as exc:
        print("onnxruntime: BŁĄD", exc)
    for label, p in (("STT", cfg.get("stt.model_dir")), ("STT zapas", cfg.get("stt.cpu_fallback.model_dir")),
                     ("llama-server", cfg.get("llm.server_exe")), ("Model LLM", cfg.get("llm.model_path"))):
        if not p:
            continue
        path = resolve_path(p)
        print(f"{label}: {'OK' if path.exists() else 'BRAK'} – {path}")
    try:
        addon = cfg.addon_dir / "WoWVoiceTranslator" / "WoWVoiceTranslator.toc"
        print(f"Addon w WoW: {'OK' if addon.exists() else 'BRAK'} – {addon}")
    except FileNotFoundError as exc:
        print("Addon w WoW: BRAK –", exc)
    try:
        from .audio import AudioRecorder

        print("Mikrofony:", AudioRecorder.list_input_devices())
    except Exception as exc:
        print("Mikrofony: BŁĄD", exc)
    return 0


def cmd_stt(cfg: Config, args) -> int:
    stt = make_stt(cfg)
    stt.load()
    print(f"STT: {stt.describe()}, ładowanie {stt.load_seconds:.1f}s")
    for f in args.files:
        x = load_wav(f)
        t0 = time.perf_counter()
        text = stt.transcribe(x)
        print(f"[{(time.perf_counter() - t0) * 1000:6.0f} ms] {Path(f).name}: {text}")
    return 0


def cmd_translate(cfg: Config, args) -> int:
    from .translator import Translator

    llm = make_llm(cfg, args.model)
    llm.start()
    try:
        tr = Translator(llm, cfg["translation"])
        res = tr.translate(args.text, args.dir)
        print(res.text)
        print(f"[{res.seconds * 1000:.0f} ms] {res.info}")
    finally:
        llm.stop()
    return 0


def cmd_mic(cfg: Config, args) -> int:
    from .audio import AudioRecorder, analyze, trim_and_normalize
    from .translator import Translator

    a = cfg["audio"]
    rec = AudioRecorder(a.get("device"), a.get("sample_rate", 16000), always_open=False)
    stt = make_stt(cfg)
    stt.load()
    llm = make_llm(cfg, args.model)
    llm.start()
    try:
        tr = Translator(llm, cfg["translation"])
        input(f"Enter = start nagrywania ({args.seconds}s)… ")
        x = rec.record_seconds(args.seconds)
        st = analyze(x, a["silence_rms_threshold"])
        print(f"nagranie {st.duration:.1f}s, mowa {st.speech_seconds:.1f}s, rms {st.rms:.4f}")
        clip = trim_and_normalize(x, a["silence_rms_threshold"])
        t0 = time.perf_counter()
        pl = stt.transcribe(clip)
        t1 = time.perf_counter()
        en = tr.translate(pl, "pl-en").text
        t2 = time.perf_counter()
        print(f"PL: {pl}\nEN: {en}\nSTT {1000 * (t1 - t0):.0f} ms | tłumaczenie {1000 * (t2 - t1):.0f} ms")
    finally:
        llm.stop()
    return 0


def cmd_bench(cfg: Config, args) -> int:
    from .translator import Translator

    data = json.loads((ROOT / "wvt" / "data" / "benchmark.json").read_text(encoding="utf-8"))
    report: dict = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "vram_start": _vram()}
    audio_dir = ROOT / "tests" / "audio"
    stt_texts: list[str] = []

    if not args.no_stt:
        stt = make_stt(cfg)
        stt.load()
        stt.warmup()
        report["stt"] = {"device": stt.device, "load_s": round(stt.load_seconds, 2), "vram_after_load": _vram()}
        print(f"== STT {stt.describe()} (ładowanie {stt.load_seconds:.1f}s)")
        rows = []
        for noisy in (False, True):
            lat, wers = [], []
            for i, item in enumerate(data["pl_to_en"], 1):
                wav = audio_dir / f"pl_{i:02d}.wav"
                if not wav.exists():
                    continue
                x = load_wav(wav)
                if noisy:
                    x = add_noise(x, 10.0, seed=i)
                t0 = time.perf_counter()
                hyp = stt.transcribe(x)
                ms = (time.perf_counter() - t0) * 1000
                e = wer(item["pl"], hyp)
                lat.append(ms)
                wers.append(e)
                if not noisy:
                    stt_texts.append(hyp)
                    rows.append({"ref": item["pl"], "hyp": hyp, "ms": round(ms), "wer": round(e, 3), "audio_s": round(len(x) / 16000, 2)})
                    print(f"  {ms:6.0f} ms  WER {e:4.2f}  {hyp}")
            key = "noisy_snr10" if noisy else "clean"
            report["stt"][key] = {
                "n": len(lat),
                "latency_ms_mean": round(statistics.mean(lat), 1) if lat else None,
                "latency_ms_p90": round(pct(lat, 0.9), 1),
                "wer_mean": round(statistics.mean(wers), 3) if wers else None,
            }
            print(f"  -> {key}: śr. {report['stt'][key]['latency_ms_mean']} ms, p90 {report['stt'][key]['latency_ms_p90']} ms, WER {report['stt'][key]['wer_mean']}")
        report["stt"]["utterances"] = rows

    if not args.no_llm:
        llm = make_llm(cfg, args.model)
        t0 = time.perf_counter()
        llm.start()
        report["llm"] = {"model": llm.model_name, "load_s": round(time.perf_counter() - t0, 2), "vram_after_load": _vram()}
        try:
            from .gpu import process_vram_mb

            report["llm"]["server_vram_mb"] = process_vram_mb(llm.pid) if llm.pid else None
            tr = Translator(llm, cfg["translation"])
            tr.translate("Cześć, rozgrzewka.", "pl-en")  # warm-up / prompt cache
            tr.translate("hello, warm up", "en-pl")
            print(f"== LLM {llm.model_name} (start {report['llm']['load_s']} s)")
            for direction, items, src_key, ref_key in (("pl-en", data["pl_to_en"], "pl", "en"), ("en-pl", data["en_to_pl"], "en", "pl")):
                lat, out_rows, tps = [], [], []
                for item in items:
                    tr.cache = type(tr.cache)(tr.cache.size)  # no cache hits in the benchmark
                    res = tr.translate(item[src_key], direction)
                    lat.append(res.seconds * 1000)
                    if res.info.get("predicted_per_second"):
                        tps.append(res.info["predicted_per_second"])
                    out_rows.append({"src": item[src_key], "out": res.text, "ref": item[ref_key], "ms": round(res.seconds * 1000),
                                     "reasoning_chars": res.info.get("reasoning_chars", 0)})
                    print(f"  {res.seconds * 1000:6.0f} ms  {item[src_key]}  =>  {res.text}")
                report["llm"][direction] = {
                    "latency_ms_mean": round(statistics.mean(lat), 1),
                    "latency_ms_p90": round(pct(lat, 0.9), 1),
                    "tokens_per_s_mean": round(statistics.mean(tps), 1) if tps else None,
                    "rows": out_rows,
                }
                print(f"  -> {direction}: śr. {report['llm'][direction]['latency_ms_mean']} ms, p90 {report['llm'][direction]['latency_ms_p90']} ms")
            if stt_texts:
                lat = []
                rows = []
                for hyp in stt_texts:
                    res = tr.translate(hyp, "pl-en")
                    lat.append(res.seconds * 1000)
                    rows.append({"stt": hyp, "out": res.text})
                report["llm"]["pl-en_from_stt"] = {"latency_ms_mean": round(statistics.mean(lat), 1), "rows": rows}
            report["vram_end"] = _vram()
        finally:
            llm.stop()

    out_dir = ROOT / "logs"
    out_dir.mkdir(exist_ok=True)
    tag = Path(args.model or cfg.get("llm.model_path")).stem if not args.no_llm else "stt"
    out = out_dir / f"bench_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Raport:", out)
    return 0


def install_addon(cfg: Config) -> Path:
    src = ROOT / "addon" / "WoWVoiceTranslator"
    dst_root = cfg.addon_dir
    if not dst_root.parent.parent.exists():
        raise FileNotFoundError(f"Nie znaleziono katalogu WoW: {dst_root.parent.parent}")
    dst = dst_root / "WoWVoiceTranslator"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    return dst


def cmd_install_addon(cfg: Config, args) -> int:
    if getattr(args, "wow", None):
        root = wow_root_from(args.wow)
        flavor = cfg.get("wow.flavor_dir", "_forever_")
        if not (root / flavor).is_dir():
            print(f"W {root} nie ma folderu {flavor} – podaj folder, w którym jest {flavor} (WoW Forever).")
            return 1
        cfg.set("wow.install_dir", str(root).replace("\\", "/"))
        cfg.save()  # remembered for the app and later installs
    try:
        dst = install_addon(cfg)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    print("Addon zainstalowany w:", dst)
    print("W grze: /reload (albo restart), a potem /wvt test")
    return 0


def cmd_router(cfg: Config, args) -> int:
    """What the local rules understood, then what the Jev router makes of the same words (assistant.router: jev)."""
    from .assistant import INTENT_CRITERIA, intent_from_label, parse_intent
    from .jev import JevError, jev_router

    text = " ".join(args.text).strip()
    local = parse_intent(text)
    print(f"local: {local.kind}" + (f" (spoken: {local.spoken})" if local.spoken else ""))
    router = jev_router(cfg)
    if router is None:
        print('Jev: wyłączony (assistant.router != "jev" albo brak klucza)')
        return 0
    print(f"Router: {router.model_name}")
    try:
        decision = router.classify(text, INTENT_CRITERIA)
    except JevError as exc:
        print(f"Jev: błąd: {exc}")
        return 1
    finally:
        router.close()
    if decision is None:
        print("Jev: brak pewnej komendy")
        return 0
    print(f"Jev: {decision.label} confidence={decision.confidence:.2f} {decision.ms} ms "
          f"is_command={decision.is_command} provider={decision.provider}")
    mapped = intent_from_label(decision.label, text)
    print(f"mapped: {mapped.kind}")
    if args.json:
        print(json.dumps({"label": decision.label, "confidence": decision.confidence,
                          "probabilities": decision.probabilities, "mapped": mapped.kind,
                          "provider": decision.provider}, ensure_ascii=False))
    return 0


PIPER_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium"


def _download(url: str, dest: Path) -> bool:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                block = resp.read(1 << 16)
                if not block:
                    break
                fh.write(block)
                done += len(block)
        tmp.replace(dest)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"BŁĄD pobierania {dest.name}: {type(exc).__name__}: {exc}")
        return False
    size = dest.stat().st_size
    print(f"Pobrano {dest.name}: {size / 1e6:.1f} MB" + (f" z {total / 1e6:.1f} MB" if total else ""))
    return True


def cmd_tts(cfg: Config, args) -> int:
    """Local voice for Iskra: download the Polish Piper voice or synthesize a test WAV."""
    from .voice.local import SAMPLE_RATE, LocalTTS

    local_cfg = dict(cfg.get("voice.local") or {})
    if getattr(args, "engine", None):
        local_cfg["engine"] = args.engine
    model = resolve_path(local_cfg.get("piper_model", "models/piper/pl_PL-gosia-medium.onnx"))
    if args.download:
        ok = _download(PIPER_URL + ".onnx", model)
        ok = _download(PIPER_URL + ".onnx.json", model.with_suffix(model.suffix + ".json")) and ok
        if not ok:
            print("Piper: model niepobrany – SAPI Paulina nadal działa (python -m wvt.cli tts --test ...).")
    tts = LocalTTS(local_cfg)
    if not tts.available():
        print("Brak lokalnego TTS: zainstaluj Piper (pip install -r requirements-tts.txt) i pobierz model "
              "(python -m wvt.cli tts --download) albo uruchom na Windows z głosem Paulina.")
        return 1
    if not args.download and not tts._piper_ready:
        print("Piper: niedostępny (brak modułu piper lub modelu) – używam SAPI, jeśli jest.")
    if not args.test:
        print(f"Silnik: {tts.describe()} (model_id {tts.model_id}), model: {model}")
        return 0
    text = args.test.strip()
    if not text:
        print("Podaj tekst do wypowiedzenia: --test \"...\"")
        return 1
    import wave as _wave

    stats: list = []
    chunks = list(tts.stream_tts("local:pl", text, "pl", on_stats=stats.append))
    pcm = b"".join(chunks)
    out = resolve_path("data/tts_test.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    with _wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    seconds = len(pcm) / 2 / SAMPLE_RATE
    st = stats[0] if stats else None
    print(f"Zapisano: {out} ({seconds:.2f} s, {len(text)} znaków, silnik {tts.describe()})")
    if st:
        print(f"Pierwszy dźwięk {st.ttfa_ms:.0f} ms, synteza {st.total_ms:.0f} ms, koszt {st.cost:.2f}")
    return 0


def cmd_cloud(cfg: Config, args) -> int:
    """One question to the OpenAI-compatible cloud model (assistant.openai) - no local models are loaded."""
    from .envfile import secret
    from .gemini import CloudError
    from .openai_compat import OpenAICompatLLM

    text = " ".join(args.text).strip()
    o = cfg.get("assistant.openai", {}) or {}
    env_name = o.get("api_key_env", "OPENCODE_API_KEY")
    key = secret(env_name)
    if not key:
        print(f"Brak klucza {env_name} w .env")
        return 1
    client = OpenAICompatLLM(key, o.get("model", "deepseek-v4.1-flash"),
                             o.get("base_url", "https://opencode.ai/inference/openai/v1"),
                             float(o.get("timeout_s", 20)), int(o.get("max_output_tokens", 4096)),
                             reasoning_effort=str(o.get("reasoning_effort", "low") or ""))
    try:
        prompt = "Odpowiedz jednym krótkim zdaniem po polsku: " + text
        answer, meta = client.chat([{"role": "user", "content": prompt}], max_tokens=200)
        print(answer)
        print(meta)
    except CloudError as exc:
        print(f"Chmura: błąd: {exc}")
        return 1
    finally:
        client.close()
    return 0


def _save_rgb_png(img: np.ndarray, path: Path) -> bool:
    """Write an RGB numpy image as PNG through Qt (no app instance needed)."""
    from PySide6.QtGui import QImage

    h, w = img.shape[:2]
    qimg = QImage(img.data, w, h, int(img.strides[0]), QImage.Format.Format_RGB888).copy()
    return qimg.save(str(path), "PNG")


def _save_rgb_ppm(img: np.ndarray, path: Path) -> None:
    h, w = img.shape[:2]
    with path.open("wb") as fh:
        fh.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        fh.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())


def cmd_signal(cfg: Config, args) -> int:
    """Grab the pixel-signal grid from WoW's top-left corner and save it, to see why nothing is read."""
    from . import protocol, winapi
    from .signal_reader import ScreenCapture, TYPE_NAMES
    from .wow_window import WowWindowTracker

    if args.delay > 0:
        print(f"Czekam {args.delay:g} s - najedz na linie czatu w WoW...")
        time.sleep(args.delay)
    tracker = WowWindowTracker(cfg.get("wow.process_names", []) or [], [])
    win = tracker.find(max_age=0)
    if win is None:
        print("Nie znaleziono okna WoW (gra nie działa?).")
        return 1
    x, y, client_w, client_h = win.client
    print(f"hwnd: {win.hwnd}")
    print(f"proces: {win.process}")
    print(f"client: x={x} y={y} w={client_w} h={client_h}")
    print(f"foreground: {winapi.foreground_window() == win.hwnd}")
    w = min(args.width, client_w)
    h = min(args.height, client_h)
    if w <= 0 or h <= 0:
        print(f"Obszar klienta za mały: {client_w}x{client_h}")
        return 1
    cap = ScreenCapture()
    try:
        img = cap.grab(x, y, w, h)
    except OSError as exc:
        print(f"Przechwycenie nieudane: {exc}")
        return 1
    finally:
        cap.close()

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if _save_rgb_png(img, out):
        print(f"PNG: {out}")
    else:
        out = out.with_suffix(".ppm")
        _save_rgb_ppm(img, out)
        print(f"PNG nieudany, zapisano PPM: {out}")
    up = np.repeat(np.repeat(img, 4, axis=0), 4, axis=1)
    up_out = out.with_name(out.stem + "_4x.png")
    if _save_rgb_png(up, up_out):
        print(f"PNG 4x: {up_out}")
    else:
        up_out = up_out.with_suffix(".ppm")
        _save_rgb_ppm(up, up_out)
        print(f"PNG 4x nieudany, zapisano PPM: {up_out}")

    print(f"obraz (h, w, kanaly): {img.shape}")
    colors = np.unique(img.reshape(-1, 3), axis=0)
    print(f"unikalne kolory RGB: {len(colors)}")
    cell = protocol.CELL_PX
    if img.shape[0] >= cell and img.shape[1] >= protocol.CALIB_CELLS * cell:
        block = img[0:cell, 0:protocol.CALIB_CELLS * cell].reshape(cell, protocol.CALIB_CELLS, cell, 3)
        means = block.mean(axis=(0, 2))
        print(f"kalibracja ({protocol.CALIB_CELLS} komorki {cell}x{cell}, srednie RGB):")
        for i, m in enumerate(means):
            print(f"  cell {i}: ({m[0]:.1f}, {m[1]:.1f}, {m[2]:.1f})")
    else:
        print("kalibracja: obraz za mały")

    names = {protocol.TYPE_HOVER: "hover", protocol.TYPE_KEEP: "keep", protocol.TYPE_TEST: "test"}
    names.update(TYPE_NAMES)
    pkt = protocol.try_decode(img)
    if pkt is not None:
        print(f"dekodowanie: {names.get(pkt.type, str(pkt.type))} seq={pkt.seq} tekst={len(pkt.text)} znakow")
    else:
        reason = "brak dopasowanej siatki"
        try:
            protocol.decode_image(img)
        except protocol.DecodeError as exc:
            reason = str(exc)
        print(f"dekodowanie: nic ({reason})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m wvt.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("stt")
    p.add_argument("files", nargs="+")
    p = sub.add_parser("translate")
    p.add_argument("text")
    p.add_argument("--dir", default="pl-en", choices=["pl-en", "en-pl"])
    p.add_argument("--model")
    p = sub.add_parser("mic")
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--model")
    p = sub.add_parser("bench")
    p.add_argument("--model")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--no-stt", action="store_true")
    p = sub.add_parser("install-addon")
    p.add_argument("--wow", help="folder World of Warcraft (ten z _forever_), gdy nie wykryje się sam")
    p = sub.add_parser("router")
    p.add_argument("text", nargs="+")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("cloud")
    p.add_argument("text", nargs="+")
    p = sub.add_parser("tts")
    p.add_argument("--download", action="store_true", help="pobierz polski głos Piper pl_PL-gosia-medium")
    p.add_argument("--test", metavar="TEXT", help="syntezuj tekst do data/tts_test.wav")
    p.add_argument("--engine", choices=["auto", "piper", "sapi"], help="nadpisz voice.local.engine")
    p = sub.add_parser("signal")
    p.add_argument("--delay", type=float, default=3.0, help="ile sekund czekać (najedź na linię czatu w WoW)")
    p.add_argument("--out", default="logs/signal_debug.png", help="plik PNG z podglądem siatki")
    p.add_argument("--width", type=int, default=320, help="szerokość przechwycenia w pikselach")
    p.add_argument("--height", type=int, default=120, help="wysokość przechwycenia w pikselach")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # Polish text in a cp1250 console
        except (AttributeError, ValueError):
            pass

    load_env()  # .env before anything reads a secret (Config.load itself needs no key)
    cfg = Config.load()
    setup_logging(cfg, console=True)
    return {
        "check": cmd_check,
        "stt": cmd_stt,
        "translate": cmd_translate,
        "mic": cmd_mic,
        "bench": cmd_bench,
        "install-addon": cmd_install_addon,
        "router": cmd_router,
        "cloud": cmd_cloud,
        "tts": cmd_tts,
        "signal": cmd_signal,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
