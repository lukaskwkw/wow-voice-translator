"""Push-to-talk pipeline: record -> speech recognition (Whisper, PL) -> LLM (EN) -> WoW chat input.

UI-agnostic: progress is reported through the ``ui`` callbacks object, which the Qt
app implements with thread-safe signals.  One job at a time (spec: single-task queue).

With the names of the moment (``lexicon``, wvt/lexicon.py) Whisper gets them as its prompt, misheard names
are put right before the line is translated ("Targasa" -> "Thargas"), and the translator is told how the
game spells them; the overlay shows what was heard when a name was changed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

from . import audio as audio_mod
from .chat_insert import ChatInserter, build_chat_line, sanitize_chat_text, set_clipboard_text

log = logging.getLogger("wvt.pipeline")


class PipelineUI(Protocol):
    def status(self, kind: str, title: str, lines: list[str] | None = None, timeout: float = 0.0) -> None: ...
    def result(self, item: "HistoryItem") -> None: ...


@dataclass
class HistoryItem:
    channel: str
    polish: str
    english: str
    chat_line: str
    timings: dict = field(default_factory=dict)
    status: str = ""
    created: float = field(default_factory=time.time)


class PushToTalkPipeline:
    def __init__(self, cfg, recorder: audio_mod.AudioRecorder, stt, translator, inserter: ChatInserter,
                 ui: PipelineUI, wow_foreground: Callable[[], bool],
                 confirm: Callable[[HistoryItem], None] | None = None,
                 confirm_assistant: Callable[[str, dict], None] | None = None):
        self.cfg = cfg
        self.recorder = recorder
        self.stt = stt
        self.translator = translator
        self.inserter = inserter
        self.ui = ui
        self.wow_foreground = wow_foreground
        self.confirm = confirm
        self.confirm_assistant = confirm_assistant
        self._pending_confirm: HistoryItem | None = None
        self._pending_assistant: tuple[str, dict] | None = None
        self._skip_rules = False  # the next assistant question goes to chat only (no rules, no Jev)
        self.history: deque[HistoryItem] = deque(maxlen=10)
        self._busy = threading.Lock()
        self._recording_channel: Optional[str] = None
        self._rec_started = 0.0
        self.models_ready = threading.Event()
        self.last_timings: dict = {}
        self.assistant = None  # wvt.assistant.Assistant: the "assistant" push-to-talk channel goes to her
        self.chat_log = None  # wvt.assistant.ChatLog: recent chat, the context of what you say next
        self.lexicon = None  # wvt.lexicon.Lexicon: the names of the moment (Whisper's prompt, name fixes)

    def arm_skip_rules(self) -> None:
        """The next assistant question goes to chat only: no local rules, no Jev."""
        self._skip_rules = True

    def clear_skip_rules(self) -> None:
        self._skip_rules = False

    @property
    def skip_rules_pending(self) -> bool:
        return self._skip_rules

    def _take_skip_rules(self) -> bool:
        skip, self._skip_rules = self._skip_rules, False
        return skip

    def _prompt(self) -> str:
        if self.lexicon is None or not getattr(self.stt, "prompt_ok", False):
            return ""
        try:
            return self.lexicon.whisper_prompt()
        except Exception:
            log.exception("lexicon prompt failed")
            return ""

    def _correct(self, polish: str):
        if self.lexicon is None:
            return None
        try:
            return self.lexicon.correct(polish, strict=True)  # the line goes out: only close matches
        except Exception:
            log.exception("lexicon correction failed")
            return None

    @staticmethod
    def _heard(corr) -> list[str]:
        if corr is None or not corr.fixes:
            return []
        return ["usłyszane: " + ", ".join(f"{f.heard} → {f.name}" for f in corr.fixes)]

    # ------------------------------------------------------------------ hotkey events
    @property
    def recording(self) -> bool:
        return self._recording_channel is not None

    def on_press(self, channel: str) -> None:
        if not self.models_ready.is_set():
            self.ui.status("busy", "Modele jeszcze się ładują…", timeout=2.5)
            return
        if self._pending_confirm is not None:
            self.ui.status("busy", "Dokończ edycję tekstu…", timeout=2.0)
            return
        if self._pending_assistant is not None:
            self.ui.status("busy", "Dokończ tekst dla Iskry…", timeout=2.0)
            return
        if self._busy.locked():
            self.ui.status("busy", "Poczekaj – poprzednia wypowiedź jeszcze się tłumaczy", timeout=2.0)
            return
        try:
            self.recorder.start()
        except Exception as exc:
            self.ui.status("error", "Mikrofon niedostępny", [str(exc)], timeout=6)
            return
        self._recording_channel = channel
        self._rec_started = time.perf_counter()
        self.ui.status("rec", f"Nagrywanie → {self._channel_label(channel)}", ["puść klawisz, aby przetłumaczyć · Esc anuluje"])

    def on_cancel(self) -> None:
        if self._recording_channel is None:
            return
        self.recorder.cancel()
        self._recording_channel = None
        self.ui.status("info", "Anulowano", timeout=1.5)

    def on_release(self, channel: str) -> None:
        if self._recording_channel is None:
            return
        channel = self._recording_channel
        self._recording_channel = None
        audio = self.recorder.stop()
        released_at = time.perf_counter()
        threading.Thread(target=self._process, args=(channel, audio, released_at), name="wvt-ptt", daemon=True).start()

    # ------------------------------------------------------------------ processing
    def _channel_label(self, channel: str) -> str:
        if channel == "assistant":
            return str(self.cfg.get("assistant.name", "Asystentka"))
        cmd = self.cfg.get(f"channels.{channel}", "")
        return f"{channel.upper()} ({cmd})" if cmd else channel.upper()

    # chat kinds that belong to the conversation of each push-to-talk channel
    CONTEXT_KINDS = {
        "say": ("SAY", "YELL"), "yell": ("SAY", "YELL"),
        "party": ("PARTY", "PARTY_LEADER"), "raid": ("RAID", "RAID_LEADER", "PARTY", "PARTY_LEADER"),
        "reply": ("WHISPER", "WHISPER_INFORM"),
    }

    def conversation(self, channel: str) -> list[str]:
        """The last lines of this channel's conversation ("Ann: ...", "You: ...") for the translator."""
        kinds = self.CONTEXT_KINDS.get(channel)
        if self.chat_log is None or not kinds:
            return []
        lines = self.chat_log.recent(n=int(self.cfg.get("translation.context_lines", 6)), kinds=kinds,
                                     max_age_s=float(self.cfg.get("translation.context_seconds", 180)))
        if channel == "reply" and lines:  # /r answers whoever whispered you last: only that conversation
            partner = next((ln.sender for ln in reversed(lines) if ln.kind == "WHISPER"), "")
            lines = [ln for ln in lines if ln.sender == partner]
        return [f"{'You' if ln.is_self else ln.sender.split('-')[0]}: {ln.text}" for ln in lines]

    def process_audio(self, audio: np.ndarray, channel: str, insert: bool = True) -> Optional[HistoryItem]:
        """Synchronous variant used by the CLI and the settings 'test' button."""
        return self._process(channel, audio, time.perf_counter(), insert=insert)

    def _process(self, channel: str, audio: np.ndarray, released_at: float, insert: bool = True) -> Optional[HistoryItem]:
        if not self._busy.acquire(blocking=False):
            self.ui.status("busy", "Poczekaj – poprzednia wypowiedź jeszcze się tłumaczy", timeout=2.0)
            return None
        timings: dict = {}
        polish = ""
        try:
            acfg = self.cfg["audio"]
            threshold = float(acfg.get("silence_rms_threshold", 0.006))
            raw = audio_mod.analyze(audio, threshold)
            audio, gain_db = audio_mod.input_gain(audio, threshold, float(acfg.get("input_gain_db", 0.0) or 0.0),
                                                  bool(acfg.get("auto_gain", True)))
            stats = audio_mod.analyze(audio, threshold)
            log.info("audio %.1fs: level %.0f dBFS (peak %.0f), gain %+.0f dB, speech %.1fs", raw.duration,
                     audio_mod.dbfs(raw.rms), audio_mod.dbfs(raw.peak), gain_db, stats.speech_seconds)
            timings["audio_s"] = round(stats.duration, 2)
            if stats.duration < 0.25 or stats.speech_seconds < float(acfg.get("min_speech_seconds", 0.35)):
                self.ui.status("info", "Nie usłyszałem mowy", [f"nagranie {stats.duration:.1f}s, poziom {stats.rms:.4f} – sprawdź mikrofon/próg ciszy"], timeout=3.5)
                return None
            clip = audio_mod.trim_and_normalize(audio, threshold)

            self.ui.status("busy", "Rozpoznawanie mowy…")
            t0 = time.perf_counter()
            prompt = self._prompt()
            polish = self.stt.transcribe(clip, prompt=prompt) if prompt else self.stt.transcribe(clip)
            timings["stt_ms"] = round((time.perf_counter() - t0) * 1000)
            if not polish:
                self.ui.status("info", "Nic nie rozpoznano", timeout=3)
                return None

            if channel == "assistant":  # her own key: no translation, she decides what to do
                if self.assistant is None:
                    self.ui.status("info", "Asystentka jest wyłączona", ["assistant.enabled w config.json"], timeout=4)
                    return None
                if self.confirm_assistant is not None and bool(self.cfg.get("assistant.confirm_transcript", True)):
                    timings["release_to_assistant_ms"] = round((time.perf_counter() - released_at) * 1000)
                    self.last_timings = dict(timings)
                    self._pending_assistant = (polish, dict(timings))
                    self.ui.status("busy", "Popraw tekst dla Iskry i wciśnij Enter…", ["Esc anuluje"], timeout=0.0)
                    try:
                        self.confirm_assistant(polish, dict(timings))
                    except Exception:
                        self._pending_assistant = None
                        raise
                    return None
                self.ui.status("busy", f"{self._channel_label(channel)} myśli…", [f"PL: {polish}"])
                t1 = time.perf_counter()
                reply, line = self.assistant.handle(polish, force_chat=self._take_skip_rules())
                timings["assistant_ms"] = round((time.perf_counter() - t1) * 1000)
                self.last_timings = timings
                corr = getattr(self.assistant, "last_correction", None)
                shown = corr.text if corr is not None and corr.fixes else polish
                item = HistoryItem(channel, shown, reply, line, timings, "assistant")
                if self.cfg.get("privacy.keep_history", True):
                    self.history.appendleft(item)
                self.ui.result(item)
                return item

            corr = self._correct(polish)
            if corr is not None and corr.fixes:
                timings["names_fixed"] = len(corr.fixes)
                polish = corr.text
            self.ui.status("busy", "Tłumaczenie…", [f"PL: {polish}"] + self._heard(corr))
            t1 = time.perf_counter()
            context = self.conversation(channel)
            names = self.lexicon.translation_names(corr) if self.lexicon is not None else []
            res = self.translator.translate(polish, "pl-en", context=context, names=names) if names else \
                self.translator.translate(polish, "pl-en", context=context)
            timings["context_lines"] = len(context)
            timings["translate_ms"] = round((time.perf_counter() - t1) * 1000)
            english = res.text
            if not english:
                raise RuntimeError("model zwrócił pustą odpowiedź")

            cmd = self.cfg.get(f"channels.{channel}", "")
            chat_line = build_chat_line(cmd, english, int(self.cfg.get("translation.max_chat_chars", 255)))
            # hovering our own message later shows the original Polish instantly
            self.translator.remember(chat_line[len(cmd) + 1:] if cmd else chat_line, polish, "en-pl")

            if insert and self.confirm is not None and bool(self.cfg.get("insert.confirm", False)):
                timings["release_to_text_ms"] = round((time.perf_counter() - released_at) * 1000)
                self.last_timings = timings
                item = HistoryItem(channel, polish, english, chat_line, timings, "pending")
                self._pending_confirm = item
                self.ui.status("busy", "Popraw tekst i wciśnij Enter…", ["Esc anuluje"], timeout=0.0)
                try:
                    self.confirm(item)
                except Exception:
                    self._pending_confirm = None
                    raise
                return item

            status = "preview"
            if insert:
                if not cmd:
                    # no channel prefix (the "free" channel): never go through Enter-first insertion,
                    # which would send whatever is already typed in the chat box - copy it instead
                    set_clipboard_text(chat_line)
                    status = "free"
                else:
                    t2 = time.perf_counter()
                    status = self.inserter.insert(chat_line, self.wow_foreground())
                    timings["insert_ms"] = round((time.perf_counter() - t2) * 1000)
            timings["release_to_text_ms"] = round((time.perf_counter() - released_at) * 1000)
            self.last_timings = timings
            item = HistoryItem(channel, polish, english, chat_line, timings, status)
            if self.cfg.get("privacy.keep_history", True):
                self.history.appendleft(item)
            if self.cfg.get("privacy.log_texts", False):
                log.info("PTT %s | PL=%r | EN=%r | %s", channel, polish, english, timings)
            else:
                log.info("PTT %s | %d->%d chars | %s", channel, len(polish), len(english), timings)
            footer = {
                "inserted": "W czacie – Enter wysyła, Esc anuluje",
                "copied": "Skopiowano do schowka (WoW nie był aktywny) – wklej Ctrl+V",
                "free": "Skopiowano – wklej w wybranym czacie (Ctrl+V)",
                "preview": "Podgląd",
            }.get(status, status)
            self.ui.status("ok", footer, [f"PL: {polish}"] + self._heard(corr) + [
                f"EN: {chat_line}",
                f"⏱ STT {timings.get('stt_ms', 0)} ms · tłumaczenie {timings.get('translate_ms', 0)} ms"],
                timeout=float(self.cfg.get("hud.show_seconds", 6)))
            self.ui.result(item)
            return item
        except Exception as exc:
            log.exception("PTT pipeline failed")
            lines = [str(exc)]
            if polish:
                set_clipboard_text(polish)
                lines.append(f"Transkrypcja skopiowana do schowka: {polish}")
            self.ui.status("error", "Błąd – nic nie zostało wysłane", lines, timeout=8)
            return None
        finally:
            self._busy.release()

    # ------------------------------------------------------------------ confirm before insert
    def finish_confirm(self, item: HistoryItem, text: str) -> str | None:
        if item is not self._pending_confirm:
            self.ui.status("info", "Nieaktualny tekst – zignorowano", timeout=2.0)
            return None
        self._pending_confirm = None
        max_bytes = int(self.cfg.get("translation.max_chat_chars", 255))
        edited = sanitize_chat_text(text, max_bytes)
        if not edited.strip():
            self.ui.status("info", "Pusty tekst – anulowano", timeout=2.0)
            return None
        cmd = self.cfg.get(f"channels.{item.channel}", "")
        prefix = f"{cmd} " if cmd else ""
        body = edited[len(prefix):] if prefix and edited.startswith(prefix) else edited
        item.english = body
        item.chat_line = edited
        # hovering the edited line keeps showing the original Polish instantly
        self.translator.remember(body, item.polish, "en-pl")
        self._busy.acquire()
        try:
            t2 = time.perf_counter()
            if not cmd and not edited.startswith("/"):
                # the free channel: no prefix and none typed - copy, because Enter-first insertion
                # would send whatever is already typed in the chat box
                set_clipboard_text(edited)
                status = "free"
            else:
                try:
                    status = self.inserter.insert(edited, self.wow_foreground())
                except Exception as exc:
                    log.exception("insert failed")
                    self.ui.status("error", "Błąd – nic nie zostało wysłane", [str(exc)], timeout=8)
                    return None
                item.timings["insert_ms"] = round((time.perf_counter() - t2) * 1000)
            item.status = status
            if self.cfg.get("privacy.keep_history", True):
                self.history.appendleft(item)
            if self.cfg.get("privacy.log_texts", False):
                log.info("PTT %s | PL=%r | EN=%r | %s", item.channel, item.polish, body, item.timings)
            else:
                log.info("PTT %s | %d->%d chars | %s", item.channel, len(item.polish), len(body), item.timings)
            footer = {"inserted": "W czacie – Enter wysyła, Esc anuluje",
                      "copied": "Skopiowano do schowka (WoW nie był aktywny) – wklej Ctrl+V",
                      "free": "Skopiowano – wklej w wybranym czacie (Ctrl+V)"}.get(status, status)
            self.ui.status("ok", footer, [f"PL: {item.polish}", f"EN: {item.chat_line}",
                          f"⏱ STT {item.timings.get('stt_ms', 0)} ms · tłumaczenie {item.timings.get('translate_ms', 0)} ms"],
                          timeout=float(self.cfg.get("hud.show_seconds", 6)))
            self.ui.result(item)
            return status
        finally:
            self._busy.release()

    def cancel_confirm(self, item: HistoryItem) -> None:
        if item is not self._pending_confirm:
            return
        self._pending_confirm = None
        self.ui.status("info", "Anulowano", timeout=1.5)

    # ------------------------------------------------------------------ confirm the assistant transcript
    def finish_assistant(self, text: str, timings: dict | None = None) -> HistoryItem | None:
        if self._pending_assistant is None:
            self.ui.status("info", "Nieaktualny tekst – zignorowano", timeout=2.0)
            return None
        self._pending_assistant = None
        edited = (text or "").strip()
        if not edited:
            self.ui.status("info", "Pusty tekst – anulowano", timeout=2.0)
            return None
        if self.assistant is None:
            self.ui.status("info", "Asystentka jest wyłączona", ["assistant.enabled w config.json"], timeout=4)
            return None
        timings = dict(timings or {})
        self._busy.acquire()
        try:
            self.ui.status("busy", f"{self._channel_label('assistant')} myśli…", [f"PL: {edited}"])
            t1 = time.perf_counter()
            reply, line = self.assistant.handle(  # speaks and inserts the command line herself
                edited, force_chat=self._take_skip_rules())
            timings["assistant_ms"] = round((time.perf_counter() - t1) * 1000)
            self.last_timings = timings
            corr = getattr(self.assistant, "last_correction", None)
            shown = corr.text if corr is not None and corr.fixes else edited
            item = HistoryItem("assistant", shown, reply, line, timings, "assistant")
            if self.cfg.get("privacy.keep_history", True):
                self.history.appendleft(item)
            self.ui.result(item)
            return item
        except Exception as exc:
            log.exception("assistant pipeline failed")
            self.ui.status("error", "Błąd – nic nie zostało wysłane", [str(exc)], timeout=8)
            return None
        finally:
            self._busy.release()

    def cancel_assistant(self) -> None:
        self._pending_assistant = None
        self.ui.status("info", "Anulowano", timeout=1.5)

    # ------------------------------------------------------------------ typed question
    def ask_assistant(self, text: str) -> HistoryItem | None:
        """A question typed in the history overlay: no STT, no transcript confirm."""
        text = " ".join((text or "").split()).strip()
        if not text:
            return None
        if self._pending_confirm is not None or self._pending_assistant is not None:
            self.ui.status("busy", "Dokończ edycję tekstu…", timeout=2.0)
            return None
        if not self._busy.acquire(blocking=False):
            self.ui.status("busy", "Poczekaj – poprzednia wypowiedź jeszcze się tłumaczy", timeout=2.0)
            return None
        try:
            if self.assistant is None:
                self.ui.status("info", "Asystentka jest wyłączona", ["assistant.enabled w config.json"], timeout=4)
                return None
            self.ui.status("busy", f"{self._channel_label('assistant')} myśli…", [f"PL: {text}"])
            t1 = time.perf_counter()
            reply, line = self.assistant.handle(text, force_chat=self._take_skip_rules())
            timings = {"assistant_ms": round((time.perf_counter() - t1) * 1000)}
            self.last_timings = timings
            corr = getattr(self.assistant, "last_correction", None)
            shown = corr.text if corr is not None and corr.fixes else text
            item = HistoryItem("assistant", shown, reply, line, timings, "assistant")
            if self.cfg.get("privacy.keep_history", True):
                self.history.appendleft(item)
            self.ui.result(item)
            return item
        except Exception:
            log.exception("typed assistant question failed")
            self.ui.status("error", "Błąd – nic nie zostało wysłane", timeout=8)
            return None
        finally:
            self._busy.release()
