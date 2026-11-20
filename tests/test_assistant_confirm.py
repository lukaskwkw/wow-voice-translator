"""Tests for the assistant transcript confirm step (no Qt)."""

from __future__ import annotations

import threading
from collections import deque

from wvt.pipeline import PushToTalkPipeline


class Cfg:
    def __init__(self, data=None):
        self.data = data or {}

    def get(self, dotted, default=None):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def __getitem__(self, key):
        return self.get(key) if "." in key else self.data[key]


class FakeUI:
    def __init__(self):
        self.statuses = []
        self.results = []

    def status(self, kind, title, lines=None, timeout=0.0):
        self.statuses.append((kind, title))

    def result(self, item):
        self.results.append(item)


class FakeAssistant:
    def __init__(self):
        self.calls = []
        self.forced = []
        self.last_correction = None

    def handle(self, polish, force_chat=False):
        self.calls.append(polish)
        self.forced.append(force_chat)
        return "reply", ""


class FakeInserter:
    def __init__(self):
        self.calls = []

    def insert(self, chat_line, wow_is_foreground):
        self.calls.append((chat_line, wow_is_foreground))
        return "inserted"


def make_pipeline(pending=("stary", {"stt_ms": 100})):
    p = PushToTalkPipeline.__new__(PushToTalkPipeline)
    p.cfg = Cfg({"privacy": {"keep_history": True, "log_texts": False},
                 "assistant": {"name": "Iskra", "confirm_transcript": True},
                 "hud": {"show_seconds": 6}})
    p.assistant = FakeAssistant()
    p.inserter = FakeInserter()
    p.ui = FakeUI()
    p.wow_foreground = lambda: True
    p._busy = threading.Lock()
    p._pending_confirm = None
    p._pending_assistant = pending
    p._skip_rules = False
    p.history = deque(maxlen=10)
    p.last_timings = {}
    p.lexicon = None
    p.translator = None
    return p


def test_finish_assistant_uses_the_edited_text():
    p = make_pipeline()
    before = dict(p._pending_assistant[1])

    item = p.finish_assistant("poprawiony", dict(before))

    assert p.assistant.calls == ["poprawiony"]
    assert item is not None
    assert item.channel == "assistant" and item.status == "assistant"
    assert item.polish == "poprawiony"
    assert item.english == "reply"
    assert item.timings["stt_ms"] == 100
    assert "assistant_ms" in item.timings
    assert list(p.history) == [item]
    assert p.ui.results == [item]
    assert p._pending_assistant is None
    assert p.last_timings is item.timings
    assert not p._busy.locked()


def test_finish_assistant_strips_whitespace():
    p = make_pipeline()

    item = p.finish_assistant("  poprawiony  ", {"stt_ms": 100})

    assert p.assistant.calls == ["poprawiony"]
    assert item.polish == "poprawiony"


def test_cancel_assistant_clears_pending_without_calling_her():
    p = make_pipeline()

    p.cancel_assistant()

    assert p.assistant.calls == []
    assert p._pending_assistant is None
    assert p.ui.statuses[-1] == ("info", "Anulowano")
    assert p.history == deque(maxlen=10)


def test_finish_assistant_without_pending_is_stale():
    p = make_pipeline(pending=None)

    assert p.finish_assistant("cokolwiek") is None
    assert p.assistant.calls == []
    assert p.ui.statuses[-1] == ("info", "Nieaktualny tekst – zignorowano")


def test_finish_assistant_with_empty_text_cancels():
    p = make_pipeline()

    assert p.finish_assistant("   ") is None
    assert p.assistant.calls == []
    assert p._pending_assistant is None
    assert p.ui.statuses[-1] == ("info", "Pusty tekst – anulowano")


def test_on_press_is_refused_while_the_transcript_is_pending():
    p = make_pipeline()
    p.models_ready = threading.Event()
    p.models_ready.set()
    p._recording_channel = None

    class Recorder:
        def start(self):
            raise AssertionError("must not record while the transcript is being edited")

    p.recorder = Recorder()

    p.on_press("assistant")

    assert p.ui.statuses[-1] == ("busy", "Dokończ tekst dla Iskry…")
    assert p._pending_assistant is not None


def test_ask_assistant_typed_question_runs_her():
    p = make_pipeline(pending=None)

    item = p.ask_assistant("  jak    leci quest?  ")

    assert p.assistant.calls == ["jak leci quest?"]
    assert item is not None
    assert item.channel == "assistant" and item.status == "assistant"
    assert item.polish == "jak leci quest?"
    assert item.english == "reply"
    assert "assistant_ms" in item.timings
    assert list(p.history) == [item]
    assert p.ui.results == [item]
    assert p.last_timings is item.timings
    assert not p._busy.locked()


def test_ask_assistant_empty_text_does_nothing():
    p = make_pipeline(pending=None)

    assert p.ask_assistant("   ") is None
    assert p.assistant.calls == []
    assert p.history == deque(maxlen=10)


def test_ask_assistant_busy_lock_refuses():
    p = make_pipeline(pending=None)
    p._busy.acquire()
    try:
        assert p.ask_assistant("pytanie") is None
    finally:
        p._busy.release()

    assert p.assistant.calls == []
    assert p.ui.statuses[-1][0] == "busy"


def test_ask_assistant_without_assistant_status():
    p = make_pipeline(pending=None)
    p.assistant = None

    assert p.ask_assistant("pytanie") is None
    assert p.ui.statuses[-1] == ("info", "Asystentka jest wyłączona")
    assert not p._busy.locked()


def test_ask_assistant_refused_while_transcript_pending():
    p = make_pipeline(pending=("stary", {}))

    assert p.ask_assistant("pytanie") is None
    assert p.assistant.calls == []
    assert p.ui.statuses[-1] == ("busy", "Dokończ edycję tekstu…")


def test_arm_skip_rules_forces_the_next_question_to_chat_only():
    p = make_pipeline(pending=None)

    p.arm_skip_rules()
    assert p.skip_rules_pending is True

    p.ask_assistant("pytanie")

    assert p.assistant.calls == ["pytanie"]
    assert p.assistant.forced == [True]  # the next question went straight to chat
    assert p.skip_rules_pending is False  # one shot only

    p.ask_assistant("drugie pytanie")

    assert p.assistant.forced == [True, False]


def test_clear_skip_rules_leaves_the_next_question_normal():
    p = make_pipeline(pending=None)

    p.arm_skip_rules()
    p.clear_skip_rules()

    assert p.skip_rules_pending is False

    p.ask_assistant("pytanie")

    assert p.assistant.forced == [False]
