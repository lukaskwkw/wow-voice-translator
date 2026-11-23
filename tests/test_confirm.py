"""Tests for the confirm-before-insert path (no Qt)."""

from __future__ import annotations

import threading
from collections import deque

from wvt.chat_insert import sanitize_chat_text
from wvt.pipeline import HistoryItem, PushToTalkPipeline


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

    def __getitem__(self, key):  # the pipeline asks for cfg["audio"]
        return self.get(key) if "." in key else self.data[key]


class FakeUI:
    def __init__(self):
        self.statuses = []
        self.results = []

    def status(self, kind, title, lines=None, timeout=0.0):
        self.statuses.append((kind, title))

    def result(self, item):
        self.results.append(item)


class FakeInserter:
    def __init__(self):
        self.calls = []

    def insert(self, chat_line, wow_is_foreground):
        self.calls.append((chat_line, wow_is_foreground))
        return "inserted"


class FakeTranslator:
    def __init__(self):
        self.remembers = []

    def remember(self, body, polish, direction):
        self.remembers.append((body, polish, direction))


def make_pipeline():
    p = PushToTalkPipeline.__new__(PushToTalkPipeline)
    p.cfg = Cfg({"privacy": {"keep_history": True, "log_texts": False},
                 "translation": {"max_chat_chars": 255},
                 "channels": {"party": "/p"},
                 "hud": {"show_seconds": 6}})
    p.inserter = FakeInserter()
    p.translator = FakeTranslator()
    p.ui = FakeUI()
    p.wow_foreground = lambda: True
    p._busy = threading.Lock()
    p._pending_confirm = None
    p.history = deque(maxlen=10)
    p.last_timings = {}
    return p


def test_finish_confirm_inserts_edited_line():
    p = make_pipeline()
    item = HistoryItem("party", "cześć", "hi", "/p hi")
    p._pending_confirm = item

    status = p.finish_confirm(item, "/p hello")

    assert status == "inserted"
    assert p.inserter.calls == [("/p hello", True)]
    assert item.chat_line == "/p hello"
    assert item.status == "inserted"
    assert list(p.history) == [item]
    assert p.ui.results == [item]
    assert p._pending_confirm is None


def test_cancel_confirm_does_not_insert():
    p = make_pipeline()
    item = HistoryItem("party", "cześć", "hi", "/p hi")
    p._pending_confirm = item

    p.cancel_confirm(item)

    assert p.inserter.calls == []
    assert p._pending_confirm is None
    assert p.ui.statuses[-1] == ("info", "Anulowano")


def test_stale_item_ignored():
    p = make_pipeline()
    stale = HistoryItem("party", "cześć", "hi", "/p hi")
    fresh = HistoryItem("party", "cześć", "hi", "/p hi")
    p._pending_confirm = fresh

    assert p.finish_confirm(stale, "/p stale") is None
    assert p.inserter.calls == []
    assert p._pending_confirm is fresh
    assert p.ui.statuses[-1] == ("info", "Nieaktualny tekst – zignorowano")

    p.cancel_confirm(stale)
    assert p._pending_confirm is fresh


def test_edited_line_sanitized():
    p = make_pipeline()
    item = HistoryItem("party", "cześć", "hi", "/p hi")
    p._pending_confirm = item

    p.finish_confirm(item, "/p  hello\n|world")

    line = p.inserter.calls[0][0]
    assert "\n" not in line and "|" not in line
    assert line == "/p hello /world"

    long_item = HistoryItem("party", "cześć", "hi", "/p hi")
    p._pending_confirm = long_item
    p.finish_confirm(long_item, "a" * 400)

    long_line = p.inserter.calls[1][0]
    assert len(long_line.encode("utf-8")) <= 255
    assert long_line == sanitize_chat_text("a" * 400, 255)
    assert len(long_line) == 255


def test_finish_confirm_strips_the_channel_prefix_and_remembers_the_body():
    p = make_pipeline()
    item = HistoryItem("party", "cześć", "hi", "/p hi")
    p._pending_confirm = item

    p.finish_confirm(item, "/p hello there")

    assert item.chat_line == "/p hello there"
    assert item.english == "hello there"
    assert p.translator.remembers == [("hello there", "cześć", "en-pl")]


def test_finish_confirm_forgets_the_pending_item_when_insert_fails():
    p = make_pipeline()
    item = HistoryItem("party", "cześć", "hi", "/p hi")
    p._pending_confirm = item

    def boom(chat_line, wow_is_foreground):
        raise RuntimeError("no window")

    p.inserter.insert = boom

    assert p.finish_confirm(item, "/p hi") is None
    assert p.ui.statuses[-1] == ("error", "Błąd – nic nie zostało wysłane")
    assert not p._busy.locked()


def test_on_press_is_refused_while_confirm_is_pending():
    p = make_pipeline()
    p.models_ready = threading.Event()
    p.models_ready.set()
    p._recording_channel = None

    class Recorder:
        def start(self):
            raise AssertionError("must not record while the text is being edited")

    p.recorder = Recorder()
    p._pending_confirm = HistoryItem("party", "cześć", "hi", "/p hi")

    p.on_press("party")

    assert p.ui.statuses[-1] == ("busy", "Dokończ edycję tekstu…")
    assert p._pending_confirm is not None


def test_free_channel_copies_the_text_instead_of_inserting(monkeypatch):
    import types

    import numpy as np

    copied = []
    monkeypatch.setattr("wvt.pipeline.set_clipboard_text", copied.append)

    class Stt:
        prompt_ok = False

        def transcribe(self, clip):
            return "Zaproś Bolka"

    class Translator:
        def translate(self, text, direction, context=None):
            return types.SimpleNamespace(text="Invite Bolek")

        def remember(self, body, polish, direction):
            pass

    cfg = Cfg({"audio": {}, "channels": {}, "translation": {"max_chat_chars": 255},
               "privacy": {"keep_history": True}, "hud": {"show_seconds": 6}})
    ui, inserter = FakeUI(), FakeInserter()
    pipe = PushToTalkPipeline(cfg, None, Stt(), Translator(), inserter, ui, lambda: True)
    audio = (np.sin(2 * np.pi * 200 * np.arange(16000) / 16000) * 0.3).astype(np.float32)

    item = pipe.process_audio(audio, "free")

    assert item is not None and item.channel == "free" and item.status == "free"
    assert item.chat_line == "Invite Bolek"
    assert copied == ["Invite Bolek"]
    assert inserter.calls == []
    assert ui.statuses[-1] == ("ok", "Skopiowano – wklej w wybranym czacie (Ctrl+V)")


def test_finish_confirm_for_a_free_item_copies_plain_text(monkeypatch):
    copied = []
    monkeypatch.setattr("wvt.pipeline.set_clipboard_text", copied.append)
    p = make_pipeline()
    item = HistoryItem("free", "cześć", "hi", "hi")
    p._pending_confirm = item

    status = p.finish_confirm(item, "hello there")

    assert status == "free"
    assert copied == ["hello there"]
    assert p.inserter.calls == []
    assert item.chat_line == "hello there"
    assert item.status == "free"
    assert p.ui.statuses[-1] == ("ok", "Skopiowano – wklej w wybranym czacie (Ctrl+V)")


def test_finish_confirm_for_a_free_item_inserts_a_typed_command():
    p = make_pipeline()
    item = HistoryItem("free", "cześć", "hi", "hi")
    p._pending_confirm = item

    status = p.finish_confirm(item, "/g hello")

    assert status == "inserted"
    assert p.inserter.calls == [("/g hello", True)]
    assert item.chat_line == "/g hello"


def test_the_confirm_branch_shows_the_prompt_and_does_not_insert():
    import types

    import numpy as np

    class Stt:
        prompt_ok = False

        def transcribe(self, clip):
            return "Zaproś Bolka"

    class Translator:
        def __init__(self):
            self.remembers = []

        def translate(self, text, direction, context=None):
            return types.SimpleNamespace(text="Invite Bolek")

        def remember(self, body, polish, direction):
            self.remembers.append((body, polish, direction))

    cfg = Cfg({"audio": {}, "channels": {"party": "/p"}, "insert": {"confirm": True},
               "translation": {"max_chat_chars": 255}, "privacy": {"keep_history": True},
               "hud": {"show_seconds": 6}})
    stt, tr, ui = Stt(), Translator(), FakeUI()
    inserter = FakeInserter()
    shown = []
    pipe = PushToTalkPipeline(cfg, None, stt, tr, inserter, ui, lambda: True, confirm=shown.append)
    audio = (np.sin(2 * np.pi * 200 * np.arange(16000) / 16000) * 0.3).astype(np.float32)

    item = pipe.process_audio(audio, "party")

    assert item is not None and item.status == "pending" and item.chat_line == "/p Invite Bolek"
    assert inserter.calls == [] and shown == [item]
    assert pipe._pending_confirm is item
    assert ("busy", "Popraw tekst i wciśnij Enter…") in ui.statuses
    assert not pipe._busy.locked()
