"""Tests for the ``[`` bracket autocomplete core (no Qt)."""

from types import SimpleNamespace

import pytest

from wvt.suggest import (
    Suggestion,
    apply_delimited,
    apply_suggestion,
    bracket_context,
    build_suggestions,
    delimiter_context,
    filter_suggestions,
    fold,
)


# --------------------------------------------------------------------------- bracket_context

def test_bracket_context_open():
    assert bracket_context("[de", 3) == (0, "de")


def test_bracket_context_multiple():
    text = "a [b [de"
    assert bracket_context(text, len(text)) == (5, "de")


def test_bracket_context_closed_is_none():
    assert bracket_context("[de]", 4) is None


def test_bracket_context_cursor_before_bracket():
    assert bracket_context("[de", 0) is None


def test_bracket_context_cursor_after_closed_bracket():
    text = "[de] x"
    assert bracket_context(text, 4) is None
    assert bracket_context(text, 6) is None


# --------------------------------------------------------------------------- delimiter_context

def test_delimiter_context_brace_open():
    assert delimiter_context("{quest", 6, "{", "}") == (0, "quest")


def test_delimiter_context_brace_closed_is_none():
    assert delimiter_context("{quest}", 7, "{", "}") is None


def test_delimiter_context_nearest_opener_with_both():
    text = "[foo {quest"
    square = delimiter_context(text, len(text), "[", "]")
    brace = delimiter_context(text, len(text), "{", "}")
    assert square == (0, "foo {quest")
    assert brace == (5, "quest")
    assert brace[0] > square[0]


# --------------------------------------------------------------------------- apply_suggestion

def test_apply_suggestion_basic():
    # cursor sits at the end of "pytanie [Dea" (length 12)
    text, cursor = apply_suggestion("pytanie [Dea", 12, "Deadmines")
    assert text == "pytanie [Deadmines]"
    assert cursor == len(text)


def test_apply_suggestion_existing_closing_bracket_not_duplicated():
    text, cursor = apply_suggestion("pytanie [Dea]", 12, "Deadmines")
    assert text == "pytanie [Deadmines]"
    assert cursor == len(text)


def test_apply_suggestion_keeps_text_after_bracket():
    text, cursor = apply_suggestion("pytanie [Dea] dalej", 12, "Deadmines")
    assert text == "pytanie [Deadmines] dalej"
    assert cursor == len("pytanie [Deadmines]")


def test_apply_suggestion_no_context_returns_unchanged():
    assert apply_suggestion("plain text", 5, "Deadmines") == ("plain text", 5)


def test_apply_delimited_inserts_brace():
    text, cursor = apply_delimited("pytanie {quest", 14, "quest_list", "{", "}")
    assert text == "pytanie {quest_list}"
    assert cursor == len(text)


def test_apply_delimited_consumes_existing_closer():
    text, cursor = apply_delimited("pytanie {quest}", 14, "quest_list", "{", "}")
    assert text == "pytanie {quest_list}"
    assert cursor == len(text)


# --------------------------------------------------------------------------- fold / filter

def test_fold_diacritics_and_case():
    assert fold("Żółć Górą!") == "zolc gora"
    assert fold("ZUL'GURUB") == "zul gurub"


def _suggestions():
    return [
        Suggestion("The Deadmines", "Dungeon"),
        Suggestion("Deadmines", "Dungeon"),
        Suggestion("Deadmine Pass", "Miejsce"),
        Suggestion("Żółć", "Przedmiot"),
        Suggestion("Westfall", "Miejsce"),
    ]


def test_filter_prefix_and_word_start_tie():
    items = [
        Suggestion("The Deadmines", "Dungeon"),
        Suggestion("Deadmines", "Dungeon"),
        Suggestion("Deadmines Elite", "Dungeon"),
    ]
    result = filter_suggestions(items, "dead")
    # prefix matches come first; equal tier and group are decided by name length
    assert [s.name for s in result] == ["Deadmines", "The Deadmines", "Deadmines Elite"]


def test_filter_word_start_matches_multiword_names():
    items = [
        Suggestion("Stormwind", "Miejsce"),
        Suggestion("Messenger to Stormwind", "Dziennik"),
    ]
    result = filter_suggestions(items, "storm")
    assert [s.name for s in result] == ["Messenger to Stormwind", "Stormwind"]


def test_filter_fuzzy_typo():
    items = [Suggestion("Deadmines", "Dungeon"), Suggestion("Westfall", "Miejsce")]
    assert filter_suggestions(items, "deadmnes")[0].name == "Deadmines"


def test_filter_fuzzy_transposition():
    items = [Suggestion("Stormwind", "Miejsce"), Suggestion("Westfall", "Miejsce")]
    assert filter_suggestions(items, "strom")[0].name == "Stormwind"


def test_filter_no_fuzzy_noise_when_enough_exact():
    items = [
        Suggestion("Deadmines", "Dungeon"),
        Suggestion("Deadmine Pass", "Miejsce"),
        Suggestion("Deadmines Elite", "Dungeon"),
        Suggestion("Deadmines Dungeon", "Dungeon"),
        Suggestion("Debris", "Przedmiot"),
    ]
    result = filter_suggestions(items, "dea", limit=2)
    assert len(result) == 2
    assert all(s.name.lower().startswith("dea") for s in result)


def test_filter_case_and_diacritic_insensitive():
    result = filter_suggestions(_suggestions(), "zol")
    assert [s.name for s in result] == ["Żółć"]
    assert filter_suggestions(_suggestions(), "ŻÓŁ")[0].name == "Żółć"


def test_filter_group_order_tie():
    items = [
        Suggestion("Deadmines", "Dungeon"),
        Suggestion("Deadmines", "Miejsce"),
    ]
    result = filter_suggestions(items, "dead")
    assert len(result) == 1
    # same folded key, so dedupe keeps the one the sort put first: Miejsce (5) before Dungeon (6)
    assert result[0].group == "Miejsce"


def test_filter_limit():
    result = filter_suggestions(_suggestions(), "", limit=2)
    assert len(result) == 2


def test_filter_empty_query_keeps_order():
    items = _suggestions()
    assert filter_suggestions(items, "   ") == items


# --------------------------------------------------------------------------- build_suggestions

class FakeQuest:
    def __init__(self, title, level=0, zone="", complete=False, failed=False, quest_id=0):
        self.title = title
        self.level = level
        self.zone = zone
        self.complete = complete
        self.failed = failed
        self.quest_id = quest_id


class FakeBook:
    def __init__(self, quests):
        self.known = True
        self._quests = quests

    def quests(self):
        return list(self._quests)


class FakeWorld:
    def __init__(self):
        self.quests = {
            1: {"n": "Deadmines", "lvl": 18, "zone": 10},
            2: {"n": "The Defias Brotherhood", "lvl": 14, "zone": 10},
            3: {"n": "High Level Quest", "lvl": 60},
            4: {"n": "Unavailable Quest", "lvl": 15, "avail": False},
        }
        self.npcs = {5: {"n": "Gryan Stoutmantle"}}
        self.names = {10: "Westfall"}
        self.dungeons = {6: {"n": "Deadmines Dungeon"}}
        self.items = {7: {"n": "Defias Bandana"}}

    def available(self, q, qid, player, active):
        return q.get("avail", True)


def _build(world=...):
    book = FakeBook([
        FakeQuest("Deadmines", 18, "Westfall", complete=True, quest_id=1),
        FakeQuest("Westfall Stew", 12, "Westfall", quest_id=8),
        FakeQuest("Failed Quest", failed=True, quest_id=9),
    ])
    player = SimpleNamespace(level=15, done_known=True)
    state = SimpleNamespace(members=["Arthas", "Jaina"])
    w = FakeWorld() if world is ... else world
    return build_suggestions(book, w, player, state)


def test_build_journal_groups_and_detail():
    out = _build()
    journal = {s.name: s for s in out}
    assert journal["Deadmines"].group == "Do oddania"
    assert "18" in journal["Deadmines"].detail
    assert "Westfall" in journal["Deadmines"].detail
    assert journal["Westfall Stew"].group == "Dziennik"
    assert "Failed Quest" not in journal


def test_build_available_level_filter_and_gating():
    out = _build()
    available = [s for s in out if s.group == "Do wzięcia"]
    assert [s.name for s in available] == ["The Defias Brotherhood"]
    assert "14" in available[0].detail
    assert "Westfall" in available[0].detail
    # quest 4 is in the level range but available() says no: only reachable as a bare Quest name
    groups = {s.name: s.group for s in out}
    assert groups["Unavailable Quest"] == "Quest"
    assert groups["High Level Quest"] == "Quest"


def test_build_dedupe_journal_wins_over_quest_db():
    out = _build()
    deadmines = [s for s in out if s.name == "Deadmines"]
    assert len(deadmines) == 1
    assert deadmines[0].group == "Do oddania"


def test_build_other_sources_present():
    groups = {s.name: s.group for s in _build()}
    assert groups["Gryan Stoutmantle"] == "NPC"
    assert groups["Westfall"] == "Miejsce"
    assert groups["Deadmines Dungeon"] == "Dungeon"
    assert groups["Defias Bandana"] == "Przedmiot"
    assert groups["Arthas"] == "Grupa"
    assert groups["Jaina"] == "Grupa"


def test_build_keys_are_folded():
    for s in _build():
        assert s.key == fold(s.name)


def test_build_world_none_does_not_raise():
    out = _build(world=None)
    names = {s.name: s.group for s in out}
    assert names["Deadmines"] == "Do oddania"
    assert names["Westfall Stew"] == "Dziennik"


# --------------------------------------------------------------------------- build ranking

class RankWorld:
    def __init__(self):
        self.quests = {}
        self.npcs = {}
        self.names = {10: "Westfall", 20: "Redridge Mountains"}
        self.dungeons = {}
        self.items = {}

    def find_area(self, name):
        return {"westfall": 10, "redridge mountains": 20}.get(fold(name or ""), 0)

    def map_of(self, area):
        return 10 if area == 10 else 20

    def zone_of_map(self, ui_map):
        return 10 if ui_map == 10 else 0

    def available(self, q, qid, player, active):
        return True

    def starter_spot(self, qid, pos=None, faction=""):
        return None


def _rank_book(quests, distances=None, zone="Westfall"):
    distances = distances or {}

    class Book:
        known = True
        pos = SimpleNamespace(map_id=10)

        def quests(self):
            return list(quests)

        def where(self, q):
            if q.title not in distances:
                return None
            return SimpleNamespace(distance_yd=distances[q.title], other_zone="")

    return Book()


def test_build_journal_near_quest_first():
    quests = [
        FakeQuest("Far Quest", 10, "Westfall", quest_id=1),
        FakeQuest("Near Quest", 10, "Westfall", quest_id=2),
    ]
    book = _rank_book(quests, {"Far Quest": 800, "Near Quest": 50})
    state = SimpleNamespace(members=[], zone="Westfall")
    out = build_suggestions(book, RankWorld(), SimpleNamespace(level=15, done_known=False), state)
    journal = [s.name for s in out if s.group == "Dziennik"]
    assert journal == ["Near Quest", "Far Quest"]
    near = next(s for s in out if s.name == "Near Quest")
    assert "~50 yd" in near.detail


def test_build_available_same_zone_first():
    world = RankWorld()
    world.quests = {
        1: {"n": "Other Zone Quest", "lvl": 15, "zone": 20},
        2: {"n": "Here Quest", "lvl": 15, "zone": 10},
    }
    book = _rank_book([])
    state = SimpleNamespace(members=[], zone="Westfall")
    out = build_suggestions(book, world, SimpleNamespace(level=15, done_known=True, faction="A"), state)
    available = [s.name for s in out if s.group == "Do wzięcia"]
    assert available == ["Here Quest", "Other Zone Quest"]
    assert next(s for s in out if s.name == "Here Quest").rank == 0
    assert next(s for s in out if s.name == "Other Zone Quest").rank == 2_000_000


def test_build_bag_group_rank_and_dedupe():
    world = RankWorld()
    world.items = {7: {"n": "Defias Bandana"}}
    bag_item = SimpleNamespace(name="Defias Bandana", count=3, quality=4)
    bags = SimpleNamespace(known=True, items=[bag_item])
    book = _rank_book([])
    out = build_suggestions(book, world, SimpleNamespace(level=15, done_known=False),
                            SimpleNamespace(members=[]), bags)
    hits = [s for s in out if s.name == "Defias Bandana"]
    assert len(hits) == 1
    assert hits[0].group == "Plecak"
    assert hits[0].detail == "3x, jakość 4"
    assert hits[0].rank == -4.0


# --------------------------------------------------------------------------- Qt popup keys

@pytest.mark.parametrize("key_name", ["Tab", "Return"])
@pytest.mark.parametrize("on_popup", [False, True])
def test_completer_confirm_from_edit_and_popup(key_name, on_popup):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtTest = pytest.importorskip("PySide6.QtTest")

    from wvt.completer import BracketCompleter

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(win)
    edit = QtWidgets.QLineEdit()
    other = QtWidgets.QLineEdit()
    layout.addWidget(edit)
    layout.addWidget(other)
    win.show()
    app.processEvents()

    completer = BracketCompleter(edit, lambda: [Suggestion("Deadmines", "Dungeon")])
    edit.setFocus()
    QtTest.QTest.keyClicks(edit, "[Dea")
    app.processEvents()
    popup = completer._completer.popup()
    assert popup is not None and popup.isVisible(), "popup did not open"
    target = popup if on_popup else edit
    QtTest.QTest.keyClick(target, getattr(QtCore.Qt.Key, "Key_" + key_name))
    app.processEvents()

    text = edit.text()
    other_focus = other.hasFocus()
    win.close()
    app.processEvents()
    assert text == "[Deadmines]", f"text={text!r}, on_popup={on_popup}"
    assert not other_focus, f"second edit gained focus (on_popup={on_popup})"


def test_completer_brace_popup_and_tab_insert():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtTest = pytest.importorskip("PySide6.QtTest")

    from wvt.completer import BracketCompleter

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(win)
    edit = QtWidgets.QLineEdit()
    layout.addWidget(edit)
    win.show()
    app.processEvents()

    completer = BracketCompleter(
        edit, lambda: [],
        brace_getter=lambda: [Suggestion("quest_list", "Reguła"),
                              Suggestion("quest_turnin", "Reguła")])
    edit.setFocus()
    QtTest.QTest.keyClicks(edit, "{quest")
    app.processEvents()
    popup = completer._completer.popup()
    assert popup is not None and popup.isVisible(), "brace popup did not open"
    assert completer._model.rowCount() > 0, "brace popup has no rows"
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key.Key_Tab)
    app.processEvents()

    text = edit.text()
    win.close()
    app.processEvents()
    assert text == "{quest_list}", f"text={text!r}"


def test_assistant_confirm_dialog_completes_brackets_and_keeps_esc_for_cancel():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtTest = pytest.importorskip("PySide6.QtTest")

    from wvt.assistant_confirm import AssistantConfirmDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sent, cancelled = [], []
    dlg = AssistantConfirmDialog("co teraz", sent.append, lambda: cancelled.append(True),
                                 suggestions=lambda: [Suggestion("Deadmines", "Dungeon")],
                                 directives=lambda: [])
    dlg.show()
    app.processEvents()
    dlg.edit.setFocus()

    QtTest.QTest.keyClicks(dlg.edit, "[Dea")
    app.processEvents()
    completer = dlg._completer
    assert completer is not None and completer.popup_visible(), "popup did not open"
    QtTest.QTest.keyClick(dlg.edit, QtCore.Qt.Key.Key_Return)
    app.processEvents()
    assert dlg.edit.text() == "[Deadmines]"
    assert sent == [] and cancelled == [], "the completing Enter must not send"

    # the completion closed the popup: open it again to check that Esc only hides the list
    dlg.edit.selectAll()
    QtTest.QTest.keyClicks(dlg.edit, "[Dea")
    app.processEvents()
    assert completer.popup_visible(), "popup did not open again"
    QtTest.QTest.keyClick(dlg.edit, QtCore.Qt.Key.Key_Escape)
    app.processEvents()
    assert not completer.popup_visible(), "first Esc closes only the popup"
    assert sent == [] and cancelled == [], "first Esc must not cancel the dialog"

    QtTest.QTest.keyClick(dlg.edit, QtCore.Qt.Key.Key_Return)
    app.processEvents()
    assert sent == ["[Dea"], "Enter with the list closed sends"

    dlg2 = AssistantConfirmDialog("co teraz", sent.append, lambda: cancelled.append(True),
                                  suggestions=lambda: [Suggestion("Deadmines", "Dungeon")],
                                  directives=lambda: [])
    dlg2.show()
    app.processEvents()
    dlg2.edit.setFocus()
    QtTest.QTest.keyClick(dlg2.edit, QtCore.Qt.Key.Key_Escape)
    app.processEvents()
    assert cancelled == [True]
