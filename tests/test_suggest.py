"""Tests for the ``[`` bracket autocomplete core (no Qt)."""

from types import SimpleNamespace

from wvt.suggest import (
    Suggestion,
    apply_suggestion,
    bracket_context,
    build_suggestions,
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


def test_filter_prefix_beats_substring():
    items = [
        Suggestion("The Deadmines", "Dungeon"),
        Suggestion("Deadmines", "Dungeon"),
        Suggestion("Deadmines Elite", "Dungeon"),
    ]
    result = filter_suggestions(items, "dead")
    # prefix match (Deadmines, Deadmines Elite) ranks above the word-start match
    assert result[0].name == "Deadmines"
    assert [s.name for s in result][-1] == "The Deadmines"


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
