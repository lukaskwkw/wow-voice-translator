"""Coloured [Name] links in the history: matching, priority, fallbacks and Qt wiring."""

from types import SimpleNamespace

import pytest

from wvt.history_links import EntityLinker, entity_command


def make_world():
    return SimpleNamespace(
        quests={
            4740: {"n": "Wanted: Murkdeep!"},
            9999: {"n": "Westfall Brigade"},
        },
        all_items={6480: {"n": "Slick Deviate Leggings"}},
        npcs={
            234: {"n": "Gryan Stoutmantle"},
            235: {"n": "Nalpak"},
            999: {"n": "Wanted: Murkdeep!"},
        },
        dungeons={1581: {"n": "The Deadmines"}},
        names={40: "Westfall"},
    )


def make_linker(world=None, enabled=True):
    box = {"world": world if world is not None else make_world()}
    return EntityLinker(lambda: box["world"], enabled=lambda: enabled), box


def test_quest_name_becomes_a_questie_link():
    linker, _ = make_linker()
    out = linker("[Wanted: Murkdeep!]")
    assert '<a href="wvt:quest:4740"' in out
    assert "#ffd100" in out
    assert "[[Wanted" not in out


def test_item_name_uses_the_item_kind_and_green():
    linker, _ = make_linker()
    out = linker("Slick Deviate Leggings")
    assert '<a href="wvt:item:6480"' in out
    assert "#1eff00" in out


def test_npc_name_uses_orange():
    linker, _ = make_linker()
    out = linker("Gryan Stoutmantle")
    assert '<a href="wvt:npc:234"' in out
    assert "#ff9f43" in out


def test_dungeon_and_zone_are_coloured_spans_without_href():
    linker, _ = make_linker()
    dungeon = linker("The Deadmines")
    assert "<span" in dungeon and "#c792ea" in dungeon and "href" not in dungeon
    zone = linker("Westfall")
    assert "<span" in zone and "#40c4ff" in zone and "href" not in zone


def test_longest_name_wins_and_contains_another_name():
    linker, _ = make_linker()
    out = linker("Westfall Brigade")
    assert '<a href="wvt:quest:9999"' in out
    assert "wvt:place:40" not in out
    deadmines = linker("The Deadmines")
    assert deadmines.count("Deadmines") == 1


def test_already_bracketed_text_is_not_wrapped_twice():
    linker, _ = make_linker()
    assert "[[Wanted" not in linker("[Wanted: Murkdeep!]")
    assert linker("Wanted: Murkdeep!").count("[Wanted: Murkdeep!]") == 1


def test_id_hint_resolves_an_ambiguous_name():
    linker, _ = make_linker()
    quest = linker("Wanted: Murkdeep! (id 4740)")
    assert "wvt:quest:4740" in quest
    npc = linker("Wanted: Murkdeep! (id 999)")
    assert "wvt:npc:999" in npc


def test_unknown_names_stay_plain_and_html_is_escaped():
    linker, _ = make_linker()
    out = linker("Zzz Unknown <b>bold</b>")
    assert "<a " not in out
    assert "&lt;b&gt;bold&lt;/b&gt;" in out


def test_disabled_or_no_world_falls_back_to_the_old_output():
    world = make_world()
    disabled, _ = make_linker(world, enabled=False)
    assert "<a" not in disabled("[Wanted: Murkdeep!]")
    box = {"world": None}
    no_world = EntityLinker(lambda: box["world"])
    assert "<a" not in no_world("[Wanted: Murkdeep!]")


def test_sentence_punctuation_stays_outside_the_link():
    linker, _ = make_linker()
    comma = linker("u Nalpak, Wailing Caverns")
    assert "</a>, Wailing" in comma
    assert "Nalpak," not in comma
    period = linker("Spotkasz go w Westfall.")
    assert "</span>." in period


def test_exclamation_mark_stays_inside_the_brackets():
    linker, _ = make_linker()
    out = linker("Wanted: Murkdeep!")
    assert ">[Wanted: Murkdeep!]</a>" in out


def test_tab_bullet_keeps_offsets_and_still_matches():
    linker, _ = make_linker()
    out = linker("-\tpunkt Gryan Stoutmantle")
    assert out.startswith("•\tpunkt")
    assert '<a href="wvt:npc:234"' in out


def test_bullets_headings_and_bold_survive_the_renderer():
    linker, _ = make_linker()
    out = linker("# Naglowek\n- punkt **wazny**")
    assert "<b>Naglowek</b>" in out
    assert "• punkt <b>wazny</b>" in out


@pytest.mark.parametrize("kind", ["quest", "item", "npc"])
def test_entity_command_exact_line(kind):
    assert entity_command(kind, 1) == (
        f'/run QuestieLoader:ImportModule("QuestieSearchResults"):GetDetailFrame("{kind}",1)')


@pytest.mark.parametrize("kind", ["dungeon", "place", "unknown"])
def test_entity_command_is_none_for_non_questie_kinds(kind):
    assert entity_command(kind, 1) is None


def test_qt_browser_reports_the_anchor_click():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtCore = pytest.importorskip("PySide6.QtCore")

    linker, _ = make_linker()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = QtWidgets.QTextBrowser()
    view.setOpenLinks(False)
    clicked = []
    view.anchorClicked.connect(lambda url: clicked.append(url.toString()))
    view.setHtml(linker("[Wanted: Murkdeep!]"))
    view.anchorClicked.emit(QtCore.QUrl("wvt:quest:4740"))
    assert clicked == ["wvt:quest:4740"]
    view.deleteLater()
    app.processEvents()
