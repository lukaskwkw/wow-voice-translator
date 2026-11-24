"""NPC loot ("co dropi z Hoggera?") from QuestieDB drop names, with no drop chance."""

from types import SimpleNamespace

from tests.test_assistant import make_assistant
from wvt.assistant import parse_intent
from wvt.kb.world import World

WORLD_DATA = {
    "quests": {},
    "objects": {},
    "items": {},
    "npcs": {
        448: {"n": "Hogger", "r": 1, "lv": [10, 11], "z": 10},
        449: {"n": "Handlarz", "fl": 4},
        450: {"n": "Rzadki Boss", "r": 4, "z": 20},
    },
    "all_items": {
        100: {"n": "Huge Gnoll Claw", "il": 12, "rl": 8, "nd": ["Hogger"]},
        101: {"n": "Torn Cloth", "il": 5, "nd": ["Hogger"]},
    },
    "zones": {"names": {"10": "Elwynn Forest", "20": "Westfall"}},
}


def loot_assistant():
    a, *_ = make_assistant()
    a.world_getter = lambda: World(WORLD_DATA)
    a.state = SimpleNamespace(level=10, zone="Elwynn Forest", subzone="", members=[])
    a.quests.player = SimpleNamespace(race_id=1, class_id=1, faction_code="A")
    return a


def test_npc_loot_indexes_items_by_dropping_npc():
    world = World(WORLD_DATA)
    assert world.npc_loot(448) == [100, 101]
    assert world.npc_loot(999) == []


def test_co_dropi_z_nazwanego_npc_daje_naglowek_przedmioty_i_stopke():
    a = loot_assistant()
    reply, line = a.handle("co dropi z Hoggera?")
    assert line == ""
    assert reply.startswith("Hogger: elita, poz. 10-11, Elwynn Forest")
    assert "- Huge Gnoll Claw, poz. 12 (wym. 8)" in reply
    assert "- Torn Cloth, poz. 5" in reply  # no required level: no "(wym. ...)"
    assert "Ranga z QuestieDB; szansa wypadu nieznana." in reply
    assert a._sources == ["npc loot: 2"]


def test_nieznany_npc_ma_komunikat_o_braku_nazwy():
    a = loot_assistant()
    reply, _ = a._npc_loot("co dropi z Nieznanego?")
    assert reply.startswith("Nie znam NPC o nazwie")


def test_vendor_hint_i_puste_lupy():
    a = loot_assistant()
    reply, _ = a._npc_loot("co dropi z Handlarza?")
    assert "QuestieDB nie ma tabeli łupów dla Handlarz." in reply
    assert "Sprzedaje też towary" in reply


def test_bez_bazy_swiata_zwykly_komunikat():
    a = loot_assistant()
    a.world_getter = lambda: None
    reply, _ = a._npc_loot("co dropi z Hoggera?")
    assert reply.startswith("Nie mam jeszcze bazy questów")


def test_co_mi_wypadlo_nadal_idzie_do_starego_lootu():
    assert parse_intent("co mi wypadło?").kind == "loot"
    assert parse_intent("co dropi z Hoggera?").kind == "npc_loot"
    assert parse_intent("co wypada z bossa?").kind == "npc_loot"
