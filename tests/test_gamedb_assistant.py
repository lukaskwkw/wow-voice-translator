"""Iskra answers from the optional gamedb: recipe facts, the recipe intent and trainer spells."""

import gzip
import json
import sqlite3

from tests.test_assistant import make_assistant
from wvt.assistant import parse_intent
from wvt.kb import gamedb
from wvt.kb.world import World

ITEMS = {"build": "test", "data": [
    {"id": 3173, "name": "Bear Meat", "quality": "common", "category": "Trade Goods", "type": "Meat",
     "stack": 20, "buy": 10, "sell": 2},
    {"id": 200, "name": "Smoked Bear Meat", "quality": "uncommon", "category": "Consumable",
     "stack": 10, "sell": 5},
    {"id": 6892, "name": "Recipe: Smoked Bear Meat", "quality": "common", "category": "Recipe", "type": "Cooking",
     "buy": 250, "sell": 62},
    {"id": 15404, "name": "Breakwater Girdle", "quality": "uncommon", "category": "Armor", "type": "Mail",
     "slot": "waist", "armor": 91, "agi": 2, "sta": 3, "buy": 1685, "sell": 337},
]}

SPELLS = {"build": "test", "data": [
    {"id": 8607, "name": "Smoked Bear Meat", "creates": 200, "reagent_ids": "3173:1", "recipe": 6892,
     "skill": "Cooking", "skillrank": 40, "cast_ms": 3000, "source": "trainer"},
    {"id": 999, "name": "Learn Thrown", "creates": 0, "reagent_ids": "", "recipe": 0,
     "skill": "Thrown", "skillrank": 1, "cast_ms": 0, "source": "trainer"},
    {"id": 2567, "name": "Thrown", "creates": 0, "reagent_ids": "", "recipe": 0,
     "skill": "Thrown", "skillrank": 1, "cast_ms": 0, "source": "trainer"},
]}

WORLD_ITEMS = {
    "3173": {"n": "Bear Meat", "c": 7, "il": 5},
    "200": {"n": "Smoked Bear Meat", "c": 0, "il": 5},
    "6892": {"n": "Recipe: Smoked Bear Meat", "c": 9, "il": 13, "v": ["Vendor A"]},
    "15404": {"n": "Breakwater Girdle", "c": 4, "sc": 3, "il": 18},
}

WORLD = {
    "quests": {}, "objects": {},
    "npcs": {
        "198": {"n": "Trainer Bob", "sub": "Weapon Master", "z": 10, "f": "A", "fl": 16, "sp": {"10": [[50, 50]]}},
        "199": {"n": "Trainer Sue", "sub": "Weapon Master", "z": 20, "f": "H", "fl": 16, "sp": {"20": [[50, 50]]}},
    },
    "items": WORLD_ITEMS,
    "all_items": WORLD_ITEMS,
    "zones": {"names": {"10": "Westfall", "20": "Durotar"}, "map": {"10": 1436, "20": 1411}},
}


def _world(tmp_path, with_gamedb=True):
    raw = tmp_path / "raw" / "everlook"
    raw.mkdir(parents=True)
    items_path, spells_path = raw / "items.json", raw / "spells.json"
    items_path.write_text(json.dumps(ITEMS), encoding="utf-8")
    spells_path.write_text(json.dumps(SPELLS), encoding="utf-8")
    db = tmp_path / "world.sqlite"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE npc_trainer (entry INTEGER, spell INTEGER, spellcost INTEGER, reqskill INTEGER,"
                " reqskillvalue INTEGER, reqlevel INTEGER)")
    con.execute("CREATE TABLE spell_template (Id INTEGER, SpellName TEXT)")
    con.executemany("INSERT INTO npc_trainer VALUES (?, ?, ?, ?, ?, ?)",
                    [(198, 2567, 100, 0, 0, 4), (199, 2567, 100, 0, 0, 4), (198, 999, 10, 0, 0, 1)])
    con.execute("INSERT INTO spell_template VALUES (?, ?)", (1142, "Fireball"))
    con.commit()
    con.close()
    if with_gamedb:
        gamedb.build(items_path, spells_path, db, tmp_path / "gamedb.json.gz")
    world_path = tmp_path / "world.json.gz"
    with gzip.open(world_path, "wt", encoding="utf-8") as fh:
        json.dump(WORLD, fh)
    return World.load(world_path)


def _assistant(world):
    a, *_ = make_assistant()
    a.world_getter = lambda: world
    return a


def test_recipe_item_shows_profession_product_reagents_and_sellers(tmp_path):
    a = _assistant(_world(tmp_path))
    text = a._recipe("skąd przepis na Smoked Bear Meat")
    assert "Recipe: Smoked Bear Meat (Cooking 40)." in text
    assert "Uczy: Smoked Bear Meat." in text
    assert "Składniki: 1× Bear Meat." in text
    assert "Sprzedają: Vendor A." in text
    assert a._sources == ["recipe: Recipe: Smoked Bear Meat"]


def test_a_reagent_lists_the_recipes_that_use_it_with_their_source(tmp_path):
    a = _assistant(_world(tmp_path))
    text = a._recipe("co ugotuję z Bear Meat")
    assert text.startswith("Bear Meat (poz. 5): używany w przepisach:")
    assert "- Smoked Bear Meat (Cooking 40): przepis Recipe: Smoked Bear Meat; sprzedają: Vendor A." in text


def test_without_the_gamedb_the_recipe_answer_says_what_is_missing(tmp_path):
    a = _assistant(_world(tmp_path, with_gamedb=False))
    assert a._recipe("co ugotuję z Bear Meat") == \
        "Brak bazy gry – zbuduj ją: python -m wvt.kb.gamedb --download."
    text = a._recipe("skąd przepis na Smoked Bear Meat")
    assert "Recipe: Smoked Bear Meat" in text
    assert "Nie mam danych, czego ten przepis uczy." in text
    assert "Sprzedają: Vendor A." in text  # still from QuestieDB


def test_a_recipe_question_without_a_known_item_asks_for_the_name(tmp_path):
    a = _assistant(_world(tmp_path))
    assert a._recipe("skąd przepis na zupę z kamieni") == \
        "Podaj nazwę przedmiotu albo przepisu, np. „skąd przepis na Smoked Bear Meat?”."


def test_item_facts_add_slot_armor_stats_and_vendor_prices(tmp_path):
    a = _assistant(_world(tmp_path))
    facts = a._item_facts("Breakwater Girdle")
    assert "Rodzaj: Pancerz (kolczuga), talia." in facts
    assert "Pancerz: 91." in facts
    assert "Statystyki: +3 Wytrzymałość, +2 Zręczność." in facts
    assert "Sprzedasz za: 3 s 37 c; kupisz za: 16 s 85 c." in facts
    assert "Brak danych o recepturach i reagentach." not in facts
    assert "Brak danych o cenach rynkowych." in facts


def test_item_facts_show_the_recipe_and_the_recipes_an_item_is_used_in(tmp_path):
    a = _assistant(_world(tmp_path))
    recipe = a._item_facts("Recipe: Smoked Bear Meat")
    assert "Przepis: Smoked Bear Meat, Cooking 40. Składniki: 1× Bear Meat." in recipe
    reagent = a._item_facts("Bear Meat")
    assert "Używany w przepisach: Smoked Bear Meat." in reagent


def test_trainer_spells_name_the_weapon_masters(tmp_path):
    a = _assistant(_world(tmp_path))
    text, line = a._spell_trainer("gdzie nauczę się rzucać")
    assert text.startswith("Thrown: uczy")
    assert "Trainer Bob" in text and "Trainer Sue" in text
    assert line == ""
    assert a._sources == ["trainers: 2567"]


def test_without_the_gamedb_the_trainer_answer_keeps_the_old_path_and_says_why(tmp_path):
    a = _assistant(_world(tmp_path, with_gamedb=False))
    text, _line = a._spell_trainer("gdzie nauczę się rzucać")
    assert "Brak bazy gry – nie mam listy umiejętności trenerów." in text


def test_an_unknown_skill_gets_a_hint(tmp_path):
    a = _assistant(_world(tmp_path))
    text, _line = a._spell_trainer("gdzie nauczę się żonglować")
    assert text.startswith("Nie znam umiejętności „")
    assert "Spróbuj np. rzucanie, łuki, tarcze." in text


def test_recipe_and_trainer_questions_parse_to_their_own_intents():
    assert parse_intent("skąd przepis na Smoked Bear Meat").kind == "recipe"
    assert parse_intent("gdzie nauczę się gotować").kind == "find"  # a profession trainer, not a spell
    assert parse_intent("gdzie nauczę się rzucać").kind == "spell_trainer"
