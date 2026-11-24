"""Optional game database: gamedb build (no network) and the World gamedb methods."""

import gzip
import json
import sqlite3

from wvt.kb import gamedb
from wvt.kb.world import World

ITEMS = {"build": "test", "data": [
    {"id": 3173, "name": "Bear Meat", "quality": "common", "category": "Trade Goods", "type": "Meat",
     "stack": 20, "buy": 10, "sell": 2, "source": "vendor", "source_ids": "npc:1", "effect": "", "speed": 0},
    {"id": 200, "name": "Smoked Bear Meat", "quality": "uncommon", "category": "Consumable",
     "stack": 10, "sell": 5},
    {"id": 6892, "name": "Recipe: Smoked Bear Meat", "quality": "common", "category": "Recipe", "buy": 100},
    {"name": "no id, skipped"},
]}

SPELLS = {"build": "test", "data": [
    {"id": 8607, "name": "Smoked Bear Meat", "creates": 200, "reagent_ids": "3173:1", "recipe": 6892,
     "skill": "Cooking", "skillrank": 40, "cast_ms": 3000, "source": "trainer"},
    {"id": 999, "name": "Learn Thrown", "creates": 0, "reagent_ids": "", "recipe": 0,
     "skill": "Thrown", "skillrank": 1, "cast_ms": 0, "source": "trainer"},
    {"id": 2567, "name": "Thrown", "creates": 0, "reagent_ids": "", "recipe": 0,
     "skill": "Thrown", "skillrank": 1, "cast_ms": 0, "source": "trainer"},
]}


def _fixtures(tmp_path):
    raw = tmp_path / "raw" / "everlook"
    raw.mkdir(parents=True)
    items_path = raw / "items.json"
    spells_path = raw / "spells.json"
    items_path.write_text(json.dumps(ITEMS), encoding="utf-8")
    spells_path.write_text(json.dumps(SPELLS), encoding="utf-8")
    db = tmp_path / "world.sqlite"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE npc_trainer (entry INTEGER, spell INTEGER, spellcost INTEGER, reqskill INTEGER,"
                " reqskillvalue INTEGER, reqlevel INTEGER)")
    con.execute("CREATE TABLE spell_template (Id INTEGER, SpellName TEXT)")
    con.executemany("INSERT INTO npc_trainer VALUES (?, ?, ?, ?, ?, ?)",
                    [(198, 2567, 100, 0, 0, 4), (198, 999, 10, 0, 0, 1), (198, 1142, 50, 0, 0, 6)])
    con.executemany("INSERT INTO spell_template VALUES (?, ?)",
                    [(1142, "Fireball"), (2567, "Wrong Name")])
    con.commit()
    con.close()
    return items_path, spells_path, db


def test_build_makes_items_spells_recipes_and_trainers(tmp_path):
    items_path, spells_path, db = _fixtures(tmp_path)
    out_path = tmp_path / "gamedb.json.gz"
    out = gamedb.build(items_path, spells_path, db, out_path)

    assert len(out["items"]) == 3 and len(out["spells"]) == 3 and len(out["recipes"]) == 1
    assert out["recipes"]["8607"] == {"spell": 8607, "creates": 200, "reagents": [[3173, 1]],
                                      "recipe_item": 6892, "profession": "Cooking", "skill": 40,
                                      "product": "Smoked Bear Meat"}
    assert [2567, 100, 0, 0, 4] in out["trainers"]["198"]
    assert out["spell_names"]["2567"] == "Thrown"
    assert out["spell_names"]["1142"] == "Fireball"  # from CMaNGOS spell_template, not in the spell list
    # quality strings become numbers; zeros and empty strings are dropped
    assert out["items"]["3173"]["quality"] == 1 and out["items"]["200"]["quality"] == 2
    assert "effect" not in out["items"]["3173"] and "speed" not in out["items"]["3173"]
    assert "creates" not in out["spells"]["999"] and out["spells"]["999"]["skill"] == "Thrown"
    # the file is a gzip JSON that reads back
    with gzip.open(out_path, "rt", encoding="utf-8") as fh:
        assert json.load(fh)["v"] == 1


def _world(tmp_path, with_gamedb=True):
    items_path, spells_path, db = _fixtures(tmp_path)
    world_path = tmp_path / "world.json.gz"
    with gzip.open(world_path, "wt", encoding="utf-8") as fh:
        json.dump({"quests": {}, "npcs": {}, "objects": {}, "items": {}, "zones": {}}, fh)
    if with_gamedb:
        gamedb.build(items_path, spells_path, db, tmp_path / "gamedb.json.gz")
    return World.load(world_path)


def test_world_reads_the_optional_gamedb(tmp_path):
    w = _world(tmp_path)
    assert w.item_extra(3173)["name"] == "Bear Meat"
    assert w.item_extra(424242) is None
    assert w.recipe_of(6892)["spell"] == 8607 and w.recipe_of(200) is None
    assert [r["spell"] for r in w.recipes_using(3173)] == [8607]
    assert w.recipes_using(6892) == []
    assert w.spell_by_name("Thrown")["id"] == 2567  # exact name wins over "Learn Thrown"
    assert w.spell_by_name("learn")["id"] == 999  # a unique word match
    assert w.spell_by_name("no such spell") is None
    assert w.trainers_of(2567) == [198]
    assert w.spells_of(198) == [
        {"id": 2567, "name": "Thrown", "cost": 100, "reqlevel": 4, "reqskill": 0},
        {"id": 999, "name": "Learn Thrown", "cost": 10, "reqlevel": 1, "reqskill": 0},
        {"id": 1142, "name": "Fireball", "cost": 50, "reqlevel": 6, "reqskill": 0},
    ]


def test_world_without_gamedb_is_unchanged(tmp_path):
    w = _world(tmp_path, with_gamedb=False)
    assert w.gamedb == {}
    assert w.item_extra(3173) is None
    assert w.recipe_of(6892) is None
    assert w.recipes_using(3173) == []
    assert w.spell_by_name("Thrown") is None
    assert w.trainers_of(2567) == []
    assert w.spells_of(198) == []


def test_everlook_trainers_merge_from_the_cache_and_win(tmp_path, monkeypatch):
    items_path, spells_path, db = _fixtures(tmp_path)
    raw = tmp_path / "raw" / "everlook"
    # every wanted spell is cached, so a network call would be a test failure
    (raw / "spell_trainers.json").write_text(json.dumps({
        "8607": [], "999": [],
        "2567": [{"npc": 11866, "cost": 1000, "req": 1}, {"npc": 2704, "cost": 1000}],
    }), encoding="utf-8")

    def no_network(*a, **k):
        raise AssertionError("network used in an offline test")

    monkeypatch.setattr(gamedb.urllib.request, "urlopen", no_network)
    out = gamedb.build(items_path, spells_path, db, tmp_path / "gamedb.json.gz",
                       everlook_trainers=True, raw_dir=tmp_path / "raw")

    # the CMaNGOS row for 2567 is replaced by the two EverlookDB weapon masters
    assert [2567, 100, 0, 0, 4] not in out["trainers"]["198"]
    assert [999, 10, 0, 0, 1] in out["trainers"]["198"]
    assert out["trainers"]["11866"] == [[2567, 1000, 0, 1, 0]]
    assert out["trainers"]["2704"] == [[2567, 1000, 0, 0, 0]]


def test_cli_no_trainers_writes_a_loadable_file(tmp_path):
    _fixtures(tmp_path)
    out_path = tmp_path / "out" / "gamedb.json.gz"
    code = gamedb.main(["--no-trainers", "--raw-dir", str(tmp_path / "raw"), "--out", str(out_path)])
    assert code == 0 and out_path.exists()
    with gzip.open(out_path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    assert len(data["items"]) == 3 and data["trainers"] == {}
