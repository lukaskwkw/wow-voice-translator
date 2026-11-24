"""Build the optional game database for WoW Forever (items, spells, recipes, trainers).

    python -m wvt.kb.gamedb --download

Sources (compiled, read from the clients, not from game files):

- EverlookDB, https://everlookdb.com - bulk JSON of every item and spell of the 1.60.1 client
  (https://everlookdb.com/downloads/items.json, .../spells.json). The project publishes the data
  without an explicit licence; this build uses it privately.
- CMaNGOS classic-db, https://github.com/cmangos/classic-db - the 1.12 world database (SQLite
  release) for the trainer lists, GPL-3.0.
- EverlookDB per-spell pages (``https://everlookdb.com/spell/<id>.json``) for the ``taught_by``
  lists, because ``npc_trainer`` has no weapon-skill rows; cached in ``spell_trainers.json`` so a
  rebuild only fetches new ids.

The result is ``data/kb/gamedb.json.gz`` (git-ignored, like all of ``data/``). Without that file
the app works exactly as before: :class:`wvt.kb.world.World` keeps its gamedb fields empty.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional

ITEMS_URL = "https://everlookdb.com/downloads/items.json"
SPELLS_URL = "https://everlookdb.com/downloads/spells.json"
CMANGOS_RELEASE = "https://api.github.com/repos/cmangos/classic-db/releases/latest"
EVERLOOK_SPELL_URL = "https://everlookdb.com/spell/{id}.json"
TRAINER_CACHE = "spell_trainers.json"
TRAINER_PAUSE = 0.15  # seconds between per-spell requests (polite)
TRAINER_SAVE_EVERY = 100

USER_AGENT = "wvt-gamedb/1 (+local build)"
CHUNK = 1 << 20

# API field -> compact field. Resistance stats end in "_res" in the API.
ITEM_FIELDS = {
    "ilvl": "ilvl", "reqlevel": "reqlevel", "quality": "quality", "category": "category",
    "type": "type", "slot": "slot", "armor": "armor", "block": "block", "block_value": "block_value",
    "mindmg": "mindmg", "maxdmg": "maxdmg", "speed": "speed", "str": "str", "agi": "agi", "sta": "sta",
    "int": "int", "spi": "spi", "fire": "fire_res", "nature": "nature_res", "frost": "frost_res",
    "shadow": "shadow_res", "arcane": "arcane_res", "spell_power": "spell_power", "healing": "healing",
    "crit": "crit", "hit": "hit", "defense": "defense", "dodge": "dodge", "parry": "parry",
    "attack_power": "attack_power", "ranged_attack_power": "ranged_attack_power", "mp5": "mp5",
    "hp5": "hp5", "spell_crit": "spell_crit", "spell_hit": "spell_hit", "spell_pen": "spell_pen",
    "buy": "buy", "sell": "sell", "stack": "stack", "binds": "binds", "source": "source",
    "source_ids": "source_ids", "profession": "profession", "skill": "skill", "set": "set",
    "effect": "effect", "use": "use",
}
SPELL_FIELDS = ("creates", "reagent_ids", "recipe", "skill", "skillrank", "cast_ms", "source")
QUALITY = {"poor": 0, "common": 1, "uncommon": 2, "rare": 3, "epic": 4, "legendary": 5}

# Polish labels the assistant reads from the optional gamedb (keys stay English in the file).
SLOT_PL = {"head": "głowa", "neck": "szyja", "shoulder": "ramiona", "chest": "tors", "waist": "talia",
           "legs": "nogi", "feet": "stopy", "wrist": "nadgarstki", "hands": "dłonie", "finger": "palec",
           "trinket": "trinket", "cloak": "płaszcz", "twohand": "dwuręczna", "mainhand": "główna ręka",
           "offhand": "off-hand", "holdable": "przedmiot trzymany", "shield": "tarcza", "thrown": "broń miotana",
           "ranged": "broń dystansowa", "none": ""}
STAT_PL = {"str": "Siła", "agi": "Zręczność", "sta": "Wytrzymałość", "int": "Intelekt", "spi": "Duch",
           "spell_power": "Moc zaklęć", "healing": "Leczenie", "crit": "Trafienie krytyczne",
           "spell_crit": "Kryt zaklęć", "hit": "Trafienie", "spell_hit": "Trafienie zaklęć", "defense": "Obrona",
           "dodge": "Unik", "parry": "Parowanie", "attack_power": "Siła ataku",
           "ranged_attack_power": "Siła ataku dystansowego", "mp5": "Mana na 5 s", "hp5": "Zdrowie na 5 s",
           "spell_pen": "Przenikanie zaklęć", "fire": "Odporność na ogień", "nature": "Odporność na naturę",
           "frost": "Odporność na mróz", "shadow": "Odporność na cień", "arcane": "Odporność na arcane"}
CATEGORY_PL = {"Armor": "Pancerz", "Weapon": "Broń", "Consumable": "Konsumowalne", "Trade Goods": "Materiały",
               "Recipe": "Przepis", "Quest": "Quest", "Container": "Torba", "Miscellaneous": "Różne",
               "Reagent": "Odczynnik", "Projectile": "Amunicja", "Quiver": "Kołczan", "Key": "Klucz"}
TYPE_PL = {"Cloth": "tkanina", "Leather": "skóra", "Mail": "kolczuga", "Plate": "płyta",
           "Miscellaneous": "różne", "Shields": "tarcza", "Daggers": "sztylet", "One-Handed Swords": "miecz",
           "One-Handed Maces": "obuch", "Staves": "kostur", "Two-Handed Maces": "dwuręczny obuch",
           "One-Handed Axes": "topór", "Two-Handed Axes": "dwuręczny topór", "Two-Handed Swords": "dwuręczny miecz",
           "Wands": "różdżka", "Bows": "łuk", "Guns": "strzelba", "Polearms": "włócznia", "Thrown": "broń miotana",
           "Fist Weapons": "broń pięści", "Crossbows": "kusza", "Fishing Pole": "wędka", "Cosmetic": "ozdoba",
           "Meat": "mięso", "Cooking": "gotowanie", "Alchemy": "alchemia", "Blacksmithing": "kowalstwo",
           "Tailoring": "krawiectwo", "Leatherworking": "skórnictwo", "Engineering": "inżynieria",
           "Enchanting": "zaklinanie", "Jewelcrafting": "jubilerstwo", "First Aid": "pierwsza pomoc"}


def _rows(data) -> list:
    """The record list of a bulk JSON file (``{"build": ..., "data": [...]}`` or a bare list)."""
    if isinstance(data, dict):
        for key in ("data", "rows"):
            if isinstance(data.get(key), list):
                return data[key]
        return []
    return data if isinstance(data, list) else []


def _kept(value) -> bool:
    """False for a value worth dropping from the compact file (empty or zero)."""
    if value is None or value == "" or value == [] or value == {}:
        return False
    return not (isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0)


def _quality(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        return QUALITY.get(value.strip().lower())
    return None


def _reagents(text) -> list[list[int]]:
    """``"3173:1, 3172:2"`` -> ``[[3173, 1], [3172, 2]]``."""
    out: list[list[int]] = []
    if not isinstance(text, str):
        return out
    for part in text.replace(" ", "").split(","):
        if ":" not in part:
            continue
        a, b = part.split(":", 1)
        try:
            out.append([int(a), int(b)])
        except ValueError:
            continue
    return out


def download(raw_dir: Path, want_trainers: bool = True) -> None:
    """Fetch the bulk item and spell JSON files, and the CMaNGOS SQLite world DB for trainers."""
    everlook = raw_dir / "everlook"
    everlook.mkdir(parents=True, exist_ok=True)
    for url, name in ((ITEMS_URL, "items.json"), (SPELLS_URL, "spells.json")):
        dest = everlook / name
        print(f"Pobieram {url} -> {dest}")
        _fetch(url, dest)
        print(f"  {dest.stat().st_size:,} bajtów".replace(",", " "))
    if not want_trainers:
        print("Trenerzy: pomijam (--no-trainers)")
        return
    cmangos = raw_dir / "cmangos"
    cmangos.mkdir(parents=True, exist_ok=True)
    print(f"Pobieram listę wydań {CMANGOS_RELEASE}")
    try:
        with urllib.request.urlopen(_request(CMANGOS_RELEASE), timeout=60) as resp:
            release = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise SystemExit(f"Błąd HTTP przy liście wydań CMaNGOS: {exc}")
    assets = [a for a in release.get("assets", []) if "sqlite" in str(a.get("name", "")).lower()]
    if not assets:
        raise SystemExit("Wydanie CMaNGOS nie ma pliku sqlite")
    asset = assets[0]
    zip_path = cmangos / asset["name"]
    print(f"Pobieram {asset['browser_download_url']} -> {zip_path}")
    _fetch(asset["browser_download_url"], zip_path)
    print(f"  {zip_path.stat().st_size:,} bajtów".replace(",", " "))
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".sqlite", ".db", ".sqlite3"))]
        if not members:
            raise SystemExit(f"Brak pliku .sqlite w {zip_path.name}")
        for member in members:
            print(f"Rozpakowuję {member} ({zf.getinfo(member).file_size:,} bajtów)".replace(",", " "))
            zf.extract(member, cmangos)


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})


def _fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(_request(url), timeout=120) as resp, open(tmp, "wb") as fh:
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                fh.write(block)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Błąd HTTP {exc.code} przy {url}: {exc.reason}")
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Błąd sieci przy {url}: {exc.reason}")
    tmp.replace(dest)


def _build_items(rows: list) -> dict[int, dict]:
    items: dict[int, dict] = {}
    for r in rows:
        iid = r.get("id")
        name = r.get("name")
        if not isinstance(iid, int) or isinstance(iid, bool) or not isinstance(name, str) or not name.strip():
            continue
        it: dict = {"id": iid, "name": name.strip()}
        for key, field in ITEM_FIELDS.items():
            value = r.get(field)
            if key == "quality":
                value = _quality(value)
            elif key == "binds" and value == "none":
                value = ""
            if _kept(value):
                it[key] = value
        items[iid] = it
    return items


def _build_spells(rows: list) -> dict[int, dict]:
    spells: dict[int, dict] = {}
    for r in rows:
        sid = r.get("id")
        name = r.get("name")
        if not isinstance(sid, int) or isinstance(sid, bool) or not isinstance(name, str) or not name.strip():
            continue
        sp: dict = {"id": sid, "name": name.strip()}
        for key in SPELL_FIELDS:
            value = r.get(key)
            if _kept(value):
                sp[key] = value
        spells[sid] = sp
    return spells


def _build_recipes(spell_rows: list, items: dict[int, dict]) -> dict[int, dict]:
    recipes: dict[int, dict] = {}
    for r in spell_rows:
        sid, creates = r.get("id"), r.get("creates")
        if not isinstance(sid, int) or isinstance(sid, bool):
            continue
        if not isinstance(creates, int) or isinstance(creates, bool) or not creates:
            continue
        reagents = _reagents(r.get("reagent_ids"))
        if not reagents:
            continue
        rec: dict = {
            "spell": sid,
            "creates": creates,
            "reagents": reagents,
            "recipe_item": int(r.get("recipe") or 0),
            "profession": r.get("skill") or "",
            "skill": int(r.get("skillrank") or 0),
        }
        product = items.get(creates)
        if product:
            rec["product"] = product["name"]
        recipes[sid] = rec
    return recipes


def _trainer_table(con: sqlite3.Connection) -> Optional[str]:
    names = {row[0] for row in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('npc_trainer', 'npc_trainer_template')")}
    for name in ("npc_trainer", "npc_trainer_template"):
        if name in names:
            return name
    return None


def _spell_template_names(con: sqlite3.Connection) -> dict[int, str]:
    """Spell names from the CMaNGOS ``spell_template`` table (most trainer spells are not in the
    Forever spell list, so the trainer answer needs this fallback)."""
    names: dict[int, str] = {}
    exists = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='spell_template'").fetchone()
    if not exists:
        return names
    columns = {row[1] for row in con.execute("PRAGMA table_info(spell_template)")}
    if "SpellName" not in columns or "Id" not in columns:
        return names
    for spell, name in con.execute("SELECT Id, SpellName FROM spell_template"):
        if isinstance(spell, int) and isinstance(name, str) and name.strip():
            names[spell] = name.strip()
    return names


def _build_trainers(trainers_db: Optional[Path]) -> tuple[dict[int, list], dict[int, str]]:
    trainers: dict[int, list] = {}
    names: dict[int, str] = {}
    if trainers_db is None:
        return trainers, names
    if not trainers_db.exists():
        print(f"Trenerzy: brak pliku {trainers_db}")
        return trainers, names
    con = sqlite3.connect(str(trainers_db))
    try:
        table = _trainer_table(con)
        if table is None:
            print(f"Trenerzy: brak tabeli npc_trainer w {trainers_db.name}")
            return trainers, names
        print(f"Trenerzy: tabela {table} z {trainers_db.name}")
        query = (f"SELECT entry, spell, spellcost, reqskill, reqskillvalue, reqlevel FROM {table}")
        for entry, spell, cost, reqskill, reqskillvalue, reqlevel in con.execute(query):
            if not isinstance(entry, int) or not isinstance(spell, int):
                continue
            trainers.setdefault(entry, []).append(
                [spell, int(cost or 0), int(reqskill or 0), int(reqskillvalue or 0), int(reqlevel or 0)])
        names = _spell_template_names(con)
        if names:
            print(f"  nazwy zaklęć z spell_template: {len(names)}")
    finally:
        con.close()
    return trainers, names


def _parse_taught_by(data) -> list[dict]:
    """``classic_db.taught_by[]`` -> ``[{"npc": ..., "cost": ..., "req": ...}]`` (only kept fields)."""
    classic = data.get("classic_db") if isinstance(data, dict) else None
    entries = classic.get("taught_by") if isinstance(classic, dict) else None
    out: list[dict] = []
    for entry in entries or []:
        npc = entry.get("npc") if isinstance(entry, dict) else None
        if not isinstance(npc, int) or isinstance(npc, bool):
            continue
        row = {"npc": npc}
        for key in ("cost", "req", "reqlevel"):
            value = entry.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[key] = int(value)
        out.append(row)
    return out


def fetch_everlook_trainers(spell_ids: Iterable[int], cache_path: Path) -> tuple[dict[int, list[dict]], int]:
    """Per-spell EverlookDB JSON: who teaches it. Cached; only missing ids are requested."""
    cache: dict[str, list[dict]] = {}
    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                cache = {str(k): v for k, v in loaded.items() if isinstance(v, list)}
        except (OSError, ValueError):
            cache = {}
    todo = sorted(int(i) for i in set(spell_ids) if str(int(i)) not in cache)
    failed = 0
    print(f"Trenerzy EverlookDB: {len(todo)} zaklęć do pobrania (w pamięci: {len(cache)})")
    for n, spell in enumerate(todo, 1):
        url = EVERLOOK_SPELL_URL.format(id=spell)
        try:
            with urllib.request.urlopen(_request(url), timeout=30) as resp:
                cache[str(spell)] = _parse_taught_by(json.loads(resp.read().decode("utf-8")))
        except urllib.error.HTTPError as exc:
            failed += 1
            if exc.code == 429:
                time.sleep(2.0)
        except (urllib.error.URLError, ValueError, OSError):
            failed += 1
        if n % 10 == 0 or n == len(todo):
            print(f"\r  EverlookDB trainers: {n}/{len(todo)} (błędy: {failed})", end="\n" if n == len(todo) else "")
        if n % TRAINER_SAVE_EVERY == 0:
            _save_trainer_cache(cache_path, cache)
        time.sleep(TRAINER_PAUSE)
    _save_trainer_cache(cache_path, cache)
    if failed:
        print(f"  EverlookDB: {failed} zapytań nie udało się (pomijam, cache zachowany)")
    return {int(k): v for k, v in cache.items()}, failed


def _save_trainer_cache(path: Path, cache: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)


def _merge_trainers(trainers: dict[int, list], everlook: dict[int, list[dict]]) -> int:
    """EverlookDB rows win for a spell id; returns how many (spell, npc) pairs it added."""
    added = 0
    for spell, entries in everlook.items():
        if not entries:
            continue
        for npc in list(trainers):
            kept = [row for row in trainers[npc] if row[0] != spell]
            if kept:
                trainers[npc] = kept
            else:
                del trainers[npc]
        for entry in entries:
            npc = entry["npc"]
            row = [spell, entry.get("cost", 0), 0, entry.get("req", 0), entry.get("reqlevel", 0)]
            rows = trainers.setdefault(npc, [])
            if row not in rows:
                rows.append(row)
                added += 1
    return added


def build(items_path: Path, spells_path: Path, trainers_db: Optional[Path], out_path: Path,
          everlook_trainers: bool = False, raw_dir: Optional[Path] = None) -> dict:
    """Write the compact gzip JSON and return the same structure."""
    with open(items_path, encoding="utf-8") as fh:
        item_rows = _rows(json.load(fh))
    with open(spells_path, encoding="utf-8") as fh:
        spell_rows = _rows(json.load(fh))

    items = _build_items(item_rows)
    spells = _build_spells(spell_rows)
    recipes = _build_recipes(spell_rows, items)
    trainers, cmangos_names = _build_trainers(trainers_db)
    if everlook_trainers:
        # EverlookDB knows the weapon masters, which the CMaNGOS npc_trainer table does not list
        spells_known = set(spells)
        source_trainer = {sid for sid, sp in spells.items() if sp.get("source") == "trainer"}
        cmangos_spells = {row[0] for rows in trainers.values() for row in rows}
        wanted = (source_trainer | cmangos_spells) & spells_known
        skipped = len((source_trainer | cmangos_spells) - spells_known)
        print(f"Trenerzy EverlookDB: {len(source_trainer)} zaklęć source=trainer, {len(cmangos_spells)} z CMaNGOS,"
              f" {len(wanted)} do sprawdzenia ({skipped} spoza EverlookDB pomijam)")
        cache_path = (raw_dir or spells_path.parent.parent) / "everlook" / TRAINER_CACHE
        cache, failed = fetch_everlook_trainers(wanted, cache_path)
        added = _merge_trainers(trainers, {sid: cache.get(sid, []) for sid in wanted})
        print(f"  EverlookDB: dodano {added} par (zaklęcie, NPC); trenerzy: {len(trainers)} NPC")
    spell_names = {sid: sp["name"] for sid, sp in spells.items()}
    for sid, name in cmangos_names.items():  # trainer spells absent from the Forever spell list
        spell_names.setdefault(sid, name)

    out = {
        "v": 1,
        "built": datetime.date.today().isoformat(),
        "items": {str(k): v for k, v in items.items()},
        "spells": {str(k): v for k, v in spells.items()},
        "recipes": {str(k): v for k, v in recipes.items()},
        "trainers": {str(k): v for k, v in trainers.items()},
        "spell_names": {str(k): v for k, v in spell_names.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(out_path)
    return out


def _trainers_from_raw(raw_dir: Path) -> Optional[Path]:
    """The biggest .sqlite/.sqlite3/.db under raw/cmangos, when a download left one there."""
    folder = raw_dir / "cmangos"
    if not folder.is_dir():
        return None
    files = [p for p in folder.iterdir() if p.suffix.lower() in (".sqlite", ".sqlite3", ".db")]
    return max(files, key=lambda p: p.stat().st_size) if files else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m wvt.kb.gamedb", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="pobierz źródła do --raw-dir")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/kb/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/kb/gamedb.json.gz"))
    ap.add_argument("--trainers", type=Path, default=None, help="ścieżka do pliku SQLite CMaNGOS")
    ap.add_argument("--no-trainers", action="store_true", help="bez list trenerów")
    ap.add_argument("--everlook-trainers", action=argparse.BooleanOptionalAction, default=None,
                    help="dopisz trenerów z EverlookDB (domyślnie przy --download); --no-everlook-trainers wyłącza")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.download:
        download(args.raw_dir, want_trainers=not args.no_trainers)

    items_path = args.raw_dir / "everlook" / "items.json"
    spells_path = args.raw_dir / "everlook" / "spells.json"
    missing = [p for p in (items_path, spells_path) if not p.exists()]
    if missing:
        print("Brak plików źródłowych: " + ", ".join(str(p) for p in missing) + " (uruchom z --download)")
        return 2

    trainers_db = None
    if not args.no_trainers:
        trainers_db = args.trainers or _trainers_from_raw(args.raw_dir)
        if trainers_db is None:
            print("Trenerzy: brak pliku SQLite (użyj --trainers albo --no-trainers)")

    out = build(items_path, spells_path, trainers_db, args.out,
                everlook_trainers=args.everlook_trainers if args.everlook_trainers is not None else args.download,
                raw_dir=args.raw_dir)
    size = args.out.stat().st_size
    print(f"Gotowe: {args.out}")
    print(f"  przedmioty: {len(out['items'])}")
    print(f"  zaklęcia:   {len(out['spells'])}")
    print(f"  przepisy:   {len(out['recipes'])}")
    print(f"  trenerzy:   {len(out['trainers'])} (NPC)")
    print(f"  rozmiar:    {size:,} bajtów".replace(",", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
