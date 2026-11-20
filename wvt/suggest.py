"""Data and matching core for ``[`` bracket autocomplete in the Iskra inputs.

No Qt and no imports from other ``wvt`` modules. Everything works by duck typing on the
objects the app already has (quest book, world data, player, game state) and never raises
when that data is missing or only partly loaded.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class Suggestion:
    name: str
    group: str = ""
    detail: str = ""
    key: str = ""  # folded name, filled by build_suggestions
    rank: float = 0.0  # lower = earlier, inside a group


GROUP_ORDER = {
    "Do oddania": 0,
    "Dziennik": 1,
    "Do wzięcia": 2,
    "Plecak": 3,
    "Quest": 4,
    "NPC": 5,
    "Miejsce": 6,
    "Dungeon": 7,
    "Przedmiot": 8,
    "Grupa": 9,
}

_NON_WORD = re.compile(r"[^\w\s+]")


def fold(text: str) -> str:
    """Lower case, Polish ``ł`` -> ``l``, no diacritics, no punctuation, single spaces.

    Same normalization the assistant uses for matching. ASCII input (the vast majority of
    the ~20k names) skips the NFKD/combining-mark pass.
    """
    t = text or ""
    if t.isascii():
        t = t.lower()
    else:
        t = t.lower().replace("ł", "l")
        t = unicodedata.normalize("NFKD", t)
        t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = _NON_WORD.sub(" ", t)
    return " ".join(t.split())


def delimiter_context(text: str, cursor: int, opener: str = "[", closer: str = "]") -> Optional[tuple[int, str]]:
    """``(index_of_opener, text_after_it)`` for an unclosed ``opener`` before ``cursor``.

    A ``closer`` met first means the pair is already closed; a cursor outside the text is
    ignored.
    """
    if cursor < 1 or cursor > len(text):
        return None
    i = cursor - 1
    while i >= 0:
        ch = text[i]
        if ch == closer:
            return None
        if ch == opener:
            return i, text[i + 1:cursor]
        i -= 1
    return None


def bracket_context(text: str, cursor: int) -> Optional[tuple[int, str]]:
    """``[`` context; delegates to :func:`delimiter_context`."""
    return delimiter_context(text, cursor, "[", "]")


def _clean(name: str) -> str:
    return " ".join((name or "").split())


def _area_id(world, value) -> int:
    """AreaTable id for an int-ish value or a zone name; 0 when unknown."""
    if value is None or value == "":
        return 0
    try:
        if isinstance(value, str):
            return world.find_area(value) if world is not None else 0
        return int(value)
    except Exception:
        return 0


def build_suggestions(book, world, player, state=None, bags=None) -> list[Suggestion]:
    """Every name worth suggesting, in priority order and deduplicated by folded name."""
    items: list[Suggestion] = []
    seen: set[str] = set()

    def add(name: str, group: str, detail: str = "", rank: float = 0.0) -> None:
        nice = _clean(name)
        if not nice:
            return
        key = fold(nice)
        if not key or key in seen:
            return
        seen.add(key)
        items.append(Suggestion(nice, group, detail, key, rank))

    # where the player is, as a cheap reference for ranking
    pos = getattr(book, "pos", None)
    zone_name = ""
    try:
        zone_name = getattr(state, "zone", "") or getattr(state, "subzone", "") or ""
    except Exception:
        zone_name = ""
    current_area = 0
    try:
        if world is not None and zone_name:
            current_area = world.find_area(zone_name)
    except Exception:
        current_area = 0
    current_map = getattr(pos, "map_id", None) if pos is not None else None
    if not current_area and current_map:
        try:
            current_area = world.zone_of_map(current_map) if world is not None else 0
        except Exception:
            current_area = 0

    def area_rank(area) -> int:
        try:
            area = int(area or 0)
        except Exception:
            return 2_000_000
        if area == current_area:
            return 0
        if area and current_map:
            try:
                if world is not None and world.map_of(area) == current_map:
                    return 1_000_000
            except Exception:
                pass
        return 2_000_000

    # 1. the player's journal: turn-ins first, then the rest
    try:
        if getattr(book, "known", False):
            for q in book.quests():
                try:
                    if getattr(q, "failed", False):
                        continue
                    bits = []
                    lvl = getattr(q, "level", 0)
                    if lvl:
                        bits.append(f"poziom {lvl}")
                    zone = getattr(q, "zone", "")
                    if zone:
                        bits.append(zone)
                    area = 0
                    try:
                        if zone and world is not None:
                            area = world.find_area(zone)
                    except Exception:
                        area = 0
                    where = None
                    try:
                        where_fn = getattr(book, "where", None)
                        if callable(where_fn):
                            where = where_fn(q)
                    except Exception:
                        where = None
                    if where is not None and getattr(where, "other_zone", ""):
                        rank = 2_000_000.0
                        bits.append(f"inna strefa: {where.other_zone}")
                    else:
                        distance = getattr(where, "distance_yd", None) if where is not None else None
                        rank = float(area_rank(area) + (distance or 0))
                        if distance is not None:
                            bits.append(f"~{int(round(distance))} yd")
                    add(getattr(q, "title", ""),
                        "Do oddania" if getattr(q, "complete", False) else "Dziennik",
                        ", ".join(bits), rank)
                except Exception:
                    continue
    except Exception:
        pass

    # 2. quests available to take right now (needs a known player and completed list)
    try:
        if world is not None and getattr(player, "done_known", False) and getattr(player, "level", 0):
            active: set = set()
            try:
                if getattr(book, "known", False):
                    active = {q.quest_id for q in book.quests()}
            except Exception:
                active = set()
            level = player.level
            try:
                quest_items = list(world.quests.items())
            except Exception:
                quest_items = []
            for qid, q in quest_items:
                try:
                    lvl = q.get("lvl") or q.get("min") or 0
                    if lvl and not (level - 4 <= lvl <= level + 2):
                        continue
                    if not world.available(q, qid, player, active):
                        continue
                    bits = []
                    if lvl:
                        bits.append(f"poziom {lvl}")
                    area = 0
                    try:
                        area = int(q.get("zone") or 0)
                    except Exception:
                        area = 0
                    if area:
                        try:
                            zone_name = world.names.get(area, "")
                        except Exception:
                            zone_name = ""
                        if zone_name:
                            bits.append(zone_name)
                    rank = area_rank(area)
                    spot = None
                    if area == current_area and pos is not None:
                        try:
                            spot = world.starter_spot(qid, pos, getattr(player, "faction", ""))
                        except Exception:
                            spot = None
                    if spot is not None and getattr(spot, "distance", None) is not None:
                        rank = float(spot.distance)
                        bits.append(f"u {spot.name}, ~{int(round(spot.distance))} yd")
                    add(q.get("n", ""), "Do wzięcia", ", ".join(bits), rank)
                except Exception:
                    continue
    except Exception:
        pass

    # 3. what is in the bags, before QuestieDB names so it wins dedupe
    if bags is not None:
        try:
            if getattr(bags, "known", False):
                for it in bags.items:
                    try:
                        quality = int(getattr(it, "quality", 0) or 0)
                        count = int(getattr(it, "count", 1) or 1)
                        add(it.name, "Plecak", f"{count}x, jakość {quality}", -float(quality))
                    except Exception:
                        continue
        except Exception:
            pass

    if world is not None:
        # 4. every QuestieDB quest name
        try:
            for q in world.quests.values():
                try:
                    area = 0
                    try:
                        area = int(q.get("zone") or 0)
                    except Exception:
                        area = 0
                    detail = ""
                    try:
                        detail = world.names.get(area, "")
                    except Exception:
                        detail = ""
                    add(q.get("n", ""), "Quest", detail, area_rank(area))
                except Exception:
                    continue
        except Exception:
            pass

        # 5. NPCs
        try:
            for n in world.npcs.values():
                try:
                    rank = 0
                    if "z" in n:
                        rank = area_rank(_area_id(world, n.get("z")))
                    add(n.get("n", ""), "NPC", "", rank)
                except Exception:
                    continue
        except Exception:
            pass

        # 6. places / zones
        try:
            for area, name in world.names.items():
                try:
                    add(name, "Miejsce", "", area_rank(_area_id(world, area)))
                except Exception:
                    continue
        except Exception:
            pass

        # 7. dungeons
        try:
            for d in world.dungeons.values():
                try:
                    anchor = d.get("parent") or (d.get("at") or [0])[0]
                    add(d.get("n", ""), "Dungeon", "", area_rank(_area_id(world, anchor)))
                except Exception:
                    continue
        except Exception:
            pass

        # 8. items
        try:
            for it in world.items.values():
                try:
                    add(it.get("n", ""), "Przedmiot")
                except Exception:
                    continue
        except Exception:
            pass

    # 9. party members
    try:
        for name in getattr(state, "members", None) or []:
            try:
                add(name, "Grupa")
            except Exception:
                continue
    except Exception:
        pass

    items.sort(key=lambda s: (GROUP_ORDER.get(s.group, 99), s.rank, s.name.casefold()))
    return items


def _score(key: str, query: str) -> Optional[tuple[int, int]]:
    """``(tier, distance)``: exact 0, prefix/word start 1, substring 2, None otherwise."""
    if key == query:
        return 0, 0
    if key.startswith(query):
        return 1, 0
    if any(word.startswith(query) for word in key.split()):
        return 1, 0
    if query in key:
        return 2, 0
    return None


def _osa(a: str, b: str, max_dist: int) -> int:
    """Optimal-string-alignment distance, bounded: returns ``max_dist + 1`` when over."""
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    if la == 0:
        return lb
    if lb == 0:
        return la
    big = max_dist + 1
    prev2 = None
    prev = [big] * (lb + 1)
    for j in range(0, min(lb, max_dist) + 1):
        prev[j] = j
    for i in range(1, la + 1):
        cur = [big] * (lb + 1)
        lo = max(1, i - max_dist)
        hi = min(lb, i + max_dist)
        if lo == 1:
            cur[0] = i
        best = cur[0] if lo == 1 else big
        ai = a[i - 1]
        for j in range(lo, hi + 1):
            cost = 0 if ai == b[j - 1] else 1
            v = prev[j] + 1
            t = cur[j - 1] + 1
            if t < v:
                v = t
            t = prev[j - 1] + cost
            if t < v:
                v = t
            if i > 1 and j > 1 and ai == b[j - 2] and a[i - 2] == b[j - 1]:
                t = prev2[j - 2] + 1
                if t < v:
                    v = t
            cur[j] = v
            if v < best:
                best = v
        if best > max_dist:
            return max_dist + 1
        prev2, prev = prev, cur
    return prev[lb]


def filter_suggestions(items: Iterable[Suggestion], query: str, limit: int = 200) -> list[Suggestion]:
    """Best matches for ``query``: exact, prefix, word start, substring, then fuzzy typos."""
    items = list(items)
    q = fold(query)
    if not q:
        return items[:limit]

    scored = []
    nonfuzzy = 0
    fuzzy_candidates = []
    for it in items:
        key = it.key or fold(it.name)
        s = _score(key, q)
        if s is not None:
            tier, distance = s
            scored.append((tier, distance, GROUP_ORDER.get(it.group, 99), it.rank,
                           len(it.name), it.name.casefold(), key, it))
            nonfuzzy += 1
        elif len(q) >= 3:
            fuzzy_candidates.append((key, it))

    if nonfuzzy < limit and fuzzy_candidates:
        max_dist = 1 if len(q) <= 5 else 2
        for key, it in fuzzy_candidates:
            best = max_dist + 1
            for word in key.split():
                # a typo of a word start: the first letter is assumed known (cheap cut)
                if not word or word[0] != q[0]:
                    continue
                if abs(len(word) - len(q)) <= max_dist:
                    d = _osa(q, word, max_dist)
                    if d < best:
                        best = d
                if len(word) > len(q):
                    d = _osa(q, word[:len(q)], max_dist)
                    if d < best:
                        best = d
                if best == 1:
                    break
            if best <= max_dist:
                scored.append((3, best, GROUP_ORDER.get(it.group, 99), it.rank,
                               len(it.name), it.name.casefold(), key, it))

    scored.sort(key=lambda t: t[:6])

    out: list[Suggestion] = []
    seen: set[str] = set()
    for _, _, _, _, _, _, key, it in scored:
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= limit:
            break
    return out


def apply_delimited(text: str, cursor: int, name: str, opener: str = "[", closer: str = "]") -> tuple[str, int]:
    """Replace the open ``opener`` and what was typed with ``opener + name + closer``.

    An existing ``closer`` right at the cursor is consumed rather than duplicated; text
    after it is kept. Returns the new text and the cursor placed after the inserted
    closer.
    """
    ctx = delimiter_context(text, cursor, opener, closer)
    if ctx is None:
        return text, cursor
    start, _ = ctx
    end = cursor
    if end < len(text) and text[end] == closer:
        end += 1
    new_text = text[:start] + opener + name + closer + text[end:]
    return new_text, start + len(name) + 2


def apply_suggestion(text: str, cursor: int, name: str) -> tuple[str, int]:
    """Replace the open ``[`` and what was typed with ``[name]``; return text and new cursor."""
    return apply_delimited(text, cursor, name, "[", "]")
