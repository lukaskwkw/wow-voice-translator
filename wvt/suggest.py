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


GROUP_ORDER = {
    "Do oddania": 0,
    "Dziennik": 1,
    "Do wzięcia": 2,
    "Quest": 3,
    "NPC": 4,
    "Miejsce": 5,
    "Dungeon": 6,
    "Przedmiot": 7,
    "Grupa": 8,
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


def bracket_context(text: str, cursor: int) -> Optional[tuple[int, str]]:
    """``(index_of_bracket, text_after_it)`` for an open ``[`` before ``cursor``, else None.

    A ``]`` met first means the bracket is already closed; a cursor outside the text is
    ignored.
    """
    if cursor < 1 or cursor > len(text):
        return None
    i = cursor - 1
    while i >= 0:
        ch = text[i]
        if ch == "]":
            return None
        if ch == "[":
            return i, text[i + 1:cursor]
        i -= 1
    return None


def _clean(name: str) -> str:
    return " ".join((name or "").split())


def build_suggestions(book, world, player, state=None) -> list[Suggestion]:
    """Every name worth suggesting, in priority order and deduplicated by folded name."""
    items: list[Suggestion] = []
    seen: set[str] = set()

    def add(name: str, group: str, detail: str = "") -> None:
        nice = _clean(name)
        if not nice:
            return
        key = fold(nice)
        if not key or key in seen:
            return
        seen.add(key)
        items.append(Suggestion(nice, group, detail, key))

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
                    add(getattr(q, "title", ""),
                        "Do oddania" if getattr(q, "complete", False) else "Dziennik",
                        ", ".join(bits))
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
                    zone_id = q.get("zone")
                    if zone_id:
                        try:
                            zone_name = world.names.get(zone_id, "")
                        except Exception:
                            zone_name = ""
                        if zone_name:
                            bits.append(zone_name)
                    add(q.get("n", ""), "Do wzięcia", ", ".join(bits))
                except Exception:
                    continue
    except Exception:
        pass

    if world is not None:
        # 3. every QuestieDB quest name
        try:
            for q in world.quests.values():
                try:
                    add(q.get("n", ""), "Quest")
                except Exception:
                    continue
        except Exception:
            pass

        # 4. NPCs
        try:
            for n in world.npcs.values():
                try:
                    add(n.get("n", ""), "NPC")
                except Exception:
                    continue
        except Exception:
            pass

        # 5. places / zones
        try:
            for name in world.names.values():
                try:
                    add(name, "Miejsce")
                except Exception:
                    continue
        except Exception:
            pass

        # 6. dungeons
        try:
            for d in world.dungeons.values():
                try:
                    add(d.get("n", ""), "Dungeon")
                except Exception:
                    continue
        except Exception:
            pass

        # 7. items
        try:
            for it in world.items.values():
                try:
                    add(it.get("n", ""), "Przedmiot")
                except Exception:
                    continue
        except Exception:
            pass

    # 8. party members
    try:
        for name in getattr(state, "members", None) or []:
            try:
                add(name, "Grupa")
            except Exception:
                continue
    except Exception:
        pass

    return items


def _score(key: str, query: str) -> Optional[int]:
    if key == query:
        return 0
    if key.startswith(query):
        return 1
    idx = key.find(query)
    if idx < 0:
        return None
    for i in range(idx, len(key)):
        if key.startswith(query, i) and (i == 0 or key[i - 1] == " "):
            return 2
    return 3


def filter_suggestions(items: Iterable[Suggestion], query: str, limit: int = 200) -> list[Suggestion]:
    """Best matches for ``query``: exact, prefix, word start, substring, in group order."""
    items = list(items)
    q = fold(query)
    if not q:
        return items[:limit]

    scored = []
    for it in items:
        key = it.key or fold(it.name)
        s = _score(key, q)
        if s is None:
            continue
        scored.append((s, GROUP_ORDER.get(it.group, 99), len(it.name), it.name.casefold(), key, it))
    scored.sort(key=lambda t: t[:4])

    out: list[Suggestion] = []
    seen: set[str] = set()
    for _, _, _, _, key, it in scored:
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= limit:
            break
    return out


def apply_suggestion(text: str, cursor: int, name: str) -> tuple[str, int]:
    """Replace the open ``[`` and what was typed with ``[name]``; return text and new cursor."""
    ctx = bracket_context(text, cursor)
    if ctx is None:
        return text, cursor
    start, _ = ctx
    end = cursor
    if end < len(text) and text[end] == "]":
        end += 1
    new_text = text[:start] + "[" + name + "]" + text[end:]
    return new_text, start + len(name) + 2
