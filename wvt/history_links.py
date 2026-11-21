"""Turn known game names in Iskra's replies into coloured `[Name]` links (no Qt import).

The history HTML only needs a callable that maps a raw reply to HTML. When the
feature is off, or the world data is not loaded, the callable falls back to the
plain rendering from `history_text`, so both views look exactly as before.
"""

from __future__ import annotations

import html
import re
from typing import Callable, Optional

from .assistant import fold
from .history_text import _light_markdown

__all__ = ["KIND_COLORS", "PRIORITY", "entity_command", "dungeon_command", "EntityLinker"]

KIND_COLORS = {
    "quest": "#ffd100",
    "item": "#1eff00",
    "npc": "#ff9f43",
    "dungeon": "#c792ea",
    "place": "#40c4ff",
}

PRIORITY = ("quest", "item", "npc", "dungeon", "place")

_PRIORITY_INDEX = {kind: pos for pos, kind in enumerate(PRIORITY)}
_LINK_KINDS = ("quest", "item", "npc")
_MAX_LINKS = 40
_MAX_WORDS = 6
_MIN_FOLDED = 4
# A word is matched without the punctuation and brackets around it.
_LEAD = "([{\"'"
_TRAIL = ".,:;!?()]}\"'"
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$")
_BULLET_RE = re.compile(r"^-[ \t]+")
_ID_HINT_RE = re.compile(r"\(\s*id\s+(\d+)\s*\)")


def entity_command(kind: str, ident: int) -> Optional[str]:
    """The WoW chat line that opens Questie's own details window, or None.

    Questie's `/questie search item|quest` is a TODO in its source, so the only
    working call is Questie's details window through QuestieLoader.
    """
    if kind not in _LINK_KINDS:
        return None
    try:
        number = int(ident)
    except (TypeError, ValueError):
        return None
    return f'/run QuestieLoader:ImportModule("QuestieSearchResults"):GetDetailFrame("{kind}",{number})'


def dungeon_command(world, area: int) -> Optional[str]:
    """The `/wvt way` line to a dungeon entrance, or None when the area is unknown."""
    if world is None:
        return None
    from .advisor import _entrance, waypoint_line

    dungeons = getattr(world, "dungeons", None)
    dungeon = dungeons.get(area) if isinstance(dungeons, dict) else None
    name = str(dungeon.get("n") or "") if isinstance(dungeon, dict) else ""
    if not name:
        return None
    spot = _entrance(world, name)
    return waypoint_line(spot, name) if spot is not None else None


def _strip_word(word: str) -> str:
    return word.strip(_LEAD + _TRAIL)


def _build_index(world) -> dict[str, list[tuple[str, int]]]:
    index: dict[str, list[tuple[str, int]]] = {}

    def add(kind: str, ident, name) -> None:
        folded = fold(str(name or ""))
        if len(folded) < _MIN_FOLDED:
            return
        try:
            number = int(ident)
        except (TypeError, ValueError):
            return
        bucket = index.setdefault(folded, [])
        if any(k == kind for k, _ in bucket):
            return  # first id per (kind, folded name)
        bucket.append((kind, number))

    def rows(table) -> list[tuple[object, object]]:
        try:
            return list(table.items())
        except AttributeError:
            return []

    for key, q in rows(getattr(world, "quests", None)):
        if isinstance(q, dict):
            add("quest", key, q.get("n"))
    for key, it in rows(getattr(world, "all_items", None)):
        if isinstance(it, dict):
            add("item", key, it.get("n"))
    for key, n in rows(getattr(world, "npcs", None)):
        if isinstance(n, dict):
            add("npc", key, n.get("n"))
    for key, d in rows(getattr(world, "dungeons", None)):
        if isinstance(d, dict):
            add("dungeon", key, d.get("n"))

    names = getattr(world, "names", None)
    if isinstance(names, dict):
        for area, zone_name in names.items():
            add("place", area, zone_name)
    return index


class EntityLinker:
    """Callable: raw reply -> HTML with coloured, clickable names."""

    def __init__(self, world_getter: Callable[[], Optional[object]],
                 enabled: Callable[[], bool] = lambda: True):
        self._world_getter = world_getter
        self._enabled = enabled
        self._world = None
        self._built = False
        self._index: dict[str, list[tuple[str, int]]] = {}

    # ------------------------------------------------------------------ index
    def _get_index(self, world) -> dict[str, list[tuple[str, int]]]:
        if not self._built or world is not self._world:
            self._world = world
            self._built = True
            self._index = _build_index(world)
        return self._index

    # ------------------------------------------------------------------ render
    def __call__(self, text: str) -> str:
        text = str(text or "")
        world = self._world_getter()
        if not self._enabled() or world is None:
            return _light_markdown(html.escape(text))
        index = self._get_index(world)
        if not index:
            return _light_markdown(html.escape(text))
        budget = [_MAX_LINKS]
        out: list[str] = []
        for line in text.split("\n"):
            heading = _HEADING_RE.match(line)
            if heading:
                out.append("<b>" + self._render_line(heading.group(2), index, budget) + "</b>")
            else:
                out.append(self._render_line(line, index, budget))
        return "\n".join(out)

    def _render_line(self, line: str, index, budget: list) -> str:
        bullet = _BULLET_RE.match(line)
        if bullet:
            line = "•" + line[1:]  # same length keeps token offsets valid
        tokens: list[tuple[str, int, int]] = []
        for m in re.finditer(r"\S+", line):
            raw = m.group(0)
            tokens.append((fold(_strip_word(raw)), m.start(), m.end()))

        matches: list[tuple[int, int, str, str, int]] = []
        i = 0
        while i < len(tokens):
            found = 0
            found_folded = ""
            for length in range(min(_MAX_WORDS, len(tokens) - i), 0, -1):
                window = tokens[i:i + length]
                if any(not t[0] for t in window):
                    continue  # punctuation-only words are not part of a name
                folded = " ".join(t[0] for t in window)
                if folded in index:
                    found = length
                    found_folded = folded
                    break
            if not found:
                i += 1
                continue

            start = tokens[i][1]
            end = tokens[i + found - 1][2]
            if start > 0 and line[start - 1] == "[" and end < len(line) and line[end] == "]":
                start -= 1
                end += 1
            while end > start and line[end - 1] in ".,:;":
                end -= 1
            raw = line[start:end]
            bracketed = raw.startswith("[") and raw.endswith("]")

            candidates = index[found_folded]
            kind, ident = self._pick(candidates, line[end:end + 12])
            if kind in _LINK_KINDS and budget[0] > 0:
                budget[0] -= 1
                display = raw if bracketed else "[" + raw.rstrip(".,:;") + "]"
                matches.append((start, end, html.escape(display), kind, ident))
            elif kind in _LINK_KINDS:
                pass  # over the cap: stays plain text
            else:
                display = raw if bracketed else "[" + raw.rstrip(".,:;") + "]"
                matches.append((start, end, html.escape(display), kind, ident))
            i += found

        parts: list[str] = []
        pos = 0
        for start, end, display, kind, ident in matches:
            parts.append(self._plain(line[pos:start]))
            color = KIND_COLORS.get(kind, "#ffffff")
            if kind in _LINK_KINDS:
                parts.append(f'<a href="wvt:{kind}:{ident}" '
                             f'style="color:{color};text-decoration:none">{display}</a>')
            else:
                parts.append(f'<span style="color:{color}">{display}</span>')
            pos = end
        parts.append(self._plain(line[pos:]))
        return "".join(parts)

    @staticmethod
    def _pick(candidates: list[tuple[str, int]], tail: str) -> tuple[str, int]:
        if len({kind for kind, _ in candidates}) > 1:
            hint = _ID_HINT_RE.search(tail or "")
            if hint:
                wanted = int(hint.group(1))
                for kind, ident in candidates:
                    if ident == wanted:
                        return kind, ident
        return min(candidates, key=lambda c: _PRIORITY_INDEX.get(c[0], 99))

    @staticmethod
    def _plain(text: str) -> str:
        text = html.escape(text)
        text = text.replace(" -&gt; ", " → ").replace(" -> ", " → ")
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
