"""``{rule}`` fragments in an Iskra question: explicit, whitelisted rule kinds.

Qt-free.  ``split`` pulls the known ``{kind}`` tokens out of the text (unknown braces stay),
and ``is_real_question`` tells a bare rule invocation ("{help}") from a hybrid question
("{help} co dalej?").
"""

from __future__ import annotations

import re

from .suggest import fold

# Whitelisted rule kinds and the short Polish descriptions the autocomplete will show.
DIRECTIVES: dict[str, str] = {
    "quest_list": "lista questów z dziennika",
    "quest_turnin": "questy gotowe do oddania",
    "quest_nearest": "najbliższy quest",
    "quest_available": "questy do wzięcia w tej strefie",
    "quest_where": "gdzie iść z questem",
    "quest_progress": "postęp w queście",
    "quest_info": "informacje o jednym queście (LLM)",
    "quest_worth": "czy warto robić questa",
    "gear_next": "questy z nagrodami na upgrade (przybliżone)",
    "gear_next_world": "questy z nagrodami na upgrade w całym świecie (przybliżone)",
    "gear_sources": "gdzie kupić albo zdobyć lepszy sprzęt: vendor, drop z moba, boss, obiekt",
    "gear_debug": "diagnostyka doradcy sprzętu (co widzi: noszone części, item levele nagród)",
    "where_next": "co robić dalej",
    "plan_next": "następny krok planu",
    "stats": "statystyki sesji, expa i złota na godzinę",
    "loot": "co wypadło, need czy greed",
    "bags": "ile miejsca w torbach",
    "gold": "ile masz złota",
    "xp": "poziom i doświadczenie",
    "where": "gdzie jesteś",
    "members": "kto jest w grupie",
    "help": "co potrafię",
    "summary": "streszczenie czatu",
    "dungeon_level": "jaki loch na twój poziom",
    "item_about": "informacje o przedmiocie",
    "item_compare": "porównanie dwóch przedmiotów (bez cen rynkowych)",
}

_TOKEN = re.compile(r"\{([^{}]*)\}")

# Words that only frame a rule ("pokaż mi listę"): no question without something more.
_FILLER = {"pokaz", "mi", "lista", "liste", "tylko", "prosze", "no", "i", "a", "to", "wszystko", "teraz"}


def split(text: str) -> tuple[list[str], str]:
    """Known ``{kind}`` tokens from ``text`` (deduped, in order) and the rest of the text.

    Unknown braces stay in the rest verbatim; whitespace in the rest is collapsed.
    """
    kinds: list[str] = []
    seen: set[str] = set()

    def interesting(match: re.Match) -> str:
        kind = match.group(1).strip().lower()
        if kind not in DIRECTIVES:
            return match.group(0)
        if kind not in seen:
            seen.add(kind)
            kinds.append(kind)
        return " "

    rest = _TOKEN.sub(interesting, text or "")
    return kinds, " ".join(rest.split())


def is_real_question(rest: str) -> bool:
    """Whether the text after the rules asks something (so the LLM should compose an answer)."""
    folded = fold(rest)
    if not folded:
        return False
    if "?" in (rest or ""):
        return True
    return any(word not in _FILLER for word in folded.split())
