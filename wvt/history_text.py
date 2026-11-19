"""HTML rendering of Iskra's in-memory conversation history (no Qt import).

Used by the Iskra tab and the always-on-top overlay, and tested without a
QApplication.
"""

from __future__ import annotations

import html
import time

EMPTY_TEXT = "Brak rozmów. Przytrzymaj mouse5 i powiedz coś do Iskry."


def render_history_html(items: list[dict], limit: int = 10, name: str = "Iskra") -> str:
    """The newest `limit` entries, oldest first, as simple inline HTML for QTextBrowser."""
    if not items:
        return f"<p style='color:#888888'>{html.escape(EMPTY_TEXT)}</p>"
    try:
        shown = int(limit)
    except (TypeError, ValueError):
        shown = 10
    if shown <= 0:
        return ""
    entries = list(items)[:shown][::-1]  # newest first -> chronological
    parts: list[str] = []
    for e in entries:
        at = float(e.get("at") or 0)
        when = time.strftime("%H:%M:%S", time.localtime(at)) if at else "--:--:--"
        heard = html.escape(str(e.get("heard") or ""))
        reply = html.escape(str(e.get("reply") or ""))
        fixed = html.escape(str(e.get("fixed") or ""))
        intent = html.escape(str(e.get("intent") or ""))
        ms = e.get("ms")
        meta = intent + (f" · {ms} ms" if ms is not None else "")
        block = (
            f"<div style='margin-bottom:10px'>"
            f"<span style='color:#888888'>{when}</span> "
            f"<b>Ty:</b> {heard}"
        )
        if fixed:
            block += f"<br><span style='color:#888888'>usłyszane: {fixed}</span>"
        block += (
            f"<br><b style='color:#ffd100'>{html.escape(name)}:</b> {reply}"
            f"<br><span style='color:#777777;font-size:9pt'>{meta}</span>"
            f"</div>"
        )
        parts.append(block)
    return "".join(parts)
