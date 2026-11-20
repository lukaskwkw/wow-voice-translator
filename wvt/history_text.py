"""HTML rendering of Iskra's in-memory conversation history (no Qt import).

Used by the Iskra tab and the always-on-top overlay, and tested without a
QApplication.
"""

from __future__ import annotations

import html
import time

EMPTY_TEXT = "Brak rozmów. Przytrzymaj mouse5 i powiedz coś do Iskry."


def render_history_html(items: list[dict], limit: int = 10, name: str = "Iskra", show_events: bool = True) -> str:
    """The newest `limit` entries, oldest first, as simple inline HTML for QTextBrowser.

    `show_events=False` hides Iskra's spontaneous comments (entries with `event`).
    """
    if not items:
        return f"<p style='color:#888888'>{html.escape(EMPTY_TEXT)}</p>"
    try:
        shown = int(limit)
    except (TypeError, ValueError):
        shown = 10
    if shown <= 0:
        return ""
    seq = list(items)
    if not show_events:
        seq = [e for e in seq if not e.get("event")]
        if not seq:
            return f"<p style='color:#888888'>{html.escape(EMPTY_TEXT)}</p>"
    entries = seq[:shown][::-1]  # newest first -> chronological
    parts: list[str] = []
    for e in entries:
        at = float(e.get("at") or 0)
        when = time.strftime("%H:%M:%S", time.localtime(at)) if at else "--:--:--"
        heard = html.escape(str(e.get("heard") or ""))
        reply = html.escape(str(e.get("reply") or ""))
        fixed = html.escape(str(e.get("fixed") or ""))
        intent = html.escape(str(e.get("intent") or ""))
        if e.get("event"):
            event = html.escape(str(e.get("event")))
            block = (
                f"<div style='margin-bottom:10px'>"
                f"<span style='color:#888888'>{when}</span> "
                f"<b style='color:#ff9f43'>{html.escape(name)} (automatycznie):</b> {reply}"
                f"<br><span style='color:#777777;font-size:9pt'>event: {event}</span>"
                f"</div>"
            )
            parts.append(block)
            continue
        ms = e.get("ms")
        bits = [intent, f"{ms} ms" if ms is not None else ""]
        llm = e.get("llm") if isinstance(e.get("llm"), dict) else {}
        if llm:
            who = "/".join(html.escape(str(x)) for x in (llm.get("provider"), llm.get("model")) if x)
            if who:
                bits.append(who)
            if llm.get("variant"):
                bits.append("variant=" + html.escape(str(llm["variant"])))
            if llm.get("fallback"):
                bits.append("fallback=local")
        meta = " · ".join(b for b in bits if b)
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
