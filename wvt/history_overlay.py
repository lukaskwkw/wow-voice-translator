"""Iskra's always-on-top conversation history placed over WoW.

A normal, focusable tool window (not click-through): the user can scroll it and
close it. It polls the assistant's in-memory transcript with a timer, so no
extra signals are needed from the worker thread.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLineEdit, QPushButton, QTextBrowser, QVBoxLayout

from .history_text import render_history_html

__all__ = ["HistoryOverlay", "render_history_html", "update_history_view"]

_STYLE = """
QDialog { background: #14141c; }
QTextBrowser { background: #1b1b26; color: #e6e6ee; border: 1px solid #2c2c3a; }
QPushButton { background: #26263a; color: #e6e6ee; border: 1px solid #3a3a52; padding: 4px 10px; }
QPushButton:hover { background: #32324a; }
"""


def update_history_view(view, html: str, force: bool = False) -> bool:
    """Set the history HTML only when it changed, keeping the user's selection and scroll position."""
    if not force and getattr(view, "_wvt_history_html", None) == html:
        return False
    if not force and view.hasFocus() and view.textCursor().hasSelection():
        return False  # never wipe a selection the user is making/holding
    bar = view.verticalScrollBar()
    at_bottom = bar.value() >= bar.maximum() - 2
    value = bar.value()
    view.setHtml(html)
    view._wvt_history_html = html
    if at_bottom:
        bar.setValue(bar.maximum())
    else:
        bar.setValue(min(value, bar.maximum()))
    return True


class HistoryOverlay(QDialog):
    """Always-on-top history window; `items_getter()` returns the transcript list."""

    closed = Signal()

    def __init__(self, items_getter, clear_callback, cfg, on_ask=None, on_skip=None, skip_pending=None,
                 on_prompt=None, parent=None):
        super().__init__(parent)
        self._items_getter = items_getter
        self._clear_callback = clear_callback
        self.cfg = cfg
        self._on_ask = on_ask
        self._on_skip = on_skip
        self._skip_pending = skip_pending
        self._on_prompt = on_prompt
        self.setWindowTitle("Iskra – historia rozmów")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.resize(440, 520)
        self.setStyleSheet(_STYLE)

        lay = QVBoxLayout(self)
        self.view = QTextBrowser()
        self.view.setReadOnly(True)
        self.view.setOpenExternalLinks(False)
        lay.addWidget(self.view, 1)

        ask_row = QHBoxLayout()
        self.ask = QLineEdit()
        self.ask.setPlaceholderText("Napisz do Iskry…")
        self.ask.returnPressed.connect(self._send)
        b_send = QPushButton("Wyślij")
        b_send.clicked.connect(self._send)
        self.b_skip = QPushButton("Bez reguł")
        self.b_skip.setCheckable(True)
        self.b_skip.setToolTip("Następne pytanie pójdzie do rozmowy bez reguł i bez Jev (jednorazowo)")
        self.b_skip.toggled.connect(self._toggle_skip)
        ask_row.addWidget(self.ask, 1)
        ask_row.addWidget(b_send)
        ask_row.addWidget(self.b_skip)
        lay.addLayout(ask_row)

        row = QHBoxLayout()
        b_clear = QPushButton("Wyczyść")
        b_clear.clicked.connect(self._on_clear)
        b_copy = QPushButton("Kopiuj")
        b_copy.clicked.connect(self._copy_selection)
        self.b_prompt = QPushButton("Prompt")
        self.b_prompt.clicked.connect(self._show_prompt)
        b_close = QPushButton("Zamknij")
        b_close.clicked.connect(self.close)
        row.addWidget(b_clear)
        row.addWidget(b_copy)
        row.addWidget(self.b_prompt)
        row.addStretch(1)
        row.addWidget(b_close)
        lay.addLayout(row)

        self._placed = False
        self._timer = QTimer(self)
        self._timer.setInterval(700)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    # ------------------------------------------------------------------ content
    def refresh(self, force: bool = False) -> None:
        items = self._items_getter() or []
        limit = int(self.cfg.get("assistant.history_limit", 10) or 10)
        name = str(self.cfg.get("assistant.name", "Iskra") or "Iskra")
        update_history_view(self.view, render_history_html(items, limit, name), force)
        if self.b_skip.isChecked() and self._skip_pending is not None and not self._skip_pending():
            self.b_skip.blockSignals(True)
            self.b_skip.setChecked(False)
            self.b_skip.blockSignals(False)

    def _on_clear(self) -> None:
        try:
            self._clear_callback()
        finally:
            self.refresh(force=True)

    def _copy_selection(self) -> None:
        cursor = self.view.textCursor()
        if not cursor.hasSelection():
            return
        QGuiApplication.clipboard().setText(cursor.selection().toPlainText())

    def _send(self) -> None:
        text = self.ask.text().strip()
        if not text:
            return
        self.ask.clear()
        if self._on_ask is None:
            return
        try:
            self._on_ask(text)
        except Exception:
            pass

    def _toggle_skip(self, checked: bool) -> None:
        if self._on_skip is None:
            self.b_skip.blockSignals(True)
            self.b_skip.setChecked(False)
            self.b_skip.blockSignals(False)
            return
        try:
            self._on_skip(checked)
        except Exception:
            self.b_skip.blockSignals(True)
            self.b_skip.setChecked(False)
            self.b_skip.blockSignals(False)

    def _show_prompt(self) -> None:
        if self._on_prompt is None:
            return
        try:
            self._on_prompt()
        except Exception:
            pass

    # ------------------------------------------------------------------ window
    def showEvent(self, event):  # noqa: N802 - Qt override
        super().showEvent(event)
        if not self._placed:
            self._placed = True
            self._place()

    def _place(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        g = screen.availableGeometry()
        self.move(g.right() - 460, g.top() + 80)

    def closeEvent(self, event):  # noqa: N802 - Qt override
        self.hide()
        self.closed.emit()
        event.accept()
