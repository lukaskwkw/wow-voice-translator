"""Iskra's always-on-top conversation history placed over WoW.

A normal, focusable tool window (not click-through): the user can scroll it and
close it. It polls the assistant's in-memory transcript with a timer, so no
extra signals are needed from the worker thread.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QTextBrowser, QVBoxLayout

from .history_text import render_history_html

__all__ = ["HistoryOverlay", "render_history_html"]

_STYLE = """
QDialog { background: #14141c; }
QTextBrowser { background: #1b1b26; color: #e6e6ee; border: 1px solid #2c2c3a; }
QPushButton { background: #26263a; color: #e6e6ee; border: 1px solid #3a3a52; padding: 4px 10px; }
QPushButton:hover { background: #32324a; }
"""


class HistoryOverlay(QDialog):
    """Always-on-top history window; `items_getter()` returns the transcript list."""

    closed = Signal()

    def __init__(self, items_getter, clear_callback, cfg, parent=None):
        super().__init__(parent)
        self._items_getter = items_getter
        self._clear_callback = clear_callback
        self.cfg = cfg
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

        row = QHBoxLayout()
        b_clear = QPushButton("Wyczyść")
        b_clear.clicked.connect(self._on_clear)
        b_close = QPushButton("Zamknij")
        b_close.clicked.connect(self.close)
        row.addWidget(b_clear)
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
    def refresh(self) -> None:
        items = self._items_getter() or []
        limit = int(self.cfg.get("assistant.history_limit", 10) or 10)
        name = str(self.cfg.get("assistant.name", "Iskra") or "Iskra")
        bar = self.view.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 2
        self.view.setHtml(render_history_html(items, limit, name))
        if at_bottom:
            bar.setValue(bar.maximum())

    def _on_clear(self) -> None:
        try:
            self._clear_callback()
        finally:
            self.refresh()

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
