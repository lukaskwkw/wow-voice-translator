"""The last prompt sent to the model, shown in an always-on-top tool window.

Memory only: the text comes straight from the assistant and is never logged.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase, QGuiApplication, QTextOption
from PySide6.QtWidgets import QDialog, QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout

__all__ = ["PromptDialog"]

_STYLE = """
QDialog { background: #14141c; }
QPlainTextEdit { background: #1b1b26; color: #e6e6ee; border: 1px solid #2c2c3a; }
QPushButton { background: #26263a; color: #e6e6ee; border: 1px solid #3a3a52; padding: 4px 10px; }
QPushButton:hover { background: #32324a; }
"""


class PromptDialog(QDialog):
    """Non-modal window with the exact last LLM request and a copy button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ostatni prompt do LLM")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setModal(False)
        self.resize(780, 640)
        self.setStyleSheet(_STYLE)

        lay = QVBoxLayout(self)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.view.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.view.setFont(font)
        lay.addWidget(self.view, 1)

        row = QHBoxLayout()
        b_copy = QPushButton("Kopiuj")
        b_copy.clicked.connect(self._copy)
        b_close = QPushButton("Zamknij")
        b_close.clicked.connect(self.close)
        row.addStretch(1)
        row.addWidget(b_copy)
        row.addWidget(b_close)
        lay.addLayout(row)

        self._placed = False

    # ------------------------------------------------------------------ content
    def set_text(self, text: str) -> None:
        self.view.setPlainText(text)
        cursor = self.view.textCursor()
        cursor.movePosition(cursor.MoveOperation.Start)
        self.view.setTextCursor(cursor)

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.view.toPlainText())

    # ------------------------------------------------------------------ window
    def showEvent(self, event):  # noqa: N802 - Qt override
        super().showEvent(event)
        if not self._placed:
            self._placed = True
            self._center()

    def _center(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        g = screen.availableGeometry()
        self.move(g.center().x() - self.width() // 2, g.center().y() - self.height() // 2)
