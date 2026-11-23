"""Small always-on-top, focusable dialog to edit the chat line before it is inserted."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from .completer import BracketCompleter
from .pipeline import HistoryItem

STYLE = """
QDialog {
    background: #14141c;
    border: 1px solid #ffd100;
    border-radius: 8px;
}
QLabel {
    color: #e8e8f0;
    font-size: 12px;
}
QLineEdit {
    background: #1e1e2a;
    color: #ffffff;
    border: 1px solid #3a3a4a;
    border-radius: 4px;
    padding: 4px 6px;
    font-size: 14px;
    selection-background-color: #ffd100;
    selection-color: #14141c;
}
QPushButton {
    background: #2a2a38;
    color: #e8e8f0;
    border: 1px solid #4a4a5a;
    border-radius: 4px;
    padding: 5px 14px;
}
QPushButton:hover {
    background: #3a3a4a;
}
QPushButton:default {
    background: #ffd100;
    color: #14141c;
    border-color: #ffd100;
}
"""


class ConfirmDialog(QDialog):
    def __init__(self, item: HistoryItem, on_insert, on_cancel, max_bytes: int = 255, parent=None,
                 suggestions=None, directives=None, enabled=None):
        super().__init__(parent)
        self.item = item
        self._on_insert = on_insert
        self._on_cancel = on_cancel
        self._max_bytes = max_bytes
        self._done = False

        self.setWindowTitle("WVT – potwierdź tekst")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setModal(False)
        self.setStyleSheet(STYLE)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)

        label = QLabel(f"Wstaw do czatu ({item.channel.upper()}) – Enter wstawia, Esc anuluje")
        label.setWordWrap(True)
        lay.addWidget(label)

        self.edit = QLineEdit(item.chat_line)
        self.edit.returnPressed.connect(self._accept)
        self.edit.textChanged.connect(self._update_counter)
        lay.addWidget(self.edit)
        self._completer = (BracketCompleter(self.edit, suggestions, enabled=enabled,
                                            brace_getter=directives, parent=self)
                           if suggestions is not None else None)

        row = QHBoxLayout()
        self.counter = QLabel("")
        row.addWidget(self.counter)
        row.addStretch(1)
        self.insert_btn = QPushButton("Wstaw (Enter)")
        self.insert_btn.setDefault(True)
        self.insert_btn.clicked.connect(self._accept)
        self.cancel_btn = QPushButton("Anuluj (Esc)")
        self.cancel_btn.clicked.connect(self.reject)
        row.addWidget(self.insert_btn)
        row.addWidget(self.cancel_btn)
        lay.addLayout(row)

        self.accepted.connect(lambda: self._finish(True))
        self.rejected.connect(lambda: self._finish(False))
        self._update_counter(self.edit.text())

    def _update_counter(self, text: str) -> None:
        size = len(text.encode("utf-8"))
        self.counter.setText(f"{size} / {self._max_bytes} B")
        self.counter.setStyleSheet("color: #ff6b6b;" if size > self._max_bytes else "color: #8a8a9a;")

    def _accept(self) -> None:
        c = getattr(self, "_completer", None)
        if c is not None and (c.popup_visible() or c.completion_pending()):
            return
        if not self.edit.text().strip():
            QMessageBox.information(self, "WVT", "Tekst jest pusty.")
            return
        self.accept()

    def _finish(self, ok: bool) -> None:
        if self._done:
            return
        self._done = True
        try:
            if ok:
                self._on_insert(self.item, self.edit.text())
            else:
                self._on_cancel(self.item)
        finally:
            self.deleteLater()

    def closeEvent(self, event):  # noqa: N802 - closing counts as cancel
        if not self._done:
            self._finish(False)
        super().closeEvent(event)

    def showEvent(self, event):  # noqa: N802 - place near the bottom-left, then take focus
        super().showEvent(event)
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.resize(620, 130)
            self.move(geo.x() + 80, geo.bottom() - 200)
        QTimer.singleShot(0, self._focus)

    def _focus(self) -> None:
        self.raise_()
        self.activateWindow()
        self.edit.setFocus()
        self.edit.selectAll()
