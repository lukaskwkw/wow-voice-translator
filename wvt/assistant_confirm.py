"""Small always-on-top dialog to edit the speech transcript before Iskra (the assistant) sees it.

Separate from ``wvt/confirm_dialog.py``: that one edits the final chat line, this one edits the
Polish transcript of the push-to-talk channel that belongs to the assistant.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

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


class AssistantConfirmDialog(QDialog):
    def __init__(self, text: str, on_send, on_cancel, name: str = "Iskra", parent=None):
        super().__init__(parent)
        self._on_send = on_send
        self._on_cancel = on_cancel
        self._name = name
        self._done = False

        self.setWindowTitle(f"{name} – potwierdź tekst")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setModal(False)
        self.setStyleSheet(STYLE)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)

        label = QLabel(f"{name} usłyszała – popraw i wyślij:")
        label.setWordWrap(True)
        lay.addWidget(label)

        self.edit = QLineEdit(text)
        self.edit.setMaxLength(4000)
        self.edit.returnPressed.connect(self._accept)
        lay.addWidget(self.edit)

        row = QHBoxLayout()
        row.addStretch(1)
        self.send_btn = QPushButton("Wyślij (Enter)")
        self.send_btn.setDefault(True)
        self.send_btn.clicked.connect(self._accept)
        self.cancel_btn = QPushButton("Anuluj (Esc)")
        self.cancel_btn.clicked.connect(self.reject)
        row.addWidget(self.send_btn)
        row.addWidget(self.cancel_btn)
        lay.addLayout(row)

        self.accepted.connect(lambda: self._finish(True))
        self.rejected.connect(lambda: self._finish(False))

    def _accept(self) -> None:
        self.accept()

    def keyPressEvent(self, event):  # noqa: N802 - Esc cancels, everything else as usual
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def _finish(self, ok: bool) -> None:
        if self._done:
            return
        self._done = True
        try:
            if ok:
                self._on_send(self.edit.text())
            else:
                self._on_cancel()
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
            self.resize(620, 120)
            self.move(geo.x() + 80, geo.bottom() - 200)
        QTimer.singleShot(0, self._focus)

    def _focus(self) -> None:
        self.raise_()
        self.activateWindow()
        self.edit.setFocus()
        self.edit.selectAll()
