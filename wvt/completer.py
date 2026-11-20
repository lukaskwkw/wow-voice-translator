"""``[`` bracket autocomplete popup for the Iskra text inputs.

A thin Qt wrapper around :mod:`wvt.suggest`: it watches a ``QLineEdit``, shows the
candidate names only while the cursor sits inside an open ``[``, and inserts the
chosen name as ``[Name]``. The insertion is done by this class, not by
``QLineEdit``'s own completer, so nothing is ever appended outside the brackets.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QModelIndex, QObject, Qt, QTimer
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QCompleter, QLineEdit

from .suggest import apply_delimited, delimiter_context, filter_suggestions

__all__ = ["BracketCompleter"]


class BracketCompleter(QObject):
    """Show ``[`` suggestions for ``edit``; ``items_getter()`` returns ``Suggestion`` objects."""

    def __init__(self, edit: QLineEdit, items_getter, enabled=None, parent=None, brace_getter=None):
        super().__init__(parent)
        self.edit = edit
        self._items_getter = items_getter
        self._brace_getter = brace_getter
        self._active = ("[", "]")
        self._enabled = enabled
        self._applying = False
        self._pending = False
        self._snapshot = (edit.text(), edit.cursorPosition())
        self._by_display: dict[str, str] = {}

        self._model = QStandardItemModel(self)
        self._completer = QCompleter(self._model, self)
        self._completer.setWidget(edit)  # no edit.setCompleter(): we control insertion
        self._completer.setCompletionMode(QCompleter.CompletionMode.UnfilteredPopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setMaxVisibleItems(10)
        self._completer.setCompletionRole(Qt.ItemDataRole.UserRole)
        self._completer.activated.connect(self._insert)

        self._popup = self._completer.popup()
        if self._popup is not None:
            self._popup.installEventFilter(self)

        edit.installEventFilter(self)
        edit.textEdited.connect(self._update)
        edit.cursorPositionChanged.connect(self._update)

    # ------------------------------------------------------------------ popup
    def popup_visible(self) -> bool:
        popup = self._completer.popup()
        return bool(popup is not None and popup.isVisible())

    def completion_pending(self) -> bool:
        """True right after an item was inserted: the same Enter must not send the question."""
        return self._pending

    def hide_popup(self) -> None:
        popup = self._completer.popup()
        if popup is not None:
            popup.hide()

    def _is_enabled(self) -> bool:
        if self._enabled is None:
            return True
        try:
            return bool(self._enabled())
        except Exception:
            return True

    # ------------------------------------------------------------------ model
    def _fill(self, matches) -> None:
        self._model.clear()
        self._by_display.clear()
        for it in matches:
            display = f"{it.name}  \u00b7  {it.group}" if it.group else it.name
            item = QStandardItem(display)
            item.setData(it.name, Qt.ItemDataRole.UserRole)
            item.setToolTip(it.detail or it.group)
            self._model.appendRow(item)
            self._by_display[display] = it.name

    def _clear_rows(self) -> None:
        self._model.clear()
        self._by_display.clear()

    def _update(self, *_args) -> None:
        if self._applying:
            return
        if not self._is_enabled():
            self._clear_rows()
            self.hide_popup()
            return
        text = self.edit.text()
        cursor = self.edit.cursorPosition()
        self._snapshot = (text, cursor)

        square = delimiter_context(text, cursor, "[", "]")
        brace = delimiter_context(text, cursor, "{", "}") if self._brace_getter is not None else None
        if square is not None and (brace is None or square[0] >= brace[0]):
            ctx, opener, closer = square, "[", "]"
        elif brace is not None:
            ctx, opener, closer = brace, "{", "}"
        else:
            ctx, opener, closer = None, "[", "]"
        if ctx is None:
            self._clear_rows()
            self.hide_popup()
            return
        self._active = (opener, closer)

        try:
            getter = self._brace_getter if opener == "{" else self._items_getter
            items = getter() or []
        except Exception:
            items = []
        matches = filter_suggestions(items, ctx[1])
        if not matches:
            self._clear_rows()
            self.hide_popup()
            return

        self._fill(matches)
        popup = self._completer.popup()
        if not self.popup_visible():
            self._completer.setCompletionPrefix("")
            self._completer.complete()
        if popup is not None:
            popup.setMinimumWidth(max(self.edit.width(), 260))
            popup.setCurrentIndex(self._model.index(0, 0))

    # ------------------------------------------------------------------ keyboard
    def eventFilter(self, obj, event):  # noqa: N802 - Qt override
        if event.type() != QEvent.Type.KeyPress:
            return super().eventFilter(obj, event)
        popup = self._completer.popup()
        if obj is popup and popup is not None:
            return self._popup_key(event)
        if obj is not self.edit:
            return super().eventFilter(obj, event)
        if not self._is_enabled():
            self.hide_popup()
            return super().eventFilter(obj, event)
        if not self.popup_visible():
            return super().eventFilter(obj, event)
        key = event.key()
        rows = self._model.rowCount()

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Tab):
            index = popup.currentIndex() if popup is not None else QModelIndex()
            if not index.isValid():
                index = self._model.index(0, 0)
            if rows <= 0:
                self.hide_popup()
                return True
            self._insert(index)
            return True  # Enter/Tab completes; returnPressed must not send the raw text

        if key == Qt.Key.Key_Escape:
            self.hide_popup()
            return True

        if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
            if rows <= 0:
                self.hide_popup()
                return True
            current = popup.currentIndex().row() if popup is not None else -1
            if current < 0:
                row = 0
            elif key == Qt.Key.Key_Down:
                row = (current + 1) % rows
            else:
                row = (current - 1) % rows
            if popup is not None:
                popup.setCurrentIndex(self._model.index(row, 0))
            return True

        return super().eventFilter(obj, event)

    def _popup_key(self, event):
        """Keys delivered to the popup: Tab/Enter insert here, arrows stay with the popup."""
        if not self._is_enabled():
            self.hide_popup()
            return super().eventFilter(self._popup, event)
        key = event.key()
        popup = self._completer.popup()
        rows = self._model.rowCount()

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Tab):
            index = popup.currentIndex() if popup is not None else QModelIndex()
            if not index.isValid():
                index = self._model.index(0, 0)
            if rows <= 0:
                self.hide_popup()
                return True
            self._insert(index)
            return True

        if key == Qt.Key.Key_Escape:
            self.hide_popup()
            return True

        return super().eventFilter(self._popup, event)

    # ------------------------------------------------------------------ insertion
    def _insert(self, chosen) -> None:
        if not self._is_enabled():
            self.hide_popup()
            return
        if isinstance(chosen, QModelIndex):
            name = chosen.data(Qt.ItemDataRole.UserRole)
        else:
            name = chosen
        if not isinstance(name, str):
            return
        name = self._by_display.get(name, name)

        text, cursor = self._snapshot
        opener, closer = self._active if self._active else ("[", "]")
        new_text, new_cursor = apply_delimited(text, cursor, name, opener, closer)
        self._applying = True
        try:
            self.edit.setText(new_text)
            self.edit.setCursorPosition(new_cursor)
        finally:
            self._applying = False
        self._snapshot = (new_text, new_cursor)
        self.hide_popup()
        self.edit.setFocus()  # the popup click may have taken focus: keep typing
        self._pending = True
        QTimer.singleShot(0, self._clear_pending)

    def _clear_pending(self) -> None:
        self._pending = False
