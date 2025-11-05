from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QTextOption, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTextBrowser,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from .utils import read_text_file, write_text_file


class NotesView(QWidget):
    """Markdown-friendly viewer/editor for notes.md files."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_path: Optional[Path] = None
        self._is_editing = False

        self._preview = QTextBrowser(self)
        self._preview.setOpenExternalLinks(False)
        self._preview.setMarkdown("*notes.md not loaded yet*")

        self._editor = QPlainTextEdit(self)
        tab_width = 4 * self._editor.fontMetrics().horizontalAdvance(" ")
        self._editor.setTabStopDistance(tab_width)
        self._editor.setWordWrapMode(QTextOption.WrapMode.WordWrap)

        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._preview)
        self._stack.addWidget(self._editor)
        self._stack.setCurrentWidget(self._preview)

        self._edit_button = QPushButton("Edit", self)
        self._edit_button.clicked.connect(self.enter_edit_mode)

        self._save_button = QPushButton("Save", self)
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self.save_changes)

        self._cancel_button = QPushButton("Cancel", self)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self.exit_edit_mode)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        controls.addWidget(self._edit_button)
        controls.addWidget(self._save_button)
        controls.addWidget(self._cancel_button)
        controls.addStretch(1)
        controls.addWidget(
            QLabel("Ctrl+E to edit - Ctrl+S to save - Esc to cancel", self)
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(controls)
        layout.addWidget(self._stack, 1)

        QShortcut(QKeySequence("Ctrl+E"), self, activated=self.enter_edit_mode)
        QShortcut(QKeySequence.Save, self, activated=self.save_changes)
        QShortcut(QKeySequence.Cancel, self, activated=self.exit_edit_mode)

    def load_file(self, path: Path | str | None) -> None:
        self._current_path = Path(path) if path else None
        self.exit_edit_mode()
        if not self._current_path or not self._current_path.exists():
            self._preview.setMarkdown("*notes.md not found.*")
            return
        text = read_text_file(self._current_path)
        if text.strip():
            self._preview.setMarkdown(text)
        else:
            self._preview.setMarkdown("*notes.md is empty.*")

    def enter_edit_mode(self) -> None:
        if not self._current_path:
            QMessageBox.information(
                self, "No notes.md", "notes.md not found for this materials directory."
            )
            return
        if self._is_editing:
            return
        self._is_editing = True
        self._editor.setPlainText(read_text_file(self._current_path))
        self._stack.setCurrentWidget(self._editor)
        self._save_button.setEnabled(True)
        self._cancel_button.setEnabled(True)

    def save_changes(self) -> None:
        if not self._is_editing or not self._current_path:
            return
        write_text_file(self._current_path, self._editor.toPlainText())
        self._is_editing = False
        self._save_button.setEnabled(False)
        self._cancel_button.setEnabled(False)
        self._stack.setCurrentWidget(self._preview)
        self.load_file(self._current_path)

    def exit_edit_mode(self) -> None:
        if not self._is_editing:
            self._stack.setCurrentWidget(self._preview)
            return
        self._is_editing = False
        self._save_button.setEnabled(False)
        self._cancel_button.setEnabled(False)
        self._stack.setCurrentWidget(self._preview)
