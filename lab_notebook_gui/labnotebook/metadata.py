from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSpinBox, QTableView, QVBoxLayout, QWidget


class MetadataView(QWidget):
    """Tabular view that reads and displays metadata.csv files."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = QStandardItemModel(self)
        self._table = QTableView(self)
        self._table.setModel(self._model)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setWordWrap(False)

        self._font_spin = QSpinBox(self)
        self._font_spin.setRange(8, 32)
        self._font_spin.setValue(11)
        self._font_spin.valueChanged.connect(self._apply_font_size)

        self._wrap_check = QCheckBox("Wrap text", self)
        self._wrap_check.stateChanged.connect(self._toggle_wrap)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(QLabel("Font size:", self))
        controls.addWidget(self._font_spin)
        controls.addSpacing(12)
        controls.addWidget(self._wrap_check)
        controls.addStretch(1)

        self._placeholder = QLabel("metadata.csv not available.", self)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #777; font-style: italic;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(controls)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._placeholder, 1)

        self._current_path: Path | None = None
        self._apply_font_size(self._font_spin.value())
        self._update_placeholder()

    def load_csv(self, csv_path: Path | str | None) -> None:
        self._model.clear()
        self._current_path = Path(csv_path) if csv_path else None
        if not self._current_path or not self._current_path.exists():
            self._update_placeholder()
            return
        try:
            with self._current_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.reader(fh)
                rows: List[Sequence[str]] = [row for row in reader]
        except UnicodeDecodeError:
            with self._current_path.open("r", errors="ignore", newline="") as fh:
                reader = csv.reader(fh)
                rows = [row for row in reader]

        if not rows:
            self._update_placeholder(message="metadata.csv is empty.")
            return

        header = [str(value) for value in rows[0]]
        self._model.setHorizontalHeaderLabels(header)

        for row in rows[1:]:
            items = [QStandardItem(str(value)) for value in row]
            for item in items:
                item.setEditable(False)
            self._model.appendRow(items)

        self._table.resizeColumnsToContents()
        self._update_placeholder(visible=False)

    def clear(self) -> None:
        self._model.clear()
        self._current_path = None
        self._update_placeholder()

    def _apply_font_size(self, size: int) -> None:
        font = QFont()
        font.setPointSize(size)
        self._table.setFont(font)
        self._table.horizontalHeader().setFont(font)

    def _toggle_wrap(self, state: int) -> None:
        should_wrap = state == Qt.CheckState.Checked.value
        self._table.setWordWrap(should_wrap)
        if should_wrap:
            self._table.resizeRowsToContents()

    def _update_placeholder(self, visible: bool | None = None, message: str | None = None) -> None:
        if message:
            self._placeholder.setText(message)
        if visible is None:
            visible = not self._model.rowCount()
        self._placeholder.setVisible(visible)
        self._table.setVisible(not visible)
