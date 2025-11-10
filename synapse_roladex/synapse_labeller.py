import json
import os
import sys
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

yaml = None  # YAML disabled

try:
    from PySide6.QtCore import (
        Qt,
        QAbstractTableModel,
        QModelIndex,
        QItemSelectionModel,
        QSize,
        QSortFilterProxyModel,
        Signal,
        QObject,
    )
    from PySide6.QtGui import (
        QAction,
        QKeySequence,
        QPixmap,
        QPainter,
        QImage,
        QColor,
        QPen,
        QBrush,
        QPainterPath,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QFileDialog,
        QToolBar,
        QStatusBar,
        QWidget,
        QVBoxLayout,
        QLabel,
        QScrollArea,
        QMessageBox,
        QTableView,
        QDockWidget,
        QLineEdit,
        QComboBox,
        QCheckBox,
        QPushButton,
        QHBoxLayout,
        QDialog,
        QDialogButtonBox,
        QTableWidget,
        QTableWidgetItem,
        QSizePolicy,
        QCompleter,
        QGraphicsView,
        QGraphicsScene,
        QGraphicsPixmapItem,
        QStackedWidget,
        QInputDialog,
        QSlider,
    )
except Exception as exc:  # pragma: no cover - helpful message if PySide6 missing
    print(
        "PySide6 is required to run this application. Please install it with 'pip install PySide6'."
    )
    raise


# -------------------------------
# Utility helpers
# -------------------------------


def natural_sort_key(s: str) -> List:
    import re

    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


SUPPORTED_IMAGE_EXTS = {".png", ".PNG"}


# -------------------------------
# Data storage (CSV-backed)
# -------------------------------


@dataclass
class RowRecord:
    source_id: str
    values: Dict[str, Optional[str]] = field(default_factory=dict)

    def get(self, column: str) -> Optional[str]:
        if column == "source-ID":
            return self.source_id
        return self.values.get(column)

    def set(self, column: str, value: Optional[str]) -> None:
        if column == "source-ID":
            # source-ID is read-only
            return
        self.values[column] = value


class DataStore(QObject):
    data_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.directory: Optional[str] = None
        self.csv_path: Optional[str] = None
        self.records: List[RowRecord] = []
        self.columns: List[str] = []
        self.image_index: Dict[str, str] = {}  # source_id -> absolute file path

    # -------- Files & CSV --------
    def open_directory(self, directory: str) -> Tuple[int, int]:
        self.directory = directory
        # Load or create table headers first
        self.columns = load_or_create_headers(directory)
        self.csv_path = os.path.join(directory, "synapse_labels.csv")
        ids_to_paths = self._scan_images(directory)
        self.image_index = ids_to_paths
        sorted_ids = sorted(ids_to_paths.keys(), key=natural_sort_key)

        if os.path.exists(self.csv_path):
            self._load_csv()
            # Sync: ensure a record per image ID, drop missing image rows
            existing_ids = {rec.source_id for rec in self.records}
            # Add missing
            for sid in sorted_ids:
                if sid not in existing_ids:
                    self.records.append(self._new_record(sid))
            # Drop rows whose image no longer exists
            self.records = [
                rec for rec in self.records if rec.source_id in ids_to_paths
            ]
            # Reorder by sorted_ids for deterministic navigation
            order = {sid: i for i, sid in enumerate(sorted_ids)}
            self.records.sort(key=lambda rec: order.get(rec.source_id, 1_000_000))
        else:
            # Create new empty table
            self.records = [self._new_record(sid) for sid in sorted_ids]
            self._save_csv()

        self.data_changed.emit()
        return len(self.records), len(self.image_index)

    def _scan_images(self, directory: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for name in os.listdir(directory):
            base, ext = os.path.splitext(name)
            if ext in SUPPORTED_IMAGE_EXTS:
                abs_path = os.path.join(directory, name)
                result[base] = abs_path
        return result

    def _new_record(self, source_id: str) -> RowRecord:
        values = {col: None for col in self.columns if col != "source-ID"}
        return RowRecord(source_id=source_id, values=values)

    def _load_csv(self) -> None:
        assert self.csv_path is not None
        self.records = []
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            read_columns = reader.fieldnames or []
            # Use headers from table_headers.json; if CSV contains extra columns, append them
            normalized_cols = list(self.columns)
            for col in read_columns:
                if col and col not in normalized_cols:
                    normalized_cols.append(col)
            self.columns = normalized_cols
            for row in reader:
                source_id = row.get("source-ID", "").strip()
                values: Dict[str, Optional[str]] = {}
                for col in self.columns:
                    if col == "source-ID":
                        continue
                    val = row.get(col, "")
                    val = val if val != "" else None
                    values[col] = val
                if source_id:
                    self.records.append(RowRecord(source_id=source_id, values=values))

    def _save_csv(self) -> None:
        assert self.csv_path is not None
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            for rec in self.records:
                row = {"source-ID": rec.source_id}
                for col in self.columns:
                    if col == "source-ID":
                        continue
                    val = rec.values.get(col)
                    row[col] = "" if val is None else str(val)
                writer.writerow(row)

    # -------- Accessors --------
    def row_count(self) -> int:
        return len(self.records)

    def get_row(self, row: int) -> RowRecord:
        return self.records[row]

    def get_image_path(self, source_id: str) -> Optional[str]:
        return self.image_index.get(source_id)

    def save(self) -> None:
        self._save_csv()

    def set_value(self, row: int, column: str, value: Optional[str]) -> None:
        if row < 0 or row >= len(self.records):
            return
        self.records[row].set(column, value)
        self.data_changed.emit()

    def set_value_for_ids(
        self, source_ids: Set[str], column: str, value: Optional[str]
    ) -> int:
        """Assign value to many rows by source-ID. Returns number of rows updated."""
        if not source_ids:
            return 0
        updated = 0
        for rec in self.records:
            if rec.source_id in source_ids:
                rec.set(column, value)
                updated += 1
        if updated:
            self.data_changed.emit()
        return updated


# -------------------------------
# Table model for viewing/editing
# -------------------------------


class LabelTableModel(QAbstractTableModel):
    def __init__(self, store: DataStore):
        super().__init__()
        self.store = store
        self.store.data_changed.connect(self.handle_store_changed)

    # Required overrides
    def rowCount(
        self, parent: QModelIndex = QModelIndex()
    ) -> int:  # noqa: N802 Qt naming
        return 0 if parent.isValid() else self.store.row_count()

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.store.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        row = index.row()
        col_name = self.store.columns[index.column()]
        rec = self.store.get_row(row)
        if role in (Qt.DisplayRole, Qt.EditRole):
            return rec.get(col_name) or ""
        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ):  # noqa: N802
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.store.columns[section]
        return section + 1

    def flags(self, index: QModelIndex):  # noqa: N802
        if not index.isValid():
            return Qt.NoItemFlags
        col_name = self.store.columns[index.column()]
        base = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if col_name != "source-ID":
            base |= Qt.ItemIsEditable
        return base

    def setData(self, index: QModelIndex, value, role: int = Qt.EditRole):  # noqa: N802
        if role != Qt.EditRole or not index.isValid():
            return False
        row = index.row()
        col_name = self.store.columns[index.column()]
        if col_name == "source-ID":
            return False
        text = str(value).strip()
        self.store.set_value(row, col_name, text if text != "" else None)
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
        return True

    # Store change callback
    def handle_store_changed(self) -> None:
        self.beginResetModel()
        self.endResetModel()


# -------------------------------
# Image viewer widget
# -------------------------------


class ImageViewer(QScrollArea):
    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.setWidget(self.label)
        self._pixmap: Optional[QPixmap] = None

    def load_image(self, path: Optional[str]) -> None:
        if path is None or not os.path.exists(path):
            self._pixmap = None
            self.label.setText("No image")
            return
        pm = QPixmap(path)
        if pm.isNull():
            self._pixmap = None
            self.label.setText("Failed to load image")
            return
        self._pixmap = pm
        self._update_scaled()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_scaled()

    def _update_scaled(self) -> None:
        if self._pixmap is None:
            return
        area_size = self.viewport().size()
        scaled = self._pixmap.scaled(
            area_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label.setPixmap(scaled)


# -------------------------------
# Hotkey management
# -------------------------------


class HotkeyManager(QObject):
    hotkeys_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.config_path: Optional[str] = None
        self.active_column: str = "synapse_type"
        self.auto_advance: bool = True
        # Mapping: column -> {key_char: value}
        self.mappings: Dict[str, Dict[str, str]] = {}
        # Global mapping: key -> {column: str, value: str}
        self.global_mapping: Dict[str, Dict[str, str]] = {}
        # YAML disabled
        self.yaml_dir_path: Optional[str] = None
        self.yaml_global_path: Optional[str] = None

    def for_directory(self, directory: str) -> None:
        self.config_path = os.path.join(directory, ".synapse_hotkeys.json")
        # YAML disabled
        self.yaml_dir_path = None
        self.yaml_global_path = None
        self.load()

    def load(self) -> None:
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.active_column = data.get("active_column", self.active_column)
                self.auto_advance = bool(data.get("auto_advance", self.auto_advance))
                self.mappings = data.get("columns", {}) or {}
                self.global_mapping = data.get("global", {}) or {}
            except Exception:
                # Ignore malformed config; keep defaults
                pass
        else:
            # Create default per-directory hotkeys with required global list
            self._create_default_hotkeys()
            self.save()
        # Ensure keys exist for known columns (active-column mappings)
        for col in set(self.mappings.keys()):
            self.mappings[col] = {
                str(k).lower(): str(v) for k, v in self.mappings[col].items()
            }

    def _create_default_hotkeys(self) -> None:
        # Defaults per user spec
        self.global_mapping = {
            "p": {"column": "synapse-type", "value": "spine"},
            "s": {"column": "synapse-type", "value": "shaft"},
            "c": {"column": "synapse-type", "value": "soma"},
            "x": {"column": "synapse-type", "value": "axon"},
            "u": {"column": "synapse-type", "value": "unclear"},
            "1": {"column": "soma-ID", "value": "soma1"},
            "2": {"column": "soma-ID", "value": "soma2"},
            "3": {"column": "soma-ID", "value": "soma3"},
            "4": {"column": "soma-ID", "value": "soma4"},
            "8": {"column": "soma-ID", "value": "somaUK1"},
            "9": {"column": "soma-ID", "value": "somaUK2"},
            "q": {"column": "soma-ID", "value": "unidentifiable_soma"},
            "a": {"column": "dend-type", "value": "apical"},
            "b": {"column": "dend-type", "value": "basal"},
            "i": {"column": "dend-type", "value": "intermediate"},
        }
        # Normalize keys lower
        self.global_mapping = {
            str(k).lower(): {
                "column": str(v.get("column", "")),
                "value": str(v.get("value", "")),
            }
            for k, v in self.global_mapping.items()
        }
        # No per-column mappings by default
        self.mappings = {}

    def save(self) -> None:
        if not self.config_path:
            return
        data = {
            "active_column": self.active_column,
            "auto_advance": self.auto_advance,
            "columns": self.mappings,
            "global": self.global_mapping,
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.hotkeys_changed.emit()

    def resolve_value(self, column: str, key_text: str) -> Optional[str]:
        if not key_text:
            return None
        return self.mappings.get(column, {}).get(key_text.lower())

    def resolve_global(self, key_text: str) -> Optional[Tuple[str, str]]:
        """Return (column, value) for a global single-key mapping if present."""
        if not key_text:
            return None
        entry = self.global_mapping.get(key_text.lower())
        if entry and "column" in entry and "value" in entry:
            return entry["column"], entry["value"]
        return None

    def _try_load_yaml_presets(self) -> None:
        return


class HotkeyEditorDialog(QDialog):
    def __init__(
        self, manager: HotkeyManager, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hotkey Editor")
        self.manager = manager
        self.resize(600, 520)

        layout = QVBoxLayout(self)

        # Global mappings table
        layout.addWidget(QLabel("Global hotkeys (key → column,value):"))
        self.global_table = QTableWidget(0, 3)
        self.global_table.setHorizontalHeaderLabels(["Key", "Column", "Value"])
        self.global_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.global_table)
        g_btn_row = QHBoxLayout()
        self.g_add_btn = QPushButton("Add Global")
        self.g_del_btn = QPushButton("Remove Global")
        g_btn_row.addWidget(self.g_add_btn)
        g_btn_row.addWidget(self.g_del_btn)
        g_btn_row.addStretch(1)
        layout.addLayout(g_btn_row)

        # Column selector
        self.column_combo = QComboBox()
        for col in self.manager.mappings.keys():
            self.column_combo.addItem(col)
        layout.addWidget(QLabel("Active-column mappings for:"))
        layout.addWidget(self.column_combo)

        # Mapping table (per-column)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        # Controls
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add")
        self.del_btn = QPushButton("Remove")
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.del_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        # Auto-advance and active column
        self.auto_chk = QCheckBox("Auto-advance after assign")
        self.auto_chk.setChecked(self.manager.auto_advance)
        layout.addWidget(self.auto_chk)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        # Connections
        self.column_combo.currentTextChanged.connect(self._load_for_column)
        self.add_btn.clicked.connect(self._add_row)
        self.del_btn.clicked.connect(self._remove_selected)
        self.g_add_btn.clicked.connect(self._g_add_row)
        self.g_del_btn.clicked.connect(self._g_remove_selected)
        buttons.accepted.connect(self._save_and_close)
        buttons.rejected.connect(self.reject)

        # Init
        self._load_globals()
        # Populate per-column list with store columns if empty
        if not self.manager.mappings:
            for col in []:
                self.manager.mappings.setdefault(col, {})
        self._load_for_column(self.column_combo.currentText())

    def _load_globals(self) -> None:
        self.global_table.setRowCount(0)
        for key, entry in sorted(self.manager.global_mapping.items()):
            row = self.global_table.rowCount()
            self.global_table.insertRow(row)
            self.global_table.setItem(row, 0, QTableWidgetItem(key))
            self.global_table.setItem(row, 1, QTableWidgetItem(entry.get("column", "")))
            self.global_table.setItem(row, 2, QTableWidgetItem(entry.get("value", "")))

    def _g_add_row(self) -> None:
        row = self.global_table.rowCount()
        self.global_table.insertRow(row)
        self.global_table.setItem(row, 0, QTableWidgetItem(""))
        self.global_table.setItem(row, 1, QTableWidgetItem(""))
        self.global_table.setItem(row, 2, QTableWidgetItem(""))

    def _g_remove_selected(self) -> None:
        rows = {idx.row() for idx in self.global_table.selectedIndexes()}
        for r in sorted(rows, reverse=True):
            self.global_table.removeRow(r)

    def _load_for_column(self, column: str) -> None:
        self.table.setRowCount(0)
        mapping = self.manager.mappings.get(column, {})
        for key, value in sorted(mapping.items()):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(key))
            self.table.setItem(row, 1, QTableWidgetItem(value))

    def _add_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(""))
        self.table.setItem(row, 1, QTableWidgetItem(""))

    def _remove_selected(self) -> None:
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        for r in sorted(rows, reverse=True):
            self.table.removeRow(r)

    def _save_and_close(self) -> None:
        # Save globals
        gm: Dict[str, Dict[str, str]] = {}
        for row in range(self.global_table.rowCount()):
            k = (
                (
                    self.global_table.item(row, 0).text()
                    if self.global_table.item(row, 0)
                    else ""
                )
                .strip()
                .lower()
            )
            col = (
                self.global_table.item(row, 1).text()
                if self.global_table.item(row, 1)
                else ""
            ).strip()
            val = (
                self.global_table.item(row, 2).text()
                if self.global_table.item(row, 2)
                else ""
            ).strip()
            if not k or not col or not val:
                continue
            if len(k) != 1:
                QMessageBox.warning(
                    self, "Invalid key", "Global keys must be a single character."
                )
                return
            gm[k] = {"column": col, "value": val}
        self.manager.global_mapping = gm

        # Save per-column
        column = self.column_combo.currentText()
        mapping: Dict[str, str] = {}
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            val_item = self.table.item(row, 1)
            key = (key_item.text() if key_item else "").strip().lower()
            val = (val_item.text() if val_item else "").strip()
            if not key or not val:
                continue
            if len(key) != 1:
                QMessageBox.warning(
                    self, "Invalid key", "Keys must be a single character."
                )
                return
            mapping[key] = val
        self.manager.mappings[column] = mapping
        self.manager.active_column = column
        self.manager.auto_advance = self.auto_chk.isChecked()
        self.manager.save()
        self.accept()


# -------------------------------
# Table headers management
# -------------------------------

DEFAULT_HEADERS = [
    "source-ID",
    "synapse-type",
    "soma-ID",
    "soma-depth",
    "dend-type",
    "dend-ID",
    "notes",
]


def load_or_create_headers(directory: str) -> List[str]:
    path = os.path.join(directory, "table_headers.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            headers = list(data) if isinstance(data, list) else DEFAULT_HEADERS
            # Ensure source-ID present and in position 0; remove dupes
            headers = [h for i, h in enumerate(headers) if h and h not in headers[:i]]
            if "source-ID" not in headers:
                headers.insert(0, "source-ID")
            else:
                # Move to front if not first
                headers = ["source-ID"] + [h for h in headers if h != "source-ID"]
            return headers
        except Exception:
            pass
    # Create default
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_HEADERS, f, indent=2)
    except Exception:
        pass
    return list(DEFAULT_HEADERS)


def save_headers(directory: str, headers: List[str]) -> None:
    path = os.path.join(directory, "table_headers.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(headers, f, indent=2)
    except Exception:
        pass


# -------------------------------
# Main window
# -------------------------------


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Synapse Labeller")
        self.resize(1100, 700)

        # Core components
        self.store = DataStore()
        self.table_model = LabelTableModel(self.store)
        self.hotkeys = HotkeyManager()

        # Central area: stacked between single-image viewer and batch master view
        self.viewer = ImageViewer()
        self.batch_view = BatchGraphicsView()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.viewer)  # index 0
        self.stack.addWidget(self.batch_view)  # index 1
        self.setCentralWidget(self.stack)

        # Dock: table view
        self.dock = QDockWidget("Labels", self)
        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setSelectionMode(QTableView.SingleSelection)
        self.table_view.doubleClicked.connect(self._table_double_clicked)
        self.dock.setWidget(self.table_view)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)

        # Toolbar
        self.toolbar = QToolBar("Main")
        self.toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(self.toolbar)

        self._build_actions()
        self._build_toolbar_widgets()

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # State
        self.current_row: int = -1
        self.unlabeled_only: bool = False
        self.batch_mode: bool = False
        self.batch_selected_count: int = 0

        self.batch_view.selection_changed.connect(self._on_batch_selection_changed)
        # Use class-level slot for selection updates
        # self.batch_view.selection_changed.connect(self.on_batch_selection_changed)

        # Keyboard focus handling
        self.installEventFilter(self)

    # ----- UI builders -----
    def _build_actions(self) -> None:
        self.open_act = QAction("Open Directory…", self)
        self.open_act.setShortcut(QKeySequence.Open)
        self.open_act.triggered.connect(self.open_directory)

        self.prev_act = QAction("Prev", self)
        self.prev_act.setShortcut(QKeySequence(Qt.Key_Left))
        self.prev_act.triggered.connect(self.go_prev)

        self.next_act = QAction("Next", self)
        self.next_act.setShortcut(QKeySequence(Qt.Key_Right))
        self.next_act.triggered.connect(self.go_next)

        self.save_act = QAction("Save", self)
        self.save_act.setShortcut(QKeySequence.Save)
        self.save_act.triggered.connect(self.save)

        self.toggle_table_act = QAction("Toggle Table", self)
        self.toggle_table_act.setShortcut(QKeySequence("Ctrl+T"))
        self.toggle_table_act.triggered.connect(self._toggle_table)

        self.edit_hotkeys_act = QAction("Edit Hotkeys…", self)
        self.edit_hotkeys_act.setShortcut(QKeySequence("Ctrl+H"))
        self.edit_hotkeys_act.triggered.connect(self.edit_hotkeys)

        self.help_act = QAction("Shortcut Help", self)
        self.help_act.setShortcut(QKeySequence.HelpContents)
        self.help_act.triggered.connect(self.show_help)

        self.toolbar.addAction(self.open_act)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.prev_act)
        self.toolbar.addAction(self.next_act)
        self.toolbar.addAction(self.save_act)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.toggle_table_act)
        self.toolbar.addAction(self.edit_hotkeys_act)
        self.toolbar.addAction(self.help_act)

        # Reload hotkeys action
        self.reload_hotkeys_act = QAction("Reload Hotkeys", self)
        self.reload_hotkeys_act.setShortcut(QKeySequence("Ctrl+R"))
        self.reload_hotkeys_act.triggered.connect(self._reload_hotkeys)
        self.toolbar.addAction(self.reload_hotkeys_act)

        # Add Column action
        self.add_column_act = QAction("Add Column", self)
        self.add_column_act.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self.add_column_act.triggered.connect(self._add_column)
        self.toolbar.addAction(self.add_column_act)

        # Batch actions
        self.toolbar.addSeparator()
        self.batch_toggle_act = QAction("Batch Mode", self)
        self.batch_toggle_act.setCheckable(True)
        self.batch_toggle_act.setShortcut(QKeySequence("Ctrl+B"))
        self.batch_toggle_act.triggered.connect(self._toggle_batch_mode)
        self.toolbar.addAction(self.batch_toggle_act)

        self.load_master_act = QAction("Load Master/Key", self)
        self.load_master_act.triggered.connect(self._load_master_and_key)
        self.toolbar.addAction(self.load_master_act)

    def _build_toolbar_widgets(self) -> None:
        # Jump to ID
        self.toolbar.addSeparator()
        self.toolbar.addWidget(QLabel("Jump:"))
        self.jump_edit = QLineEdit()
        self.jump_edit.setPlaceholderText("Type source-ID and press Enter…")
        self.jump_edit.returnPressed.connect(self._jump_to_entered)
        self.jump_edit.setMaximumWidth(250)
        self.toolbar.addWidget(self.jump_edit)

        # Active column
        self.toolbar.addSeparator()
        self.toolbar.addWidget(QLabel("Column:"))
        self.column_combo = QComboBox()
        for col in self.store.columns:
            if col != "source-ID":
                self.column_combo.addItem(col)
        self.column_combo.currentTextChanged.connect(self._active_column_changed)
        self.toolbar.addWidget(self.column_combo)

        # Auto-advance
        self.auto_chk = QCheckBox("Auto-advance")
        self.auto_chk.setChecked(True)
        self.auto_chk.stateChanged.connect(self._auto_advance_changed)
        self.toolbar.addWidget(self.auto_chk)

        # Filter unlabeled toggle
        self.filter_chk = QCheckBox("Unlabeled only")
        self.filter_chk.stateChanged.connect(self._filter_changed)
        self.toolbar.addWidget(self.filter_chk)

        # Batch selection count
        self.toolbar.addSeparator()
        self.sel_lbl = QLabel("Sel: 0")
        self.toolbar.addWidget(self.sel_lbl)

        # Batch brightness (window) controls
        self.toolbar.addSeparator()
        self.toolbar.addWidget(QLabel("Min:"))
        self.vmin_slider = QSlider(Qt.Horizontal)
        self.vmin_slider.setRange(0, 255)
        self.vmin_slider.setFixedWidth(120)
        self.vmin_slider.setValue(0)
        self.vmin_slider.valueChanged.connect(self._update_window_from_sliders)
        self.toolbar.addWidget(self.vmin_slider)
        self.toolbar.addWidget(QLabel("Max:"))
        self.vmax_slider = QSlider(Qt.Horizontal)
        self.vmax_slider.setRange(0, 255)
        self.vmax_slider.setFixedWidth(120)
        self.vmax_slider.setValue(255)
        self.vmax_slider.valueChanged.connect(self._update_window_from_sliders)
        self.toolbar.addWidget(self.vmax_slider)
        # Initially disabled until batch mode
        self._set_window_controls_enabled(False)

    # ----- Core behaviors -----
    def open_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select image directory")
        if not directory:
            return

        try:
            n_rows, n_images = self.store.open_directory(directory)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to open directory:\n{exc}")
            return

        # Hotkeys per directory
        self.hotkeys.for_directory(directory)
        # Sync UI state
        self.column_combo.clear()
        for col in self.store.columns:
            if col != "source-ID":
                self.column_combo.addItem(col)
        self.column_combo.setCurrentText(self.hotkeys.active_column)
        self.auto_chk.setChecked(self.hotkeys.auto_advance)

        # Jump completer
        ids = [self.store.get_row(i).source_id for i in range(self.store.row_count())]
        completer = QCompleter(ids)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.jump_edit.setCompleter(completer)

        self.status.showMessage(f"Loaded {n_images} images; {n_rows} rows in table")
        self.select_row(0 if n_rows > 0 else -1)

    def select_row(self, row: int) -> None:
        if row < 0 or row >= self.store.row_count():
            self.current_row = -1
            if not getattr(self, "batch_mode", False):
                self.viewer.load_image(None)
            return
        self.current_row = row
        rec = self.store.get_row(row)
        img_path = self.store.get_image_path(rec.source_id)
        if not getattr(self, "batch_mode", False):
            self.viewer.load_image(img_path)
        # Update table selection
        sel_model: QItemSelectionModel = self.table_view.selectionModel()
        index = self.table_model.index(row, 0)
        sel_model.select(
            index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
        )
        self.table_view.scrollTo(index)
        # Update status
        self._update_status()

    def _update_status(self) -> None:
        if self.current_row < 0:
            self.status.showMessage("No selection")
            return
        rec = self.store.get_row(self.current_row)
        total = self.store.row_count()
        self.status.showMessage(f"{self.current_row + 1}/{total}  ID={rec.source_id}")

    def _jump_to_entered(self) -> None:
        text = self.jump_edit.text().strip()
        if not text:
            return
        for i in range(self.store.row_count()):
            if self.store.get_row(i).source_id == text:
                self.select_row(i)
                return
        QMessageBox.information(self, "Not found", f"ID '{text}' not found")

    def _table_double_clicked(self, index: QModelIndex) -> None:
        self.select_row(index.row())

    def _active_column_changed(self, col: str) -> None:
        self.hotkeys.active_column = col
        self.hotkeys.save()

    def _auto_advance_changed(self, state: int) -> None:
        self.hotkeys.auto_advance = state == Qt.Checked
        self.hotkeys.save()

    def _filter_changed(self, state: int) -> None:
        self.unlabeled_only = state == Qt.Checked

    def _toggle_table(self) -> None:
        self.dock.setVisible(not self.dock.isVisible())

    def edit_hotkeys(self) -> None:
        dlg = HotkeyEditorDialog(self.hotkeys, self)
        dlg.exec()

    def _add_column(self) -> None:
        # Only when a directory is open
        if not self.store.directory:
            QMessageBox.information(self, "Info", "Open a directory first.")
            return
        name, ok = QInputDialog.getText(self, "Add Column", "New column name:")
        if not ok:
            return
        new_col = name.strip()
        if not new_col:
            return
        if new_col == "source-ID" or new_col in self.store.columns:
            QMessageBox.information(
                self, "Info", "Column already exists or is reserved."
            )
            return
        # Update headers JSON
        headers = list(self.store.columns)
        headers.append(new_col)
        save_headers(self.store.directory, headers)
        # Update store columns and add None values to existing records
        self.store.columns = headers
        for rec in self.store.records:
            rec.values.setdefault(new_col, None)
        # Persist to CSV and refresh model
        self.store.save()
        self.table_model.handle_store_changed()
        # Update active column combo
        self.column_combo.addItem(new_col)
        self.status.showMessage(f"Added column '{new_col}'", 2000)

    def _ensure_column(self, column_name: str) -> None:
        if not self.store.directory:
            return
        if column_name == "source-ID" or column_name in self.store.columns:
            return
        headers = list(self.store.columns)
        headers.append(column_name)
        save_headers(self.store.directory, headers)
        self.store.columns = headers
        for rec in self.store.records:
            rec.values.setdefault(column_name, None)
        self.store.save()
        self.table_model.handle_store_changed()
        self.column_combo.addItem(column_name)

    def _reload_hotkeys(self) -> None:
        # Re-initialize from directory to reapply YAML and JSON
        if not self.store.directory:
            return
        self.hotkeys.for_directory(self.store.directory)
        # Sync UI toggles
        self.column_combo.setCurrentText(self.hotkeys.active_column)
        self.auto_chk.setChecked(self.hotkeys.auto_advance)
        self.status.showMessage("Hotkeys reloaded", 2000)

    def _toggle_batch_mode(self) -> None:
        self.batch_mode = self.batch_toggle_act.isChecked()
        if self.batch_mode:
            self.stack.setCurrentWidget(self.batch_view)
            self._set_window_controls_enabled(True)
        else:
            self.stack.setCurrentWidget(self.viewer)
            self._set_window_controls_enabled(False)
            # When returning to single-image mode, refresh current image
            if self.current_row >= 0:
                rec = self.store.get_row(self.current_row)
                img_path = self.store.get_image_path(rec.source_id)
                self.viewer.load_image(img_path)
        self._update_status()

    def _on_batch_selection_changed(self, count: int) -> None:
        self.sel_lbl.setText(f"Sel: {count}")

    def _load_master_and_key(self) -> None:
        if not self.store.directory:
            QMessageBox.information(self, "Info", "Open a directory first.")
            return
        directory = self.store.directory
        image_path = os.path.join(directory, "master_image.png")
        if not os.path.exists(image_path):
            QMessageBox.information(
                self,
                "Missing master image",
                "master_image.png not found in the directory.",
            )
            return
        label_map = None
        id_list: List[str] = []
        # Preferred: NPZ bundle
        npz_path = os.path.join(directory, "source_location_key.npz")
        try:
            if os.path.exists(npz_path):
                data = np.load(npz_path, allow_pickle=False)
                label_map = data["label_map"]
                id_list = [str(s) for s in data["id_list"]]
            else:
                # Fallback: NPY + TXT
                npy_path = os.path.join(directory, "label_map.npy")
                txt_path = os.path.join(directory, "source_ids.txt")
                if os.path.exists(npy_path) and os.path.exists(txt_path):
                    label_map = np.load(npy_path, allow_pickle=False)
                    with open(txt_path, "r", encoding="utf-8") as f:
                        id_list = [line.strip() for line in f if line.strip()]
        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"Failed to load source location key:\n{exc}"
            )
            return
        if label_map is None or not id_list:
            QMessageBox.information(
                self,
                "Missing files",
                "Could not find source_location_key.npz or label_map.npy + source_ids.txt in the directory.",
            )
            return
        # Validate id_list length covers all indices in label_map
        if label_map.size > 0:
            max_idx = int(np.max(label_map))
            if max_idx >= len(id_list):
                QMessageBox.critical(
                    self,
                    "Error",
                    "label_map contains indices not covered by id_list length.",
                )
                return
        try:
            self.batch_view.load_master(image_path, label_map, id_list)
            self.batch_toggle_act.setChecked(True)
            self._toggle_batch_mode()
            # Reset sliders to default window
            self.vmin_slider.blockSignals(True)
            self.vmax_slider.blockSignals(True)
            self.vmin_slider.setValue(0)
            self.vmax_slider.setValue(255)
            self.vmin_slider.blockSignals(False)
            self.vmax_slider.blockSignals(False)
            self.batch_view.set_window(0, 255)
            self.status.showMessage(
                "Master image and key loaded. Draw polygon (double-click or Enter to finalize, Esc to cancel)."
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"Failed to load master image/key:\n{exc}"
            )

    def save(self) -> None:
        try:
            self.store.save()
            self.status.showMessage("Saved", 2000)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{exc}")

    # ----- Navigation -----
    def go_prev(self) -> None:
        if self.current_row < 0:
            return
        i = self.current_row
        n = self.store.row_count()
        for step in range(1, n + 1):
            cand = (i - step) % n
            if not self.unlabeled_only or self._is_unlabeled(cand):
                self.select_row(cand)
                return

    def go_next(self) -> None:
        if self.current_row < 0:
            return
        i = self.current_row
        n = self.store.row_count()
        for step in range(1, n + 1):
            cand = (i + step) % n
            if not self.unlabeled_only or self._is_unlabeled(cand):
                self.select_row(cand)
                return

    def _is_unlabeled(self, row: int) -> bool:
        rec = self.store.get_row(row)
        for col in self.store.columns:
            if col == "source-ID":
                continue
            if rec.get(col) in (None, ""):
                return True
        return False

    # ----- Key handling -----
    def eventFilter(self, obj, event):  # noqa: N802
        from PySide6.QtCore import QEvent

        if event.type() == QEvent.KeyPress:
            return self._handle_key(event)
        return super().eventFilter(obj, event)

    def _handle_key(self, event) -> bool:
        # Allow batch mode hotkeys even when no current row is selected
        if self.current_row < 0 and not getattr(self, "batch_mode", False):
            return False
        key = event.key()
        mod = event.modifiers()

        # Navigation arrows
        if (
            not getattr(self, "batch_mode", False)
            and key == Qt.Key_Left
            and mod == Qt.NoModifier
        ):
            self.go_prev()
            return True
        if (
            not getattr(self, "batch_mode", False)
            and key == Qt.Key_Right
            and mod == Qt.NoModifier
        ):
            self.go_next()
            return True

        # Help
        if key == Qt.Key_F1:
            self.show_help()
            return True

        # Notes input (Ctrl+N)
        if key == Qt.Key_N and (mod & (Qt.ControlModifier)) == Qt.ControlModifier:
            text, ok = QInputDialog.getText(self, "Notes", "Enter notes:")
            if ok:
                col = "notes"
                if getattr(self, "batch_mode", False):
                    selected_ids = self.batch_view.get_selected_ids()
                    if selected_ids:
                        updated = self.store.set_value_for_ids(selected_ids, col, text)
                        self.status.showMessage(f"Set notes for {updated} selected IDs")
                        return True
                if self.current_row >= 0:
                    self.store.set_value(self.current_row, col, text)
                    idx = self.table_model.index(
                        self.current_row, self.store.columns.index(col)
                    )
                    self.table_model.dataChanged.emit(
                        idx, idx, [Qt.DisplayRole, Qt.EditRole]
                    )
                    self._update_status()
                    return True

        # Dend-ID input ('n' key)
        if key == Qt.Key_N and mod == Qt.NoModifier:
            # Ensure dend-ID column exists
            self._ensure_column("dend-ID")
            text, ok = QInputDialog.getText(self, "Dend-ID", "Enter dend-ID:")
            if ok:
                col = "dend-ID"
                val = text
                if getattr(self, "batch_mode", False):
                    selected_ids = self.batch_view.get_selected_ids()
                    if selected_ids:
                        updated = self.store.set_value_for_ids(selected_ids, col, val)
                        self.status.showMessage(
                            f"Set dend-ID for {updated} selected IDs"
                        )
                        return True
                if self.current_row >= 0:
                    self.store.set_value(self.current_row, col, val)
                    idx = self.table_model.index(
                        self.current_row, self.store.columns.index(col)
                    )
                    self.table_model.dataChanged.emit(
                        idx, idx, [Qt.DisplayRole, Qt.EditRole]
                    )
                    self._update_status()
                    return True
            return True

        # Soma-depth input (Ctrl+D)
        if key == Qt.Key_D and (mod & (Qt.ControlModifier)) == Qt.ControlModifier:
            num, ok = QInputDialog.getDouble(
                self, "Soma depth", "Enter soma-depth:", decimals=3
            )
            if ok:
                col = "soma-depth"
                val = str(num)
                if getattr(self, "batch_mode", False):
                    selected_ids = self.batch_view.get_selected_ids()
                    if selected_ids:
                        updated = self.store.set_value_for_ids(selected_ids, col, val)
                        self.status.showMessage(
                            f"Set soma-depth for {updated} selected IDs"
                        )
                        return True
                if self.current_row >= 0:
                    self.store.set_value(self.current_row, col, val)
                    idx = self.table_model.index(
                        self.current_row, self.store.columns.index(col)
                    )
                    self.table_model.dataChanged.emit(
                        idx, idx, [Qt.DisplayRole, Qt.EditRole]
                    )
                    self._update_status()
                    return True

        # Enter value (Ctrl+Shift+V)
        if key == Qt.Key_V and (mod & (Qt.ControlModifier | Qt.ShiftModifier)) == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            cols = [c for c in self.store.columns if c != "source-ID"]
            if not cols:
                QMessageBox.information(self, "Info", "No editable columns available.")
                return True
            col, ok = QInputDialog.getItem(
                self,
                "Enter value",
                "Choose column:",
                cols,
                0,
                False,
            )
            if not ok or not col:
                return True
            text, ok2 = QInputDialog.getText(
                self, "Enter value", f"Enter value for '{col}':"
            )
            if not ok2:
                return True
            value = text
            if getattr(self, "batch_mode", False):
                selected_ids = self.batch_view.get_selected_ids()
                if selected_ids:
                    updated = self.store.set_value_for_ids(selected_ids, col, value)
                    self.status.showMessage(f"Set {col} for {updated} selected IDs")
                    return True
                else:
                    QMessageBox.information(
                        self,
                        "Info",
                        "No batch selection. Select IDs or exit batch mode.",
                    )
                    return True
            if self.current_row >= 0:
                self.store.set_value(self.current_row, col, value)
                idx = self.table_model.index(
                    self.current_row, self.store.columns.index(col)
                )
                self.table_model.dataChanged.emit(
                    idx, idx, [Qt.DisplayRole, Qt.EditRole]
                )
                self._update_status()
                return True
            else:
                QMessageBox.information(
                    self, "Info", "No row selected. Select a row or use batch mode."
                )
                return True

        # Assign via hotkeys when plain key is pressed
        text = event.text()
        if text and len(text) == 1 and mod in (Qt.NoModifier, Qt.ShiftModifier):
            # 1) Global mapping takes precedence
            global_hit = self.hotkeys.resolve_global(text)
            if global_hit is not None:
                col, value = global_hit
                if getattr(self, "batch_mode", False):
                    selected_ids = self.batch_view.get_selected_ids()
                    if selected_ids:
                        updated = self.store.set_value_for_ids(selected_ids, col, value)
                        self.status.showMessage(
                            f"Assigned {col}='{value}' to {updated} selected IDs"
                        )
                        return True
                if self.current_row >= 0:
                    self.store.set_value(self.current_row, col, value)
                    idx = self.table_model.index(
                        self.current_row, self.store.columns.index(col)
                    )
                    self.table_model.dataChanged.emit(
                        idx, idx, [Qt.DisplayRole, Qt.EditRole]
                    )
                    if self.hotkeys.auto_advance and not getattr(
                        self, "batch_mode", False
                    ):
                        self.go_next()
                    else:
                        self._update_status()
                    return True

            # 2) Fall back to active-column mapping
            col = self.hotkeys.active_column
            value = self.hotkeys.resolve_value(col, text)
            if value is not None:
                if getattr(self, "batch_mode", False):
                    selected_ids = self.batch_view.get_selected_ids()
                    if selected_ids:
                        updated = self.store.set_value_for_ids(selected_ids, col, value)
                        self.status.showMessage(
                            f"Assigned {col}='{value}' to {updated} selected IDs"
                        )
                        return True
                if self.current_row >= 0:
                    self.store.set_value(self.current_row, col, value)
                    idx = self.table_model.index(
                        self.current_row, self.store.columns.index(col)
                    )
                    self.table_model.dataChanged.emit(
                        idx, idx, [Qt.DisplayRole, Qt.EditRole]
                    )
                    if self.hotkeys.auto_advance and not getattr(
                        self, "batch_mode", False
                    ):
                        self.go_next()
                    else:
                        self._update_status()
                    return True

        # Clear value with Backspace or Delete
        if key in (Qt.Key_Backspace, Qt.Key_Delete) and mod == Qt.NoModifier:
            col = self.hotkeys.active_column
            if getattr(self, "batch_mode", False):
                selected_ids = self.batch_view.get_selected_ids()
                if selected_ids:
                    updated = self.store.set_value_for_ids(selected_ids, col, None)
                    self.status.showMessage(f"Cleared {col} for {updated} selected IDs")
                    return True
            if self.current_row >= 0:
                self.store.set_value(self.current_row, col, None)
                idx = self.table_model.index(
                    self.current_row, self.store.columns.index(col)
                )
                self.table_model.dataChanged.emit(
                    idx, idx, [Qt.DisplayRole, Qt.EditRole]
                )
                self._update_status()
                return True

        return False

    def show_help(self) -> None:
        col = self.hotkeys.active_column
        mapping = self.hotkeys.mappings.get(col, {})
        if not mapping:
            text = f"Active column: {col}\nNo hotkeys set yet. Use Edit Hotkeys to add them."
        else:
            lines = [f"Active column: {col}", "", "Keys:"]
            for k, v in sorted(mapping.items()):
                lines.append(f"  {k}\t→  {v}")
            text = "\n".join(lines)
        # Add global mappings if present
        if self.hotkeys.global_mapping:
            g_lines = ["", "Global mappings:"]
            for k, entry in sorted(self.hotkeys.global_mapping.items()):
                g_lines.append(
                    f"  {k}\t→  {entry.get('column','?')} = {entry.get('value','?')}"
                )
            text = text + "\n" + "\n".join(g_lines)
        QMessageBox.information(self, "Shortcut Help", text)

    def _set_window_controls_enabled(self, enabled: bool) -> None:
        for w in [self.vmin_slider, self.vmax_slider]:
            w.setEnabled(enabled)

    def _update_window_from_sliders(self) -> None:
        if not self.batch_mode:
            return
        vmin = self.vmin_slider.value()
        vmax = self.vmax_slider.value()
        self.batch_view.set_window(vmin, vmax)


# -------------------------------
# Batch selection graphics view
# -------------------------------


class BatchGraphicsView(QGraphicsView):
    selection_changed = Signal(int)  # emits count of selected IDs

    def __init__(self) -> None:
        super().__init__()
        self.setRenderHints(
            self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform
        )
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._base_item: Optional[QGraphicsPixmapItem] = None
        self._overlay_item: Optional[QGraphicsPixmapItem] = None

        self._label_map: Optional[np.ndarray] = None  # (H,W) int32, -1 background
        self._id_list: List[str] = []
        self._id_index_to_id: Dict[int, str] = {}
        self._selected_ids: Set[str] = set()

        # Lasso state
        self._lasso_path: Optional[QPainterPath] = None
        self._lassoing: bool = False

        # Keep a reference to overlay backing store
        self._overlay_np: Optional[np.ndarray] = None

        # Brightness/contrast LUT parameters for grayscale images
        self._orig_image: Optional[QImage] = None
        self._vmin: int = 0
        self._vmax: int = 255

    def clear(self) -> None:
        self._scene.clear()
        self._base_item = None
        self._overlay_item = None
        self._label_map = None
        self._id_list = []
        self._id_index_to_id = {}
        self._selected_ids = set()
        self._overlay_np = None
        self._orig_image = None
        self._vmin = 0
        self._vmax = 255
        self.selection_changed.emit(0)

    def load_master(
        self, image_path: str, label_map: np.ndarray, id_list: List[str]
    ) -> None:
        # Validate sizes
        pm = QPixmap(image_path)
        if pm.isNull():
            raise RuntimeError("Failed to load master image")
        h, w = label_map.shape
        if pm.width() != w or pm.height() != h:
            raise ValueError(
                f"Dimension mismatch: image is {pm.width()}x{pm.height()}, label_map is {w}x{h}"
            )

        if label_map.dtype.kind in ("i", "u"):
            pass
        else:
            raise ValueError(
                "label_map must be an integer array with indices into id_list"
            )

        self.clear()

        self._label_map = label_map.astype(np.int32, copy=False)
        self._id_list = [str(s) for s in list(id_list)]
        self._id_index_to_id = {i: s for i, s in enumerate(self._id_list)}

        # Keep original image and start with identity LUT
        img = QImage(image_path)
        if img.isNull():
            # fallback to pixmap->image
            img = pm.toImage()
        self._orig_image = img.convertToFormat(QImage.Format_Grayscale8)
        self._vmin, self._vmax = 0, 255

        self._base_item = QGraphicsPixmapItem(QPixmap.fromImage(self._apply_lut()))
        self._scene.addItem(self._base_item)

        # Overlay item sits above
        self._overlay_item = QGraphicsPixmapItem()
        self._overlay_item.setOpacity(0.5)
        self._overlay_item.setZValue(10)
        self._scene.addItem(self._overlay_item)

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.fitInView(self._base_item, Qt.KeepAspectRatio)

    def set_window(self, vmin: int, vmax: int) -> None:
        # Clamp and ensure vmin < vmax
        vmin = max(0, min(255, int(vmin)))
        vmax = max(0, min(255, int(vmax)))
        if vmax <= vmin:
            vmax = min(255, vmin + 1)
        if self._vmin == vmin and self._vmax == vmax:
            return
        self._vmin, self._vmax = vmin, vmax
        if self._orig_image is not None and self._base_item is not None:
            self._base_item.setPixmap(QPixmap.fromImage(self._apply_lut()))
            self.fitInView(self._base_item, Qt.KeepAspectRatio)

    def _apply_lut(self) -> QImage:
        # Apply linear window [vmin,vmax] to original grayscale image
        if self._orig_image is None:
            return QImage()
        img = self._orig_image
        w = img.width()
        h = img.height()
        bpl = img.bytesPerLine()
        ptr = img.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8, count=bpl * h).reshape((h, bpl))
        gray = arr[:, :w].astype(np.float32)
        # scale
        g = (gray - self._vmin) * (255.0 / max(1.0, (self._vmax - self._vmin)))
        g = np.clip(g, 0, 255).astype(np.uint8)
        # build new QImage
        out = QImage(w, h, QImage.Format_Grayscale8)
        out.fill(0)
        out_ptr = out.bits()
        out_bpl = out.bytesPerLine()
        out_arr = np.frombuffer(out_ptr, dtype=np.uint8, count=out_bpl * h).reshape(
            (h, out_bpl)
        )
        out_arr[:, :w] = g
        return out

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._base_item is not None:
            self.fitInView(self._base_item, Qt.KeepAspectRatio)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._base_item is not None:
            self._lasso_path = QPainterPath(self.mapToScene(event.pos()))
            self._lassoing = True
            # While lassoing, show crosshair and disable hand-drag
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._lassoing and self._lasso_path is not None:
            self._lasso_path.lineTo(self.mapToScene(event.pos()))
            self._update_lasso_overlay_path()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._lassoing and self._lasso_path is not None:
            self._finalize_lasso_selection()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._lassoing:
            # Single-click selection area is tiny; require double-click or Enter to finalize
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if (
            event.key() in (Qt.Key_Return, Qt.Key_Enter)
            and self._lassoing
            and self._lasso_path is not None
        ):
            self._finalize_lasso_selection()
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self._lassoing:
            self._lassoing = False
            self._lasso_path = None
            self._update_overlay()  # remove path overlay
            # Restore hand-drag and cursor
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.unsetCursor()
            event.accept()
            return
        super().keyPressEvent(event)

    def _update_lasso_overlay_path(self) -> None:
        # Draw temporary path overlay
        if self._base_item is None:
            return
        size = self._base_item.pixmap().size()
        w, h = size.width(), size.height()
        img = QImage(w, h, QImage.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(0, 255, 255, 180))
        pen.setWidth(2)
        p.setPen(pen)
        brush = QBrush(QColor(0, 255, 255, 60))
        p.setBrush(brush)
        if self._lasso_path is not None:
            p.drawPath(self._lasso_path)
        p.end()
        if self._overlay_item is not None:
            self._overlay_item.setPixmap(QPixmap.fromImage(img))

    def _finalize_lasso_selection(self) -> None:
        if (
            self._base_item is None
            or self._label_map is None
            or self._lasso_path is None
        ):
            return
        self._lassoing = False

        # Rasterize lasso to mask
        pm = self._base_item.pixmap()
        w, h = pm.width(), pm.height()
        poly_img = QImage(w, h, QImage.Format_Grayscale8)
        poly_img.fill(0)
        p = QPainter(poly_img)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 255))
        p.drawPath(self._lasso_path)
        p.end()

        ptr = poly_img.bits()
        bpl = poly_img.bytesPerLine()
        arr = np.frombuffer(ptr, dtype=np.uint8, count=bpl * h).reshape((h, bpl))
        mask = arr[:, :w] > 0

        ids = self._selected_ids_from_mask(mask)
        self._selected_ids = ids
        self.selection_changed.emit(len(self._selected_ids))
        self._update_overlay()
        self._lasso_path = None
        # Restore hand-drag and cursor
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.unsetCursor()

    def _selected_ids_from_mask(self, mask: np.ndarray) -> Set[str]:
        if self._label_map is None:
            return set()
        indices = np.unique(self._label_map[mask])
        valid = indices[(indices >= 0) & (indices < len(self._id_list))]
        return {self._id_index_to_id[int(i)] for i in valid}

    def _update_overlay(self) -> None:
        # Show selected IDs as a semi-transparent red overlay mask
        if (
            self._base_item is None
            or self._label_map is None
            or self._overlay_item is None
        ):
            return
        pm = self._base_item.pixmap()
        w, h = pm.width(), pm.height()
        if not self._selected_ids:
            # Clear overlay
            img = QImage(w, h, QImage.Format_ARGB32)
            img.fill(QColor(0, 0, 0, 0))
            self._overlay_item.setPixmap(QPixmap.fromImage(img))
            return
        # Build union mask
        id_to_index = {s: i for i, s in enumerate(self._id_list)}
        sel_idx = np.array(
            [id_to_index[s] for s in self._selected_ids if s in id_to_index],
            dtype=np.int32,
        )
        if sel_idx.size == 0:
            img = QImage(w, h, QImage.Format_ARGB32)
            img.fill(QColor(0, 0, 0, 0))
            self._overlay_item.setPixmap(QPixmap.fromImage(img))
            return
        mask = np.isin(self._label_map, sel_idx)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask, 0] = 255
        rgba[mask, 1] = 0
        rgba[mask, 2] = 0
        rgba[mask, 3] = 100
        # Keep ref
        self._overlay_np = rgba
        img = QImage(
            self._overlay_np.data,
            w,
            h,
            self._overlay_np.strides[0],
            QImage.Format_RGBA8888,
        ).copy()  # copy to own memory so numpy can be freed/updated later
        self._overlay_item.setPixmap(QPixmap.fromImage(img))

    def get_selected_ids(self) -> Set[str]:
        return set(self._selected_ids)

    def set_selected_ids(self, ids: Set[str]) -> None:
        self._selected_ids = set(ids)
        self.selection_changed.emit(len(self._selected_ids))
        self._update_overlay()

    def clear_selection(self) -> None:
        self._selected_ids.clear()
        self.selection_changed.emit(0)
        self._update_overlay()


# -------------------------------
# Entry point
# -------------------------------


def _parse_cli_directory() -> Optional[str]:
    # Simple CLI argument: first non-option is directory
    # Usage: python synapse_labeller.py /path/to/images
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue
        if os.path.isdir(arg):
            return os.path.abspath(arg)
    return None


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    # Optional CLI directory open
    cli_dir = _parse_cli_directory()
    if cli_dir:
        try:
            win.store.open_directory(cli_dir)
            win.hotkeys.for_directory(cli_dir)
            win.column_combo.setCurrentText(win.hotkeys.active_column)
            win.auto_chk.setChecked(win.hotkeys.auto_advance)
            ids = [win.store.get_row(i).source_id for i in range(win.store.row_count())]
            completer = QCompleter(ids)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            win.jump_edit.setCompleter(completer)
            win.select_row(0 if win.store.row_count() > 0 else -1)
            win.status.showMessage(
                f"Loaded {len(ids)} images; {len(ids)} rows in table"
            )
        except Exception as exc:
            QMessageBox.critical(
                win, "Error", f"Failed to open directory from CLI:\n{exc}"
            )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
