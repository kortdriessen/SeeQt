from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QFrame,
)

from .canvas import CanvasPanel
from .directory_tree import MaterialsTree, ROLE_IS_MATERIALS, ROLE_PATH
from .notes import NotesView
from .utils import read_text_file
import subprocess


@dataclass
class MaterialsContext:
    root: Path
    materials_dir: Path
    canvas_dir: Optional[Path]
    notes_md: Optional[Path]
    synapse_ids_dir: Optional[Path]
    materials_txt: Optional[str]


class BreadcrumbBar(QWidget):
    """Simple breadcrumb bar showing the active materials directory."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(6)

    def set_path(self, path: Optional[Path], root: Optional[Path]) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if not path or not root:
            label = QLabel("No materials directory selected.", self)
            label.setStyleSheet("color: #777;")
            self._layout.addWidget(label)
            return

        try:
            relative = path.relative_to(root)
            segments = [root.name or str(root)] + list(relative.parts)
        except ValueError:
            segments = [str(path)]

        for segment in segments:
            badge = QLabel(segment, self)
            badge.setStyleSheet(
                "padding: 2px 6px; border: 1px solid #448aff; border-radius: 4px; "
                "background-color: #e3f2fd; color: #0d47a1; font-weight: 500;"
            )
            self._layout.addWidget(badge)
        self._layout.addStretch(1)


class LabNotebookMainWindow(QMainWindow):
    """Main application window orchestrating all views."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lab Notebook")
        self.resize(1400, 900)

        self._root_path: Optional[Path] = None
        self._context: Optional[MaterialsContext] = None

        self.sidebar = MaterialsTree(self)
        self.sidebar.materialsSelected.connect(self._handle_materials_selected)

        self.canvas_panel = CanvasPanel(self)
        self.notes_view = NotesView(self)

        self.breadcrumb = BreadcrumbBar(self)
        self.materials_info = QLabel(self)
        self.materials_info.setWordWrap(True)
        self.materials_info.setStyleSheet("color: #555;")

        center_splitter = QSplitter(Qt.Orientation.Vertical, self)
        center_splitter.addWidget(self.canvas_panel)
        center_splitter.addWidget(self.notes_view)
        center_splitter.setStretchFactor(0, 3)
        center_splitter.setStretchFactor(1, 1)

        info_frame = QFrame(self)
        info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(8, 4, 8, 4)
        info_layout.addWidget(QLabel("materials.txt:", self))
        info_layout.addWidget(self.materials_info, 1)

        self.launch_synapse_button = QPushButton("Launch Synapse Labeller", self)
        self.launch_synapse_button.setToolTip("Launch synapse_labeller.py for each synapse_ids sub-directory.")
        self.launch_synapse_button.clicked.connect(self._launch_synapse_labellers)
        info_layout.addWidget(self.launch_synapse_button, 0, Qt.AlignmentFlag.AlignRight)

        center_widget = QWidget(self)
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)
        center_layout.addWidget(self.breadcrumb)
        center_layout.addWidget(info_frame)
        center_layout.addWidget(center_splitter, 1)

        self.sidebar_frame = QFrame(self)
        sidebar_layout = QVBoxLayout(self.sidebar_frame)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.addWidget(QLabel("Experiments", self))
        sidebar_layout.addWidget(self.sidebar, 1)

        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_splitter.addWidget(self.sidebar_frame)
        main_splitter.addWidget(center_widget)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 4)

        self.setCentralWidget(main_splitter)
        self._main_splitter = main_splitter

        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)

        self._create_actions()
        self._create_menus()
        self._update_materials_info(None)

        QTimer.singleShot(0, self._prompt_for_root_directory)

    def _create_actions(self) -> None:
        self.open_directory_action = QAction("Open &Root Directory...", self)
        self.open_directory_action.setShortcut("Ctrl+O")
        self.open_directory_action.triggered.connect(self._prompt_for_root_directory)

        self.toggle_sidebar_action = QAction("Toggle Sidebar", self, checkable=True)
        self.toggle_sidebar_action.setChecked(True)
        self.toggle_sidebar_action.setShortcut("Ctrl+B")
        self.toggle_sidebar_action.toggled.connect(self._toggle_sidebar)

        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut("Alt+F4")
        self.exit_action.triggered.connect(self.close)

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_directory_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.toggle_sidebar_action)

        toolbar = QToolBar("Main", self)
        toolbar.setMovable(False)
        toolbar.addAction(self.open_directory_action)
        toolbar.addAction(self.toggle_sidebar_action)
        self.addToolBar(toolbar)

    def _prompt_for_root_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select experiment root directory")
        if not directory:
            if not self._root_path:
                QMessageBox.information(
                    self,
                    "Directory required",
                    "Please choose a root directory to work with.",
                )
            return
        self.set_root_directory(Path(directory))

    def set_root_directory(self, root: Path) -> None:
        root = root.expanduser()
        if not root.exists():
            QMessageBox.warning(self, "Invalid directory", f"{root} does not exist.")
            return
        self._root_path = root
        self.sidebar.build_tree(root)
        self._status_bar.showMessage(f"Root: {root}")
        self._select_first_materials()

    def _select_first_materials(self) -> None:
        if self.sidebar.topLevelItemCount() == 0:
            return
        stack = [self.sidebar.topLevelItem(i) for i in range(self.sidebar.topLevelItemCount())]
        while stack:
            item = stack.pop(0)
            if item.data(0, ROLE_IS_MATERIALS):
                self.sidebar.setCurrentItem(item)
                self.sidebar.materialsSelected.emit(Path(str(item.data(0, ROLE_PATH))))
                return
            for idx in range(item.childCount()):
                stack.append(item.child(idx))

    def _handle_materials_selected(self, materials_dir: Path) -> None:
        if not self._root_path:
            return
        context = self._build_context(self._root_path, materials_dir)
        self._context = context
        self.breadcrumb.set_path(materials_dir, self._root_path)
        self._update_materials_info(context.materials_txt)

        if context.canvas_dir and context.canvas_dir.exists():
            images = sorted(context.canvas_dir.glob("*.png"))
            self.canvas_panel.load_images(images)
        else:
            self.canvas_panel.clear()

        self.notes_view.load_file(context.notes_md)

        self._status_bar.showMessage(f"Loaded materials: {materials_dir}")

    def _update_materials_info(self, text: Optional[str]) -> None:
        if not text:
            self.materials_info.setText("No materials.txt summary available.")
        else:
            self.materials_info.setText(text.strip())

    @staticmethod
    def _build_context(root: Path, materials_dir: Path) -> MaterialsContext:
        canvas_dir = materials_dir / "canvas"
        notes_md = materials_dir / "notes.md"
        synapse_ids_dir = materials_dir / "synapse_ids"
        materials_txt_path = materials_dir / "materials.txt"
        materials_txt = read_text_file(materials_txt_path) if materials_txt_path.exists() else None
        return MaterialsContext(
            root=root,
            materials_dir=materials_dir,
            canvas_dir=canvas_dir if canvas_dir.exists() else None,
            notes_md=notes_md if notes_md.exists() else None,
            synapse_ids_dir=synapse_ids_dir if synapse_ids_dir.exists() else None,
            materials_txt=materials_txt,
        )

    def _toggle_sidebar(self, visible: bool) -> None:
        sizes = self._main_splitter.sizes()
        if visible:
            if sizes[0] == 0:
                self._main_splitter.setSizes([250, max(400, sizes[1])])
        else:
            total = sum(sizes) if sum(sizes) else 1
            self._main_splitter.setSizes([0, total])

    def closeEvent(self, event: QCloseEvent) -> None:
        event.accept()

    def _launch_synapse_labellers(self) -> None:
        if not self._context:
            QMessageBox.information(self, "No Materials Selected", "Select a materials directory first.")
            return

        synapse_dir = self._context.synapse_ids_dir
        if not synapse_dir or not synapse_dir.exists():
            QMessageBox.information(self, "No synapse_ids Folder", "This materials directory has no synapse_ids folder.")
            return

        subdirs = sorted([p for p in synapse_dir.iterdir() if p.is_dir()])
        if not subdirs:
            QMessageBox.information(self, "No synapse_ids Subdirectories", "No subdirectories found within synapse_ids.")
            return

        project_root = Path(__file__).resolve().parents[2]
        interpreter = project_root / ".venv" / "Scripts" / "python.exe"
        if not interpreter.exists():
            interpreter = Path(sys.executable)

        script_path = project_root / "synapse_roladex" / "synapse_labeller.py"
        if not script_path.exists():
            QMessageBox.warning(
                self,
                "Script not found",
                f"Could not locate synapse_labeller.py at {script_path}",
            )
            return

        launched = 0
        for subdir in subdirs:
            try:
                subprocess.Popen(
                    [str(interpreter), str(script_path), str(subdir)],
                    cwd=str(project_root),
                    creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
                )
                launched += 1
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Launch failed",
                    f"Failed to launch for {subdir.name}:\n{exc}",
                )

        if launched:
            self._status_bar.showMessage(f"Launched {launched} synapse labeller instance(s).")


def create_app() -> tuple[QApplication, LabNotebookMainWindow]:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app.setApplicationName("Lab Notebook")
        app.setOrganizationName("LabNotebook")
    window = LabNotebookMainWindow()
    return app, window
