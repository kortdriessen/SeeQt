from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QStyle, QTreeWidget, QTreeWidgetItem

from .utils import create_colored_icon, iter_sorted_dirs

ROLE_PATH = Qt.ItemDataRole.UserRole
ROLE_IS_MATERIALS = Qt.ItemDataRole.UserRole + 1


class MaterialsTree(QTreeWidget):
    """Tree widget that exposes materials directories as selectable leaves."""

    materialsSelected = Signal(Path)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._root_path: Optional[Path] = None
        self.setHeaderHidden(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(True)
        self.setExpandsOnDoubleClick(True)
        self.itemSelectionChanged.connect(self._handle_selection)
        self.itemDoubleClicked.connect(self._handle_double_click)

        style = self.style()
        self._folder_icon = style.standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        self._materials_icon = create_colored_icon("#2d9bf0")

    @property
    def root_path(self) -> Optional[Path]:
        return self._root_path

    def build_tree(self, root_path: Path) -> None:
        self.clear()
        root = Path(root_path).expanduser()
        if not root.exists():
            self._root_path = None
            return

        self._root_path = root
        root_item = self._build_item(root, is_root=True)
        if root_item is None:
            root_item = QTreeWidgetItem([root.name or str(root)])
            root_item.setData(0, ROLE_PATH, str(root))
        self.addTopLevelItem(root_item)
        root_item.setExpanded(True)

    def select_materials(self, materials_path: Path) -> None:
        if not self._root_path:
            return
        target = Path(materials_path)
        stack = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if Path(str(item.data(0, ROLE_PATH))) == target:
                self.setCurrentItem(item)
                self.scrollToItem(item)
                return
            for row in range(item.childCount()):
                stack.append(item.child(row))

    def _build_item(self, directory: Path, *, is_root: bool = False) -> Optional[QTreeWidgetItem]:
        is_materials = (directory / "materials.txt").exists()
        item_text = directory.name or str(directory)
        item = QTreeWidgetItem([item_text])
        item.setData(0, ROLE_PATH, str(directory))
        item.setData(0, ROLE_IS_MATERIALS, is_materials)
        item.setIcon(0, self._materials_icon if is_materials else self._folder_icon)

        if is_materials:
            return item

        added_child = False
        for child in iter_sorted_dirs(directory):
            if child.name.startswith(".") or child.name == "__pycache__":
                continue
            child_item = self._build_item(child)
            if child_item is not None:
                item.addChild(child_item)
                added_child = True

        if added_child or is_root:
            return item
        return None

    def _handle_selection(self) -> None:
        item = self.currentItem()
        if not item:
            return
        if item.data(0, ROLE_IS_MATERIALS):
            materials_path = Path(str(item.data(0, ROLE_PATH)))
            self.materialsSelected.emit(materials_path)

    @staticmethod
    def _handle_double_click(item: QTreeWidgetItem, column: int) -> None:
        if not item.data(0, ROLE_IS_MATERIALS):
            item.setExpanded(not item.isExpanded())
