from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap


def create_colored_icon(color: QColor | str, size: int = 18) -> QIcon:
    qcolor = QColor(color)
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(qcolor)
    painter.setPen(Qt.transparent)
    painter.drawEllipse(1, 1, size - 2, size - 2)
    painter.end()
    return QIcon(pixmap)


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except UnicodeDecodeError:
        return path.read_text(errors="ignore")


def write_text_file(path: Path, content: str) -> None:
    ensure_directory(path)
    path.write_text(content, encoding="utf-8")


def iter_sorted_dirs(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    dirs = [p for p in path.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.name.lower())
    return dirs
