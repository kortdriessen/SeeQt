from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .configuration import VideoSpec


class ImageDisplay(QLabel):
    """Displays a scaled pixmap while keeping aspect ratio."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._placeholder = "No frame"
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(80, 60)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #111; color: #bbb; border: 1px solid #333;")

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        if self._pixmap is None:
            self.setText(text)

    def set_pixmap(self, pixmap: Optional[QPixmap], message: Optional[str] = None) -> None:
        self._pixmap = pixmap
        if pixmap is None:
            self.setText(message or self._placeholder)
            super().setPixmap(QPixmap())
        else:
            self.setText("")
            self._update_scaled()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._pixmap is not None:
            self._update_scaled()

    def _update_scaled(self) -> None:
        if self._pixmap is None:
            return
        target = self.size()
        scaled = self._pixmap.scaled(
            target, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        super().setPixmap(scaled)


class VideoContainerWidget(QWidget):
    """Widget embedded into each QMdiSubWindow."""

    video_changed = Signal(object)  # emits video_id or None

    def __init__(self, container_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.container_id = container_id
        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(self._on_selection_changed)
        self.display = ImageDisplay()
        self.status_label = QLabel("No video assigned")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.status_label.setStyleSheet("color: #aaa;")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Video:"))
        controls.addWidget(self.selector, 1)
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(controls)
        layout.addWidget(self.display, 1)
        layout.addWidget(self.status_label)

    def populate_videos(self, videos: Iterable[VideoSpec]) -> None:
        current = self.selector.currentData()
        self.selector.blockSignals(True)
        self.selector.clear()
        self.selector.addItem("Unassigned", None)
        for spec in videos:
            label = spec.name or spec.id
            self.selector.addItem(label, spec.id)
        self.selector.blockSignals(False)
        self.set_selected_video(current if current in self._video_ids() else None, emit=False)

    def _video_ids(self) -> set[str]:
        ids: set[str] = set()
        for i in range(self.selector.count()):
            data = self.selector.itemData(i)
            if data is not None:
                ids.add(data)
        return ids

    def set_selected_video(self, video_id: Optional[str], emit: bool = True) -> None:
        target_index = 0
        for i in range(self.selector.count()):
            if self.selector.itemData(i) == video_id:
                target_index = i
                break
        old_state = self.selector.blockSignals(True)
        self.selector.setCurrentIndex(target_index)
        self.selector.blockSignals(old_state)
        if emit:
            self.video_changed.emit(video_id)

    def _on_selection_changed(self, index: int) -> None:
        self.video_changed.emit(self.selector.itemData(index))

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def show_pixmap(self, pixmap: Optional[QPixmap], message: Optional[str] = None) -> None:
        self.display.set_pixmap(pixmap, message=message)


def styled_subwindow_title(container_id: str) -> str:
    return f"Container: {container_id}"
