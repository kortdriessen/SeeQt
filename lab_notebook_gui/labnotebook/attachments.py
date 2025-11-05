from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

from PySide6.QtCore import Qt, QSignalBlocker, QUrl
from PySide6.QtGui import QDesktopServices, QPalette, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QSizePolicy,
    QWidget,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


class ImageAttachmentViewer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setBackgroundRole(QPalette.ColorRole.Base)
        self._label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._label.setScaledContents(True)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll, 1)

    def load(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._label.setText("Unable to render image.")
        else:
            self._label.setPixmap(pixmap)
            self._label.adjustSize()


class VideoAttachmentViewer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)

        self.video_widget = QVideoWidget(self)
        self.player.setVideoOutput(self.video_widget)

        self.play_button = QToolButton(self)
        self.play_button.setText("Play")
        self.play_button.clicked.connect(self._toggle_playback)

        self.position_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.player.setPosition)

        self.player.positionChanged.connect(self._update_position)
        self.player.durationChanged.connect(self._update_duration)
        self.player.playbackStateChanged.connect(self._sync_button)

        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.position_slider, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.video_widget, 1)
        layout.addLayout(controls)

    def load(self, path: Path) -> None:
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.player.pause()
        self.play_button.setText("Play")
        self.position_slider.setRange(0, 0)

    def stop(self) -> None:
        self.player.stop()

    def _toggle_playback(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _update_position(self, position: int) -> None:
        with QSignalBlocker(self.position_slider):
            self.position_slider.setValue(position)

    def _update_duration(self, duration: int) -> None:
        self.position_slider.setRange(0, duration)

    def _sync_button(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("Pause")
        else:
            self.play_button.setText("Play")


class UnsupportedAttachmentViewer(QWidget):
    def __init__(self, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path = path
        self._label = QLabel(f"Preview not available for {path.name}.", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._button = QPushButton("Open Externally", self)
        self._button.clicked.connect(self._open_external)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addStretch(1)
        layout.addWidget(self._label, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._button, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(2)

    def _open_external(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._path)))


class AttachmentsView(QWidget):
    """List and preview attachments inside a materials directory."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_dir: Optional[Path] = None
        self._viewer_cache: Dict[Path, QWidget] = {}

        self._list = QListWidget(self)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.itemSelectionChanged.connect(self._handle_selection)

        self._stack = QStackedWidget(self)

        self._placeholder = QLabel("No attachments found.", self)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #777; font-style: italic;")

        list_container = QVBoxLayout()
        list_container.setContentsMargins(0, 0, 0, 0)
        list_container.addWidget(QLabel("Attachments", self))
        list_container.addWidget(self._list, 1)

        outer_layout = QHBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(8)
        outer_layout.addLayout(list_container, 1)
        outer_layout.addWidget(self._stack, 3)
        outer_layout.addWidget(self._placeholder, 3)

        self._update_placeholder(True)

    def load_directory(self, directory: Path | str | None) -> None:
        self._current_dir = Path(directory) if directory else None
        self._viewer_cache.clear()
        self._list.clear()
        self._clear_stack()

        if not self._current_dir or not self._current_dir.exists():
            self._update_placeholder(True, "Attachments directory not found.")
            return

        attachments = sorted(
            [p for p in self._current_dir.iterdir() if p.is_file()],
            key=lambda p: p.name.lower(),
        )

        if not attachments:
            self._update_placeholder(True, "Attachments directory is empty.")
            return

        for path in attachments:
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView)
            elif path.suffix.lower() in VIDEO_EXTENSIONS:
                icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            else:
                icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
            item.setIcon(icon)
            self._list.addItem(item)

        self._list.setCurrentRow(0)
        self._update_placeholder(False)

    def clear(self) -> None:
        self._current_dir = None
        self._viewer_cache.clear()
        self._list.clear()
        self._clear_stack()
        self._update_placeholder(True)

    def _handle_selection(self) -> None:
        item = self._list.currentItem()
        if not item:
            return
        path = Path(str(item.data(Qt.ItemDataRole.UserRole)))
        self._show_attachment(path)

    def _show_attachment(self, path: Path) -> None:
        viewer = self._viewer_cache.get(path)
        if viewer is None:
            viewer = self._create_viewer(path)
            self._viewer_cache[path] = viewer
            self._stack.addWidget(viewer)
        for cached_path, cached_viewer in self._viewer_cache.items():
            if cached_path != path and isinstance(cached_viewer, VideoAttachmentViewer):
                cached_viewer.stop()
        self._stack.setCurrentWidget(viewer)
        self._update_placeholder(False)

    def _create_viewer(self, path: Path) -> QWidget:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            viewer = ImageAttachmentViewer(self)
            viewer.load(path)
            return viewer
        if suffix in VIDEO_EXTENSIONS:
            viewer = VideoAttachmentViewer(self)
            viewer.load(path)
            return viewer
        return UnsupportedAttachmentViewer(path, self)

    def _update_placeholder(self, visible: bool, message: Optional[str] = None) -> None:
        if message:
            self._placeholder.setText(message)
        self._placeholder.setVisible(visible)
        self._stack.setVisible(not visible)
        self._list.setVisible(not visible)

    def _clear_stack(self) -> None:
        while self._stack.count():
            widget = self._stack.widget(0)
            if isinstance(widget, VideoAttachmentViewer):
                widget.stop()
            self._stack.removeWidget(widget)
            widget.deleteLater()
