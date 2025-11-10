from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import tifffile
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap, QShortcut, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsPixmapItem


@dataclass
class ChannelSettings:
    visible: bool
    opacity: float
    gamma: float
    black: float
    white: float


class SignalBlocker:
    def __init__(self, widget):
        self.widget = widget

    def __enter__(self):
        self.widget.blockSignals(True)

    def __exit__(self, exc_type, exc, tb):
        self.widget.blockSignals(False)


class ChannelControl(QWidget):
    settingsChanged = Signal()

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.visible_box = QCheckBox(f"{title} visible", self)
        self.visible_box.setChecked(True)
        self.visible_box.toggled.connect(self.settingsChanged.emit)
        layout.addWidget(self.visible_box)

        self.opacity_slider = self._make_slider(0, 100, 100)
        self.opacity_label = QLabel("Opacity: 100%", self)
        self.opacity_slider.valueChanged.connect(self._handle_opacity)
        layout.addWidget(self.opacity_label)
        layout.addWidget(self.opacity_slider)

        self.gamma_slider = self._make_slider(10, 400, 100)
        self.gamma_label = QLabel("Gamma: 1.00", self)
        self.gamma_slider.valueChanged.connect(self._handle_gamma)
        layout.addWidget(self.gamma_label)
        layout.addWidget(self.gamma_slider)

        self.black_slider = self._make_slider(0, 1000, 0)
        self.black_label = QLabel("Black point: 0.00", self)
        self.black_slider.valueChanged.connect(self._handle_black)
        layout.addWidget(self.black_label)
        layout.addWidget(self.black_slider)

        self.white_slider = self._make_slider(0, 1000, 1000)
        self.white_label = QLabel("White point: 1.00", self)
        self.white_slider.valueChanged.connect(self._handle_white)
        layout.addWidget(self.white_label)
        layout.addWidget(self.white_slider)

        layout.addStretch(1)

    @staticmethod
    def _make_slider(minimum: int, maximum: int, value: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        return slider

    def _handle_opacity(self, value: int) -> None:
        self.opacity_label.setText(f"Opacity: {value}%")
        self.settingsChanged.emit()

    def _handle_gamma(self, value: int) -> None:
        gamma = value / 100.0
        self.gamma_label.setText(f"Gamma: {gamma:.2f}")
        self.settingsChanged.emit()

    def _handle_black(self, value: int) -> None:
        black = value / 1000.0
        white = self.white_slider.value() / 1000.0
        if black >= white:
            white = min(1.0, black + 0.01)
            with SignalBlocker(self.white_slider):
                self.white_slider.setValue(int(white * 1000))
            self.white_label.setText(f"White point: {white:.2f}")
        self.black_label.setText(f"Black point: {black:.2f}")
        self.settingsChanged.emit()

    def _handle_white(self, value: int) -> None:
        white = value / 1000.0
        black = self.black_slider.value() / 1000.0
        if white <= black:
            black = max(0.0, white - 0.01)
            with SignalBlocker(self.black_slider):
                self.black_slider.setValue(int(black * 1000))
            self.black_label.setText(f"Black point: {black:.2f}")
        self.white_label.setText(f"White point: {white:.2f}")
        self.settingsChanged.emit()

    def settings(self) -> ChannelSettings:
        return ChannelSettings(
            visible=self.visible_box.isChecked(),
            opacity=self.opacity_slider.value() / 100.0,
            gamma=max(0.1, self.gamma_slider.value() / 100.0),
            black=min(self.black_slider.value() / 1000.0, 0.99),
            white=max(self.white_slider.value() / 1000.0, 0.01),
        )

    def set_visible(self, visible: bool) -> None:
        with SignalBlocker(self.visible_box):
            self.visible_box.setChecked(visible)
        self.settingsChanged.emit()


class StackGraphicsView(QGraphicsView):
    def __init__(self, parent_window: "StackViewerWindow") -> None:
        super().__init__()
        self.window = parent_window
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._zoom_factor = 1.0
        self._user_scaled = False
        self._space_pressed = False
        self._is_panning = False
        self._last_pan_pos = None

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def set_pixmap(self, pixmap: QPixmap, *, auto_fit: bool = False) -> None:
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pixmap)
        else:
            self._pixmap_item.setPixmap(pixmap)
        if auto_fit or not self._user_scaled:
            self.reset_view()

    def reset_view(self) -> None:
        self.resetTransform()
        self._zoom_factor = 1.0
        self._user_scaled = False
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            angle = event.angleDelta().y()
            factor = 1.25 if angle > 0 else 0.8
            self.zoom_by(factor)
            event.accept()
        else:
            angle = event.angleDelta().y()
            if angle != 0:
                step = -1 if angle > 0 else 1
                self.window.step_plane(step)
            event.accept()

    def zoom_by(self, factor: float) -> None:
        if self._pixmap_item is None:
            return
        new_zoom = self._zoom_factor * factor
        new_zoom = max(0.05, min(new_zoom, 50.0))
        factor = new_zoom / self._zoom_factor
        if math.isclose(factor, 1.0, abs_tol=1e-3):
            return
        self._zoom_factor = new_zoom
        self._user_scaled = True
        self.scale(factor, factor)

    def _update_cursor(self) -> None:
        if self._space_pressed:
            cursor = (
                Qt.CursorShape.ClosedHandCursor
                if self._is_panning
                else Qt.CursorShape.OpenHandCursor
            )
            self.viewport().setCursor(cursor)
        else:
            self.viewport().unsetCursor()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._space_pressed:
            self._is_panning = True
            self._last_pan_pos = event.position()
            self._update_cursor()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._is_panning and self._last_pan_pos is not None:
            delta = event.position() - self._last_pan_pos
            self._last_pan_pos = event.position()
            self._pan_by(delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._is_panning:
            self._is_panning = False
            self._update_cursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            self._update_cursor()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            self._is_panning = False
            self._update_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _pan_by(self, delta) -> None:
        h_bar = self.horizontalScrollBar()
        v_bar = self.verticalScrollBar()
        h_bar.setValue(int(h_bar.value() - delta.x()))
        v_bar.setValue(int(v_bar.value() - delta.y()))


_open_windows: list["StackViewerWindow"] = []


class StackViewerWindow(QMainWindow):
    def __init__(self, path: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.path = Path(path)
        self.setWindowTitle(f"TIFF Viewer - {self.path.name}")
        self.resize(1200, 800)

        try:
            self.data, self.channel_count = self._load_stack(self.path)
        except Exception as exc:
            QMessageBox.critical(
                self, "TIFF Load Error", f"Failed to load {self.path}:\n{exc}"
            )
            raise

        self.num_planes = self.data.shape[0]
        self.height = self.data.shape[2]
        self.width = self.data.shape[3]
        self.current_plane = 0
        self._current_frame = None

        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(6, 6, 6, 6)
        central_layout.setSpacing(6)

        self.file_label = QLabel(self.path.name, self)
        self.file_label.setStyleSheet("font-weight: 600; font-size: 16px;")
        central_layout.addWidget(self.file_label)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        central_layout.addWidget(self.splitter, 1)

        # Controls panel
        self.controls_panel = QWidget(self.splitter)
        controls_layout = QVBoxLayout(self.controls_panel)
        controls_layout.setContentsMargins(6, 6, 6, 6)
        controls_layout.setSpacing(10)

        collapse_btn = QToolButton(self.controls_panel)
        collapse_btn.setText("« Hide Controls")
        collapse_btn.clicked.connect(self._toggle_controls)
        controls_layout.addWidget(collapse_btn)

        self.channel_controls: List[ChannelControl] = []
        colors = ["Magenta overlay", "Cyan overlay"]
        for idx in range(self.channel_count):
            box = QGroupBox(colors[idx], self.controls_panel)
            box_layout = QVBoxLayout(box)
            ctrl = ChannelControl(colors[idx], self.controls_panel)
            ctrl.settingsChanged.connect(self.update_display)
            box_layout.addWidget(ctrl)
            controls_layout.addWidget(box)
            self.channel_controls.append(ctrl)

        if self.channel_count == 1:
            info_label = QLabel("Only one channel detected.", self.controls_panel)
            info_label.setStyleSheet("color: #888;")
            controls_layout.addWidget(info_label)

        controls_layout.addStretch(1)

        # View panel
        self.view_container = QWidget(self.splitter)
        view_layout = QVBoxLayout(self.view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(4)

        self.view = StackGraphicsView(self)
        view_layout.addWidget(self.view, 1)

        # Depth panel
        self.depth_panel = QWidget(self.splitter)
        depth_layout = QVBoxLayout(self.depth_panel)
        depth_layout.setContentsMargins(6, 6, 6, 6)
        depth_layout.setSpacing(4)
        self.depth_label = QLabel(self.depth_panel)
        self.depth_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        depth_layout.addWidget(self.depth_label)

        self.depth_slider = QSlider(Qt.Orientation.Vertical, self.depth_panel)
        self.depth_slider.setRange(0, max(0, self.num_planes - 1))
        self.depth_slider.setInvertedAppearance(True)
        self.depth_slider.valueChanged.connect(self._slider_changed)
        depth_layout.addWidget(self.depth_slider, 1)
        self.depth_panel.setMaximumWidth(80)

        self.splitter.addWidget(self.controls_panel)
        self.splitter.addWidget(self.view_container)
        self.splitter.addWidget(self.depth_panel)
        self.splitter.setCollapsible(0, True)
        self.splitter.setCollapsible(2, False)
        self.splitter.setSizes([260, 880, 80])

        self.setCentralWidget(central)

        # Toolbar
        toolbar = QToolBar("Viewer", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        reset_zoom_action = QAction("Reset Zoom", self)
        reset_zoom_action.setShortcut("Ctrl+0")
        reset_zoom_action.triggered.connect(self.view.reset_view)
        toolbar.addAction(reset_zoom_action)

        open_action = QAction("Open TIFF…", self)
        open_action.setShortcut("Ctrl+Shift+O")
        open_action.triggered.connect(self._open_additional_tiffs)
        toolbar.addAction(open_action)

        toggle_controls_action = QAction("Toggle Controls", self)
        toggle_controls_action.setShortcut("Ctrl+L")
        toggle_controls_action.triggered.connect(self._toggle_controls)
        toolbar.addAction(toggle_controls_action)

        self.statusBar().showMessage(
            f"{self.num_planes} planes • {self.height}×{self.width}px"
        )

        # Shortcuts for channel visibility
        shortcut1 = QShortcut(QKeySequence("1"), self)
        shortcut1.activated.connect(lambda: self._toggle_channel(0))
        shortcut2 = QShortcut(QKeySequence("2"), self)
        shortcut2.activated.connect(lambda: self._toggle_channel(1))

        self._controls_visible = True
        self.update_display(initial=True)

    def _load_stack(self, path: Path) -> tuple[np.ndarray, int]:
        with tifffile.TiffFile(str(path)) as tif:
            arr = tif.asarray()
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 3:
            raise ValueError(
                "Unsupported TIFF dimensionality (expected frames × height × width)."
            )
        frames, height, width = arr.shape
        if frames >= 2 and frames % 2 == 0:
            channels = 2
            planes = frames // 2
        else:
            channels = 1
            planes = frames
        arr = arr.reshape(planes, channels, height, width).astype(np.float32)
        norm = np.zeros_like(arr, dtype=np.float32)
        for ch in range(channels):
            channel = arr[:, ch, :, :]
            min_val = float(np.min(channel))
            max_val = float(np.max(channel))
            if math.isclose(min_val, max_val):
                norm[:, ch, :, :] = 0.0
            else:
                norm[:, ch, :, :] = (channel - min_val) / (max_val - min_val)
        return norm, channels

    def _toggle_controls(self) -> None:
        sizes = self.splitter.sizes()
        if self._controls_visible and sizes[0] > 0:
            self.splitter.setSizes([0, sizes[1] + sizes[0], sizes[2]])
            self._controls_visible = False
        else:
            self.splitter.setSizes([260, max(200, sizes[1]), sizes[2]])
            self._controls_visible = True

    def _open_additional_tiffs(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Open additional TIFF stacks",
            str(self.path.parent),
            "TIFF files (*.tif *.tiff)",
        )
        if not files:
            return
        for file in files:
            launch_tiff_viewer(Path(file))

    def _toggle_channel(self, index: int) -> None:
        if index >= len(self.channel_controls):
            return
        control = self.channel_controls[index]
        control.set_visible(not control.visible_box.isChecked())
        self.update_display()

    def _slider_changed(self, value: int) -> None:
        self.set_plane(value)

    def step_plane(self, delta: int) -> None:
        self.set_plane((self.current_plane + delta) % self.num_planes)

    def set_plane(self, index: int) -> None:
        index = max(0, min(index, self.num_planes - 1))
        if index == self.current_plane:
            return
        self.current_plane = index
        with SignalBlocker(self.depth_slider):
            self.depth_slider.setValue(self.num_planes - 1 - index)
        self.update_display()

    def update_display(self, initial: bool = False) -> None:
        frame = np.zeros((self.height, self.width, 3), dtype=np.float32)
        colors = [
            np.array([1.0, 0.0, 1.0], dtype=np.float32),  # magenta
            np.array([0.0, 1.0, 1.0], dtype=np.float32),  # cyan
        ]
        for idx in range(self.channel_count):
            settings = self.channel_controls[idx].settings()
            if not settings.visible:
                continue
            channel_image = self.data[self.current_plane, idx]
            frame += self._compose_channel(channel_image, settings, colors[idx])
        frame = np.clip(frame, 0.0, 1.0)
        self._current_frame = (frame * 255.0).astype(np.uint8)
        qimage = QImage(
            self._current_frame.data,
            self.width,
            self.height,
            self.width * 3,
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage)
        auto_fit = initial and not self.view._user_scaled
        self.view.set_pixmap(pixmap, auto_fit=auto_fit)
        self.depth_label.setText(f"{self.current_plane + 1}/{self.num_planes}")
        self.statusBar().showMessage(
            f"Plane {self.current_plane + 1}/{self.num_planes} • Zoom {self.view._zoom_factor:.2f}×"
        )

    def _compose_channel(
        self,
        plane: np.ndarray,
        settings: ChannelSettings,
        color_multiplier: np.ndarray,
    ) -> np.ndarray:
        if plane is None:
            return np.zeros((self.height, self.width, 3), dtype=np.float32)
        value = np.clip(
            (plane - settings.black) / max(settings.white - settings.black, 1e-5),
            0.0,
            1.0,
        )
        value = np.power(value, settings.gamma)
        rgb = value[..., None] * color_multiplier * settings.opacity
        return rgb


def launch_tiff_viewer(path: Path) -> Optional[StackViewerWindow]:
    """Launch a viewer window for the provided TIFF stack."""
    app = QApplication.instance()
    if app is None:
        raise RuntimeError(
            "QApplication must be running before launching the TIFF viewer."
        )
    try:
        window = StackViewerWindow(Path(path))
    except Exception as exc:
        QMessageBox.critical(None, "TIFF Viewer Error", f"Failed to open {path}:\n{exc}")
        return None
    _open_windows.append(window)
    window.destroyed.connect(lambda _obj=None, w=window: _cleanup_window(w))
    window.show()
    return window


def _cleanup_window(window: StackViewerWindow) -> None:
    try:
        _open_windows.remove(window)
    except ValueError:
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Standalone entry point for manual viewing."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("Usage: python -m tiffviewer.viewer <stack1.tif> [stack2.tif ...]")
        return 1
    app = QApplication.instance() or QApplication(sys.argv)
    for arg in args:
        launch_tiff_viewer(Path(arg))
    if not _open_windows:
        return 1
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
