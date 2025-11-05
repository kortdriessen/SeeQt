from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal, QObject, QSignalBlocker
from PySide6.QtGui import (
    QAction,
    QColor,
    QCursor,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .utils import create_colored_icon, ensure_directory


@dataclass
class Stroke:
    color: str
    width: float
    points: List[Tuple[float, float]]


class CanvasDocument(QObject):
    """Container for stroke annotations tied to an image on disk."""

    strokesChanged = Signal()

    def __init__(self, image_path: Path) -> None:
        super().__init__()
        self.image_path = Path(image_path)
        self.annotation_path = self._build_annotation_path(self.image_path)
        self._base_image = QImage(str(self.image_path))
        if self._base_image.isNull():
            raise ValueError(f"Could not load image at {self.image_path}")
        self._strokes: list[Stroke] = []
        self.load()

    @staticmethod
    def _build_annotation_path(image_path: Path) -> Path:
        suffix = image_path.suffix
        return image_path.with_suffix(f"{suffix}.annotations.json")

    @property
    def base_image(self) -> QImage:
        return self._base_image

    @property
    def size(self) -> tuple[int, int]:
        return self._base_image.width(), self._base_image.height()

    @property
    def strokes(self) -> Sequence[Stroke]:
        return tuple(self._strokes)

    def load(self) -> None:
        if not self.annotation_path.exists():
            self._strokes = []
            return
        try:
            data = json.loads(self.annotation_path.read_text(encoding="utf-8"))
            strokes = []
            for entry in data.get("strokes", []):
                points = [(float(x), float(y)) for x, y in entry.get("points", [])]
                color = entry.get("color", "#ff0000")
                width = float(entry.get("width", 3.0))
                strokes.append(Stroke(color=color, width=width, points=points))
            self._strokes = strokes
        except Exception:
            self._strokes = []

    def save(self) -> None:
        ensure_directory(self.annotation_path)
        payload = {"strokes": [asdict(stroke) for stroke in self._strokes]}
        self.annotation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add_stroke(self, stroke: Stroke) -> None:
        self._strokes.append(stroke)
        self.save()
        self.strokesChanged.emit()

    def undo_last_stroke(self) -> None:
        if not self._strokes:
            return
        self._strokes.pop()
        self.save()
        self.strokesChanged.emit()

    def clear(self) -> None:
        if not self._strokes:
            return
        self._strokes.clear()
        self.save()
        self.strokesChanged.emit()


class CanvasView(QGraphicsView):
    """Graphics view capable of freehand drawing with undo support."""

    activated = Signal()

    def __init__(self, document: CanvasDocument, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.document = document
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._base_item: Optional[QGraphicsPixmapItem] = None
        self._overlay_item: Optional[QGraphicsPixmapItem] = None
        self._overlay_pixmap = QPixmap()

        self._is_drawing = False
        self._current_points: list[tuple[float, float]] = []
        self._preview_points: list[tuple[float, float]] = []
        self._pen_color = QColor("#ff1744")
        self._pen_width = 4.0
        self._zoom_factor = 1.0

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)

        self._init_scene()
        self.document.strokesChanged.connect(self._update_overlay)

    def set_pen(self, color: Optional[QColor] = None, width: Optional[float] = None) -> None:
        if color:
            self._pen_color = QColor(color)
        if width is not None:
            self._pen_width = max(0.5, float(width))

    def reset_view(self) -> None:
        self.resetTransform()
        self._zoom_factor = 1.0
        if self._base_item:
            self.fitInView(self._base_item, Qt.AspectRatioMode.KeepAspectRatio)

    def undo_last_stroke(self) -> None:
        self.document.undo_last_stroke()

    def wheelEvent(self, event) -> None:
        if not self._base_item:
            return
        angle = event.angleDelta().y()
        if angle == 0:
            return
        factor = 1.25 if angle > 0 else 0.8
        new_zoom = self._zoom_factor * factor
        new_zoom = max(0.2, min(new_zoom, 50.0))
        factor = new_zoom / self._zoom_factor
        if math.isclose(factor, 1.0, abs_tol=1e-3):
            return
        self._zoom_factor = new_zoom
        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit()
            mapped = self._map_to_image(event.position())
            if mapped is not None:
                self._is_drawing = True
                self._current_points = [mapped]
                self._preview_points = [mapped]
                event.accept()
                return
        if event.button() == Qt.MouseButton.RightButton:
            self.activated.emit()
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._is_drawing:
            mapped = self._map_to_image(event.position())
            if mapped is not None:
                last = self._current_points[-1]
                if (last[0] - mapped[0]) ** 2 + (last[1] - mapped[1]) ** 2 > 0.5:
                    self._current_points.append(mapped)
                    self._preview_points = list(self._current_points)
                    self._update_overlay(preview_points=self._preview_points)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._is_drawing:
            self._is_drawing = False
            if len(self._current_points) >= 1:
                stroke = Stroke(
                    color=self._pen_color.name(),
                    width=self._pen_width,
                    points=list(self._current_points),
                )
                self.document.add_stroke(stroke)
            self._current_points.clear()
            self._preview_points.clear()
            self._update_overlay()
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            QApplication.restoreOverrideCursor()
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)

    def _init_scene(self) -> None:
        base_pixmap = QPixmap.fromImage(self.document.base_image)
        self._base_item = self._scene.addPixmap(base_pixmap)
        self._base_item.setZValue(0)
        self._overlay_pixmap = QPixmap(base_pixmap.size())
        self._overlay_pixmap.fill(Qt.transparent)
        self._overlay_item = self._scene.addPixmap(self._overlay_pixmap)
        self._overlay_item.setZValue(1)
        self._scene.setSceneRect(QRectF(base_pixmap.rect()))
        self.reset_view()
        self._update_overlay()

    def _map_to_image(self, view_point) -> Optional[tuple[float, float]]:
        scene_point = self.mapToScene(int(view_point.x()), int(view_point.y()))
        if not (0 <= scene_point.x() < self.document.size[0] and 0 <= scene_point.y() < self.document.size[1]):
            return None
        x = max(0.0, min(scene_point.x(), self.document.size[0] - 1))
        y = max(0.0, min(scene_point.y(), self.document.size[1] - 1))
        return (x, y)

    def _update_overlay(self, preview_points: Optional[Sequence[tuple[float, float]]] = None) -> None:
        if not self._overlay_item:
            return
        overlay_image = QImage(
            self.document.size[0],
            self.document.size[1],
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        overlay_image.fill(Qt.transparent)
        painter = QPainter(overlay_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for stroke in self.document.strokes:
            self._draw_stroke(painter, stroke)
        if preview_points:
            preview = Stroke(color=self._pen_color.name(), width=self._pen_width, points=list(preview_points))
            self._draw_stroke(painter, preview)
        painter.end()
        self._overlay_pixmap = QPixmap.fromImage(overlay_image)
        self._overlay_item.setPixmap(self._overlay_pixmap)

    @staticmethod
    def _draw_stroke(painter: QPainter, stroke: Stroke) -> None:
        if not stroke.points:
            return
        color = QColor(stroke.color)
        pen = QPen(color, stroke.width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        if len(stroke.points) == 1:
            point = stroke.points[0]
            radius = stroke.width / 2.0
            painter.setBrush(color)
            painter.drawEllipse(QPointF(point[0], point[1]), radius, radius)
            painter.setBrush(Qt.NoBrush)
            return
        path = QPainterPath(QPointF(*stroke.points[0]))
        for x, y in stroke.points[1:]:
            path.lineTo(x, y)
        painter.drawPath(path)


class CanvasToolControls(QWidget):
    """Toolbar that exposes pen color, width, undo and fullscreen actions."""

    colorChanged = Signal(QColor)
    widthChanged = Signal(float)
    undoRequested = Signal()
    fullscreenRequested = Signal()

    DEFAULT_COLORS: Sequence[tuple[str, str]] = (
        ("Red", "#ff1744"),
        ("Blue", "#2979ff"),
        ("Yellow", "#ffca28"),
        ("Green", "#00c853"),
        ("Pink", "#ff4081"),
        ("Orange", "#ff9100"),
    )

    def __init__(
        self,
        *,
        colors: Optional[Sequence[tuple[str, str]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._colors = list(colors or self.DEFAULT_COLORS)
        self._current_color = QColor(self._colors[0][1])
        self._current_width = 4.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._color_group = QButtonGroup(self)
        self._color_group.setExclusive(True)

        for index, (label, color_hex) in enumerate(self._colors):
            button = QToolButton(self)
            button.setCheckable(True)
            button.setIcon(create_colored_icon(color_hex))
            button.setToolTip(f"Pen color: {label}")
            button.setAutoRaise(True)
            self._color_group.addButton(button, index)
            layout.addWidget(button)

        if self._color_group.buttons():
            self._color_group.buttons()[0].setChecked(True)
        self._color_group.idClicked.connect(self._internal_color_changed)

        self._width_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._width_slider.setRange(1, 40)
        self._width_slider.setValue(int(self._current_width))
        self._width_slider.setFixedWidth(150)
        self._width_slider.valueChanged.connect(self._internal_width_changed)

        self._width_label = QLabel("Width: 4 px", self)
        self._width_label.setMinimumWidth(80)

        layout.addSpacing(12)
        layout.addWidget(self._width_label)
        layout.addWidget(self._width_slider)

        layout.addItem(QSpacerItem(12, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        self.undo_button = QToolButton(self)
        self.undo_button.setText("Undo")
        self.undo_button.setToolTip("Undo last stroke (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undoRequested.emit)
        layout.addWidget(self.undo_button)

        self.fullscreen_button = QToolButton(self)
        self.fullscreen_button.setText("Full Screen")
        self.fullscreen_button.setToolTip("Toggle full screen view (F11)")
        self.fullscreen_button.clicked.connect(self.fullscreenRequested.emit)
        layout.addWidget(self.fullscreen_button)

    def sync_color(self, color: QColor) -> None:
        self._current_color = QColor(color)
        matched = False
        with QSignalBlocker(self._color_group):
            for button in self._color_group.buttons():
                idx = self._color_group.id(button)
                if QColor(self._colors[idx][1]).name() == self._current_color.name():
                    button.setChecked(True)
                    matched = True
                else:
                    button.setChecked(False)
            if not matched and self._color_group.buttons():
                self._color_group.buttons()[0].setChecked(True)
                self._current_color = QColor(self._colors[0][1])

    def sync_width(self, width: float) -> None:
        self._current_width = float(width)
        with QSignalBlocker(self._width_slider):
            self._width_slider.setValue(int(round(self._current_width)))
        self._update_width_label()

    def _internal_color_changed(self, button_id: int) -> None:
        label, color_hex = self._colors[button_id]
        self._current_color = QColor(color_hex)
        self.colorChanged.emit(QColor(color_hex))

    def _internal_width_changed(self, value: int) -> None:
        self._current_width = float(value)
        self._update_width_label()
        self.widthChanged.emit(self._current_width)

    def _update_width_label(self) -> None:
        self._width_label.setText(f"Width: {int(self._current_width)} px")


class CanvasViewWidget(QFrame):
    """Wrapper that hosts a CanvasView and exposes utility helpers."""

    def __init__(self, document: CanvasDocument, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.document = document
        self.view = CanvasView(document, self)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(320)
        self.setMinimumHeight(260)

        self._base_style = (
            "QFrame#canvasViewFrame {"
            " border: 2px solid %s;"
            " border-radius: 8px;"
            " background-color: #121212;"
            "}"
        )
        self.setObjectName("canvasViewFrame")

        self._title_label = QLabel(document.image_path.name, self)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("color: #d0d0d0; font-size: 12px; padding: 2px 0;")
        self.set_active(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self.view, 1)
        layout.addWidget(self._title_label, 0)

    def set_pen(self, color: QColor, width: float) -> None:
        self.view.set_pen(color, width)

    def undo_last_stroke(self) -> None:
        self.view.undo_last_stroke()

    def set_active(self, active: bool) -> None:
        border = "#2d9bf0" if active else "#3a3a3a"
        self.setStyleSheet(self._base_style % border)
        if active:
            self._title_label.setStyleSheet("color: #ffffff; font-weight: 600; padding: 2px 0;")
        else:
            self._title_label.setStyleSheet("color: #d0d0d0; font-size: 12px; padding: 2px 0;")


class CanvasFullScreenWindow(QMainWindow):
    """Dedicated window for drawing on a canvas in full screen."""

    closed = Signal()
    colorChanged = Signal(QColor)
    widthChanged = Signal(float)

    def __init__(
        self,
        document: CanvasDocument,
        *,
        colors: Optional[Sequence[tuple[str, str]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(document.image_path.name)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setWindowState(Qt.WindowState.WindowFullScreen)

        self.view = CanvasView(document, self)
        self.controls = CanvasToolControls(colors=colors)
        self.controls.fullscreen_button.setText("Exit Full Screen")

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self.controls)
        layout.addWidget(self.view, 1)
        self.setCentralWidget(central)

        self.controls.undoRequested.connect(self.view.undo_last_stroke)
        self.controls.fullscreenRequested.connect(self.close)
        self.controls.colorChanged.connect(self._forward_color_change)
        self.controls.widthChanged.connect(self._forward_width_change)

        close_action = QAction("Exit Full Screen", self)
        close_action.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        close_action.triggered.connect(self.close)
        self.addAction(close_action)

    def set_pen_state(self, color: QColor, width: float) -> None:
        self.view.set_pen(color, width)
        self.controls.sync_color(color)
        self.controls.sync_width(width)

    def _forward_color_change(self, color: QColor) -> None:
        self.view.set_pen(color=color)
        self.colorChanged.emit(color)

    def _forward_width_change(self, width: float) -> None:
        self.view.set_pen(width=width)
        self.widthChanged.emit(width)

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)


class CanvasPanel(QWidget):
    """Composite widget hosting canvas views, controls and full-screen logic."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._documents: dict[Path, CanvasDocument] = {}
        self._current_color = QColor(CanvasToolControls.DEFAULT_COLORS[0][1])
        self._current_width = 4.0
        self._active_fullscreen: Optional[CanvasFullScreenWindow] = None

        self._view_widgets: list[CanvasViewWidget] = []
        self._active_widget: Optional[CanvasViewWidget] = None

        self.controls = CanvasToolControls()
        self.controls.colorChanged.connect(self._handle_color_change)
        self.controls.widthChanged.connect(self._handle_width_change)
        self.controls.undoRequested.connect(self._handle_undo)
        self.controls.fullscreenRequested.connect(self._handle_fullscreen_request)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._canvas_holder = QWidget(self.scroll_area)
        self._canvas_layout = QHBoxLayout(self._canvas_holder)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas_layout.setSpacing(12)
        self.scroll_area.setWidget(self._canvas_holder)

        self.placeholder = QLabel("Select a materials directory to load canvases.", self)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #666; font-style: italic;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.controls)
        layout.addWidget(self.scroll_area, 1)
        layout.addWidget(self.placeholder, 1)

        self._update_placeholder_visibility()

        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._handle_undo)

    def load_images(self, image_paths: Sequence[Path]) -> None:
        self._clear_views()
        if not image_paths:
            self._update_placeholder_visibility()
            return

        for img_path in image_paths:
            img_path = Path(img_path)
            try:
                document = self._documents.setdefault(img_path, CanvasDocument(img_path))
            except ValueError as exc:
                QMessageBox.warning(self, "Canvas Load Error", str(exc))
                continue

            document.load()

            widget = CanvasViewWidget(document, self)
            widget.set_pen(self._current_color, self._current_width)
            widget.view.activated.connect(lambda w=widget: self._set_active_widget(w))

            self._canvas_layout.addWidget(widget, 1)
            self._view_widgets.append(widget)

        if self._view_widgets:
            self._set_active_widget(self._view_widgets[0])

        self._update_placeholder_visibility()

    def clear(self) -> None:
        self._clear_views()
        self._update_placeholder_visibility()

    def current_view(self) -> Optional[CanvasViewWidget]:
        return self._active_widget

    def _handle_color_change(self, color: QColor) -> None:
        self._current_color = QColor(color)
        view = self.current_view()
        if view:
            view.set_pen(self._current_color, self._current_width)

    def _handle_width_change(self, width: float) -> None:
        self._current_width = float(width)
        view = self.current_view()
        if view:
            view.set_pen(self._current_color, self._current_width)

    def _handle_undo(self) -> None:
        view = self.current_view()
        if view:
            view.undo_last_stroke()

    def _handle_fullscreen_request(self) -> None:
        if self._active_fullscreen:
            self._active_fullscreen.raise_()
            return
        view = self.current_view()
        if not view:
            return
        document = view.document
        self._active_fullscreen = CanvasFullScreenWindow(
            document,
            colors=CanvasToolControls.DEFAULT_COLORS,
            parent=self.window(),
        )
        self._active_fullscreen.set_pen_state(self._current_color, self._current_width)
        self._active_fullscreen.colorChanged.connect(self._sync_color_from_fullscreen)
        self._active_fullscreen.widthChanged.connect(self._sync_width_from_fullscreen)
        self._active_fullscreen.closed.connect(self._handle_fullscreen_closed)
        self._active_fullscreen.showFullScreen()

    def _handle_fullscreen_closed(self) -> None:
        self._active_fullscreen = None

    def _sync_color_from_fullscreen(self, color: QColor) -> None:
        self._current_color = QColor(color)
        self.controls.sync_color(color)
        view = self.current_view()
        if view:
            view.set_pen(self._current_color, self._current_width)

    def _sync_width_from_fullscreen(self, width: float) -> None:
        self._current_width = float(width)
        self.controls.sync_width(width)
        view = self.current_view()
        if view:
            view.set_pen(self._current_color, self._current_width)

    def _set_active_widget(self, widget: CanvasViewWidget) -> None:
        if widget not in self._view_widgets:
            return
        if self._active_widget is widget:
            return
        if self._active_widget:
            self._active_widget.set_active(False)
        self._active_widget = widget
        self._active_widget.set_active(True)
        self._active_widget.set_pen(self._current_color, self._current_width)
        self.scroll_area.ensureWidgetVisible(widget, 20, 20)

    def _clear_views(self) -> None:
        while self._canvas_layout.count():
            item = self._canvas_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._view_widgets.clear()
        self._active_widget = None

    def _update_placeholder_visibility(self) -> None:
        has_views = bool(self._view_widgets)
        self.scroll_area.setVisible(has_views)
        self.placeholder.setVisible(not has_views)
