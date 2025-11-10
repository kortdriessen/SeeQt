from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal, QObject, QSignalBlocker, QTimer, QEvent
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
    QTabletEvent,
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
    eraser: bool = False


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
                eraser = bool(entry.get("eraser", False))
                strokes.append(Stroke(color=color, width=width, points=points, eraser=eraser))
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
        self._overlay_image: Optional[QImage] = None
        self._tablet_override = False
        self._tablet_prev_eraser = False

        self._is_drawing = False
        self._current_points: list[tuple[float, float]] = []
        self._preview_points: list[tuple[float, float]] = []
        self._pen_color = QColor("#ff1744")
        self._pen_width = 12.0
        self._zoom_factor = 1.0
        self._user_scaled = False
        self._space_pressed = False
        self._is_panning = False
        self._last_pan_pos = QPointF()
        self._is_eraser = False

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TabletTracking, True)

        self._init_scene()
        self.document.strokesChanged.connect(self._handle_document_changed)

        self._zoom_in_shortcut = QShortcut(QKeySequence("Ctrl+I"), self)
        self._zoom_in_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._zoom_in_shortcut.activated.connect(lambda: self.zoom_by(1.25))

        self._zoom_out_shortcut = QShortcut(QKeySequence("Ctrl+T"), self)
        self._zoom_out_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._zoom_out_shortcut.activated.connect(lambda: self.zoom_by(0.8))

    def set_pen(self, color: Optional[QColor] = None, width: Optional[float] = None) -> None:
        if color:
            self._pen_color = QColor(color)
        if width is not None:
            self._pen_width = max(0.5, float(width))
        self._is_eraser = False
        self._tablet_override = False
        self._update_cursor()

    def set_eraser(self, width: Optional[float] = None) -> None:
        self._is_eraser = True
        if width is not None:
            self._pen_width = max(0.5, float(width))
        self._tablet_override = False
        self._update_cursor()

    def reset_view(self) -> None:
        self.resetTransform()
        self._zoom_factor = 1.0
        self._user_scaled = False
        if self._base_item:
            self.fitInView(self._base_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._update_cursor()

    def undo_last_stroke(self) -> None:
        self.document.undo_last_stroke()

    def _update_cursor(self) -> None:
        if self._space_pressed:
            cursor = Qt.CursorShape.ClosedHandCursor if self._is_panning else Qt.CursorShape.OpenHandCursor
            self.viewport().setCursor(cursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def zoom_by(self, factor: float) -> bool:
        if not self._base_item:
            return False
        new_zoom = self._zoom_factor * factor
        new_zoom = max(0.2, min(new_zoom, 50.0))
        factor = new_zoom / self._zoom_factor
        if math.isclose(factor, 1.0, abs_tol=1e-3):
            return False
        self._zoom_factor = new_zoom
        self._user_scaled = True
        self.scale(factor, factor)
        return True

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._user_scaled:
            self.reset_view()

    def wheelEvent(self, event) -> None:
        if not self._base_item:
            return
        angle = event.angleDelta().y()
        if angle == 0:
            return
        factor = 1.25 if angle > 0 else 0.8
        if self.zoom_by(factor):
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._space_pressed:
            self.activated.emit()
            self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
            self._is_panning = True
            self._last_pan_pos = event.position()
            self._update_cursor()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit()
            self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
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
        if self._is_panning:
            delta = event.position() - self._last_pan_pos
            self._last_pan_pos = event.position()
            self._pan_by(delta)
            event.accept()
            return
        if self._is_drawing:
            mapped = self._map_to_image(event.position())
            if mapped is not None:
                last = self._current_points[-1]
                if (last[0] - mapped[0]) ** 2 + (last[1] - mapped[1]) ** 2 > 0.5:
                    self._current_points.append(mapped)
                    self._preview_points = list(self._current_points)
                    self._refresh_overlay(preview_points=self._preview_points)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._is_panning:
            self._is_panning = False
            self._update_cursor()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._is_drawing:
            self._is_drawing = False
            if len(self._current_points) >= 1:
                stroke = Stroke(
                    color=self._pen_color.name(),
                    width=self._pen_width,
                    points=list(self._current_points),
                    eraser=self._is_eraser,
                )
                self.document.add_stroke(stroke)
            self._current_points.clear()
            self._preview_points.clear()
            self._refresh_overlay()
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            QApplication.restoreOverrideCursor()
        super().mouseReleaseEvent(event)
        self._update_cursor()

    def tabletEvent(self, event: QTabletEvent) -> None:
        event_type = event.type()
        if event_type == QEvent.Type.TabletPress:
            self.activated.emit()
            self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
            if self._space_pressed:
                self._is_panning = True
                self._last_pan_pos = event.position()
                self._update_cursor()
                event.accept()
                return
            mapped = self._map_to_image(event.position())
            if mapped is not None:
                self._is_drawing = True
                if event.pointerType() == QTabletEvent.PointerType.Eraser:
                    self._tablet_override = True
                    self._tablet_prev_eraser = self._is_eraser
                    self._is_eraser = True
                else:
                    self._tablet_override = False
                self._current_points = [mapped]
                self._preview_points = [mapped]
            event.accept()
            return
        if event_type == QEvent.Type.TabletMove:
            if self._is_panning:
                delta = event.position() - self._last_pan_pos
                self._last_pan_pos = event.position()
                self._pan_by(delta)
                event.accept()
                return
            if self._is_drawing:
                mapped = self._map_to_image(event.position())
                if mapped is not None:
                    last = self._current_points[-1]
                    if (last[0] - mapped[0]) ** 2 + (last[1] - mapped[1]) ** 2 > 0.5:
                        self._current_points.append(mapped)
                        self._preview_points = list(self._current_points)
                        self._refresh_overlay(preview_points=self._preview_points)
            event.accept()
            return
        if event_type == QEvent.Type.TabletRelease:
            if self._is_panning:
                self._is_panning = False
                self._update_cursor()
                event.accept()
                return
            if self._is_drawing:
                self._is_drawing = False
                if len(self._current_points) >= 1:
                    stroke = Stroke(
                        color=self._pen_color.name(),
                        width=self._pen_width,
                        points=list(self._current_points),
                        eraser=self._is_eraser,
                    )
                    self.document.add_stroke(stroke)
                self._current_points.clear()
                self._preview_points.clear()
                self._refresh_overlay()
            if self._tablet_override:
                self._is_eraser = self._tablet_prev_eraser
                self._tablet_override = False
            self._update_cursor()
            event.accept()
            return
        super().tabletEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            if not self._is_drawing:
                self._update_cursor()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            if self._is_panning:
                self._is_panning = False
            if not self._is_drawing:
                self._update_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _init_scene(self) -> None:
        base_pixmap = QPixmap.fromImage(self.document.base_image)
        self._base_item = self._scene.addPixmap(base_pixmap)
        self._base_item.setZValue(0)
        self._overlay_image = QImage(
            base_pixmap.size().width(),
            base_pixmap.size().height(),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        self._overlay_image.fill(Qt.transparent)
        self._overlay_item = self._scene.addPixmap(QPixmap.fromImage(self._overlay_image))
        self._overlay_item.setZValue(1)
        self._scene.setSceneRect(QRectF(base_pixmap.rect()))
        self.reset_view()
        self._handle_document_changed()

    def _map_to_image(self, view_point) -> Optional[tuple[float, float]]:
        scene_point = self.mapToScene(int(view_point.x()), int(view_point.y()))
        if not (0 <= scene_point.x() < self.document.size[0] and 0 <= scene_point.y() < self.document.size[1]):
            return None
        x = max(0.0, min(scene_point.x(), self.document.size[0] - 1))
        y = max(0.0, min(scene_point.y(), self.document.size[1] - 1))
        return (x, y)

    def _handle_document_changed(self) -> None:
        if self._overlay_image is None:
            return
        self._overlay_image.fill(Qt.transparent)
        painter = QPainter(self._overlay_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for stroke in self.document.strokes:
            self._draw_stroke(painter, stroke)
        painter.end()
        self._refresh_overlay()

    def _refresh_overlay(self, preview_points: Optional[Sequence[tuple[float, float]]] = None) -> None:
        if self._overlay_image is None or not self._overlay_item:
            return
        if preview_points:
            temp = QImage(self._overlay_image)
            painter = QPainter(temp)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            preview = Stroke(
                color=self._pen_color.name(),
                width=self._pen_width,
                points=list(preview_points),
                eraser=self._is_eraser,
            )
            self._draw_stroke(painter, preview)
            painter.end()
            self._overlay_item.setPixmap(QPixmap.fromImage(temp))
        else:
            self._overlay_item.setPixmap(QPixmap.fromImage(self._overlay_image))

    def _pan_by(self, delta: QPointF) -> None:
        h_bar = self.horizontalScrollBar()
        v_bar = self.verticalScrollBar()
        h_bar.setValue(int(h_bar.value() - delta.x()))
        v_bar.setValue(int(v_bar.value() - delta.y()))

    @staticmethod
    def _draw_stroke(painter: QPainter, stroke: Stroke) -> None:
        if not stroke.points:
            return
        if stroke.eraser:
            painter.save()
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            pen = QPen(Qt.GlobalColor.transparent, stroke.width)
        else:
            color = QColor(stroke.color)
            pen = QPen(color, stroke.width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        if len(stroke.points) == 1:
            point = stroke.points[0]
            radius = stroke.width / 2.0
            if stroke.eraser:
                painter.drawEllipse(QPointF(point[0], point[1]), radius, radius)
                painter.restore()
            else:
                painter.setBrush(color)
                painter.drawEllipse(QPointF(point[0], point[1]), radius, radius)
                painter.setBrush(Qt.NoBrush)
            return
        path = QPainterPath(QPointF(*stroke.points[0]))
        for x, y in stroke.points[1:]:
            path.lineTo(x, y)
        painter.drawPath(path)
        if stroke.eraser:
            painter.restore()


class CanvasToolControls(QWidget):
    """Toolbar that exposes pen color, width, undo and fullscreen actions."""

    colorChanged = Signal(QColor)
    widthChanged = Signal(float)
    undoRequested = Signal()
    fullscreenRequested = Signal()
    eraserToggled = Signal(bool)

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
        self._current_width = 12.0
        self._last_color_index = 0

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

        self._width_label = QLabel("Width: 12 px", self)
        self._width_label.setMinimumWidth(80)

        layout.addSpacing(12)
        layout.addWidget(self._width_label)
        layout.addWidget(self._width_slider)

        layout.addItem(QSpacerItem(12, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        self.eraser_button = QToolButton(self)
        self.eraser_button.setText("Eraser")
        self.eraser_button.setCheckable(True)
        self.eraser_button.setAutoRaise(True)
        self.eraser_button.setToolTip("Toggle eraser mode")
        self.eraser_button.toggled.connect(self._handle_eraser_toggled)
        layout.addWidget(self.eraser_button)

        self.undo_button = QToolButton(self)
        self.undo_button.setText("Undo")
        self.undo_button.setToolTip("Undo last stroke (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undoRequested.emit)
        layout.addWidget(self.undo_button)

        self.fullscreen_button = QToolButton(self)
        self.fullscreen_button.setText("Full Screen")
        self.fullscreen_button.setToolTip("Toggle full screen view (Ctrl+F)")
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
                    self._last_color_index = idx
                    matched = True
                else:
                    button.setChecked(False)
            if not matched and self._color_group.buttons():
                self._color_group.buttons()[0].setChecked(True)
                self._current_color = QColor(self._colors[0][1])
                self._last_color_index = 0
        with QSignalBlocker(self.eraser_button):
            self.eraser_button.setChecked(False)

    def sync_width(self, width: float) -> None:
        self._current_width = float(width)
        with QSignalBlocker(self._width_slider):
            self._width_slider.setValue(int(round(self._current_width)))
        self._update_width_label()

    def _internal_color_changed(self, button_id: int) -> None:
        label, color_hex = self._colors[button_id]
        self._current_color = QColor(color_hex)
        self._last_color_index = button_id
        with QSignalBlocker(self.eraser_button):
            self.eraser_button.setChecked(False)
        self.colorChanged.emit(QColor(color_hex))

    def _internal_width_changed(self, value: int) -> None:
        self._current_width = float(value)
        self._update_width_label()
        self.widthChanged.emit(self._current_width)

    def _update_width_label(self) -> None:
        self._width_label.setText(f"Width: {int(self._current_width)} px")

    def sync_eraser(self, enabled: bool) -> None:
        with QSignalBlocker(self.eraser_button):
            self.eraser_button.setChecked(enabled)
        self._apply_eraser_state(enabled)

    def current_width(self) -> float:
        return self._current_width

    def current_color(self) -> QColor:
        return QColor(self._current_color)

    def _apply_eraser_state(self, checked: bool) -> None:
        if checked:
            with QSignalBlocker(self._color_group):
                for button in self._color_group.buttons():
                    button.setChecked(False)
        else:
            with QSignalBlocker(self._color_group):
                if self._color_group.buttons():
                    target = self._color_group.button(self._last_color_index)
                    if target:
                        target.setChecked(True)
                        self._current_color = QColor(self._colors[self._last_color_index][1])

    def _handle_eraser_toggled(self, checked: bool) -> None:
        self._apply_eraser_state(checked)
        self.eraserToggled.emit(checked)
        if not checked:
            self.colorChanged.emit(self._current_color)


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

    def set_eraser(self, width: float) -> None:
        self.view.set_eraser(width)

    def undo_last_stroke(self) -> None:
        self.view.undo_last_stroke()

    def set_active(self, active: bool) -> None:
        border = "#2d9bf0" if active else "#3a3a3a"
        self.setStyleSheet(self._base_style % border)
        if active:
            self._title_label.setStyleSheet("color: #ffffff; font-weight: 600; padding: 2px 0;")
            self.view.setFocus()
        else:
            self._title_label.setStyleSheet("color: #d0d0d0; font-size: 12px; padding: 2px 0;")


class CanvasFullScreenWindow(QMainWindow):
    """Dedicated window for drawing on a canvas in full screen."""

    closed = Signal()
    colorChanged = Signal(QColor)
    widthChanged = Signal(float)
    eraserToggled = Signal(bool)

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
        self.controls.eraserToggled.connect(self._forward_eraser_toggle)

        self._undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._undo_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._undo_shortcut.activated.connect(self.view.undo_last_stroke)

        self._fullscreen_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        self._fullscreen_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._fullscreen_shortcut.activated.connect(self.close)

        close_action = QAction("Exit Full Screen", self)
        close_action.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        close_action.triggered.connect(self.close)
        self.addAction(close_action)

    def set_tool_state(self, color: QColor, width: float, eraser: bool) -> None:
        if eraser:
            self.view.set_eraser(width)
        else:
            self.view.set_pen(color, width)
        self.controls.sync_color(color)
        self.controls.sync_width(width)
        self.controls.sync_eraser(eraser)

    def _forward_color_change(self, color: QColor) -> None:
        self.controls.sync_eraser(False)
        self.view.set_pen(color=color)
        self.colorChanged.emit(color)

    def _forward_width_change(self, width: float) -> None:
        if self.controls.eraser_button.isChecked():
            self.view.set_eraser(width=width)
        else:
            self.view.set_pen(width=width)
        self.widthChanged.emit(width)

    def _forward_eraser_toggle(self, enabled: bool) -> None:
        if enabled:
            self.view.set_eraser(self.controls.current_width())
        else:
            self.view.set_pen(color=self.controls.current_color(), width=self.controls.current_width())
        self.eraserToggled.emit(enabled)

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)


class CanvasPanel(QWidget):
    """Composite widget hosting canvas views, controls and full-screen logic."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._documents: dict[Path, CanvasDocument] = {}
        self._current_color = QColor(CanvasToolControls.DEFAULT_COLORS[0][1])
        self._current_width = 12.0
        self._active_fullscreen: Optional[CanvasFullScreenWindow] = None
        self._eraser_enabled = False

        self._view_widgets: list[CanvasViewWidget] = []
        self._active_widget: Optional[CanvasViewWidget] = None

        self.controls = CanvasToolControls()
        self.controls.colorChanged.connect(self._handle_color_change)
        self.controls.widthChanged.connect(self._handle_width_change)
        self.controls.undoRequested.connect(self._handle_undo)
        self.controls.fullscreenRequested.connect(self._handle_fullscreen_request)
        self.controls.eraserToggled.connect(self._handle_eraser_toggle)

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
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self._handle_fullscreen_request)

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
            document.strokesChanged.emit()

        if self._view_widgets:
            self._set_active_widget(self._view_widgets[0])

        self._update_placeholder_visibility()
        QTimer.singleShot(0, self._reset_all_views)

    def clear(self) -> None:
        self._clear_views()
        self._update_placeholder_visibility()

    def current_view(self) -> Optional[CanvasViewWidget]:
        return self._active_widget

    def _handle_color_change(self, color: QColor) -> None:
        self._current_color = QColor(color)
        self._eraser_enabled = False
        self.controls.sync_eraser(False)
        view = self.current_view()
        if view:
            view.set_pen(self._current_color, self._current_width)

    def _handle_width_change(self, width: float) -> None:
        self._current_width = float(width)
        view = self.current_view()
        if view:
            if self._eraser_enabled:
                view.set_eraser(self._current_width)
            else:
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
        self._active_fullscreen.set_tool_state(self._current_color, self._current_width, self._eraser_enabled)
        self._active_fullscreen.colorChanged.connect(self._sync_color_from_fullscreen)
        self._active_fullscreen.widthChanged.connect(self._sync_width_from_fullscreen)
        self._active_fullscreen.eraserToggled.connect(self._sync_eraser_from_fullscreen)
        self._active_fullscreen.closed.connect(self._handle_fullscreen_closed)
        self._active_fullscreen.showFullScreen()

    def _handle_fullscreen_closed(self) -> None:
        self._active_fullscreen = None

    def _sync_color_from_fullscreen(self, color: QColor) -> None:
        self._current_color = QColor(color)
        self.controls.sync_color(color)
        self._eraser_enabled = False
        view = self.current_view()
        if view:
            view.set_pen(self._current_color, self._current_width)

    def _sync_width_from_fullscreen(self, width: float) -> None:
        self._current_width = float(width)
        self.controls.sync_width(width)
        view = self.current_view()
        if view:
            if self._eraser_enabled:
                view.set_eraser(self._current_width)
            else:
                view.set_pen(self._current_color, self._current_width)

    def _sync_eraser_from_fullscreen(self, enabled: bool) -> None:
        self._eraser_enabled = enabled
        self.controls.sync_eraser(enabled)
        view = self.current_view()
        if view:
            if enabled:
                view.set_eraser(self._current_width)
            else:
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
        if self._eraser_enabled:
            self._active_widget.set_eraser(self._current_width)
        else:
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

    def _handle_eraser_toggle(self, enabled: bool) -> None:
        self._eraser_enabled = enabled
        view = self.current_view()
        if view:
            if enabled:
                view.set_eraser(self._current_width)
            else:
                view.set_pen(self._current_color, self._current_width)

    def _reset_all_views(self) -> None:
        for widget in self._view_widgets:
            widget.view.reset_view()
