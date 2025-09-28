import sys
import os
from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import Qt, QPoint, QRect, QSize, QEvent
from PySide6.QtGui import (
    QAction,
    QColor,
    QGuiApplication,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QTabletEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSlider,
    QToolBar,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
)


def load_image_any(path: str) -> Optional[QImage]:
    img = QImage()
    if not img.load(path):
        return None
    # Normalize to a fast format for painting
    return img.convertToFormat(QImage.Format.Format_RGBA8888)


class Canvas(QWidget):
    """
    A widget that displays a base image and an overlay image to draw on.
    Supports tablet pressure; falls back to mouse.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.base_image: Optional[QImage] = None
        self.overlay: Optional[QImage] = None
        self.last_pt: Optional[QPoint] = None

        self.pen_color = QColor(255, 0, 0)
        self.eraser_on = False
        self.base_pen_size = 3  # adjusted by pressure when using tablet
        self.pressure_scale = (0.5, 4.0)  # min/max multiplier for pressure
        self.zoom_scale: float = 1.0
        self._history: list[QImage] = []
        self._history_limit: int = 50
        # Panning state
        self.panning: bool = False
        self.pan_last_pos: Optional[QPoint] = None  # in global coordinates
        self.space_pan_active: bool = False
        self.setAttribute(Qt.WidgetAttribute.WA_TabletTracking, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ---------- image management ----------
    def set_image(self, img: QImage):
        self.base_image = img.copy()
        self.overlay = QImage(self.base_image.size(), QImage.Format.Format_RGBA8888)
        self.overlay.fill(Qt.GlobalColor.transparent)
        # Reset zoom and history on new image
        self.zoom_scale = 1.0
        self._history.clear()
        self._update_widget_size()
        self.updateGeometry()
        self.update()

    def has_image(self) -> bool:
        return self.base_image is not None

    def sizeHint(self):
        if self.base_image:
            w = int(self.base_image.width() * self.zoom_scale)
            h = int(self.base_image.height() * self.zoom_scale)
            return QSize(w, h)
        return super().sizeHint()

    # ---------- painting ----------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # center the image in the widget
        if not self.base_image:
            painter.fillRect(self.rect(), Qt.GlobalColor.darkGray)
            return

        target_rect = QRect(
            0,
            0,
            int(self.base_image.width() * self.zoom_scale),
            int(self.base_image.height() * self.zoom_scale),
        )
        painter.fillRect(self.rect(), Qt.GlobalColor.darkGray)
        painter.drawImage(target_rect, self.base_image)

        if self.overlay:
            painter.drawImage(target_rect, self.overlay)

    def _draw_line(
        self, p1: QPoint, p2: QPoint, pressure: float = 1.0, erasing: bool = False
    ):
        if not self.overlay:
            return
        pen_width = max(
            1,
            int(
                self.base_pen_size
                * max(self.pressure_scale[0], min(pressure, 1.0))
                * self.pressure_scale[1]
            ),
        )
        painter = QPainter(self.overlay)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if erasing:
            # Eraser: draw transparent by using CompositionMode_Clear
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            pen = QPen(
                Qt.GlobalColor.transparent,
                pen_width,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        else:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )
            pen = QPen(
                self.pen_color,
                pen_width,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )

        painter.setPen(pen)
        painter.drawLine(p1, p2)
        painter.end()
        self.update()

    # ---------- tablet events ----------
    def tabletEvent(self, event: QTabletEvent):
        if not self.base_image:
            event.ignore()
            return

        pos_widget = event.position().toPoint()
        pos = self._clamp_point(self._widget_to_image_point(pos_widget))

        if event.type() == QTabletEvent.Type.TabletPress:
            self._push_history()
            self.last_pt = pos
            event.accept()
            return

        if event.type() == QTabletEvent.Type.TabletMove and (
            event.buttons() & Qt.MouseButton.LeftButton
        ):
            if self.last_pt is not None:
                pressure = max(0.05, float(event.pressure()))
                erasing = self.eraser_on or (
                    event.pointerType() == QTabletEvent.PointerType.Eraser
                )
                self._draw_line(self.last_pt, pos, pressure=pressure, erasing=erasing)
                self.last_pt = pos
            event.accept()
            return

        if event.type() == QTabletEvent.Type.TabletRelease:
            self.last_pt = None
            event.accept()
            return

        event.ignore()

    # ---------- mouse fallback ----------
    def mousePressEvent(self, event):
        if not self.base_image:
            return
        # Alt+Left or Space+Left -> start panning
        if event.button() == Qt.MouseButton.LeftButton and (
            (event.modifiers() & Qt.KeyboardModifier.AltModifier)
            or self.space_pan_active
        ):
            self.panning = True
            self.pan_last_pos = event.globalPosition().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._push_history()
            self.last_pt = self._clamp_point(
                self._widget_to_image_point(event.position().toPoint())
            )

    def mouseMoveEvent(self, event):
        if not self.base_image:
            return
        # Handle panning
        if self.panning and (event.buttons() & Qt.MouseButton.LeftButton):
            sa = self._scroll_area()
            if sa is not None and self.pan_last_pos is not None:
                cur_global = event.globalPosition().toPoint()
                dx = cur_global.x() - self.pan_last_pos.x()
                dy = cur_global.y() - self.pan_last_pos.y()
                hbar = sa.horizontalScrollBar()
                vbar = sa.verticalScrollBar()
                hbar.setValue(hbar.value() - dx)
                vbar.setValue(vbar.value() - dy)
                self.pan_last_pos = cur_global
            return
        if event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton):
            if self.last_pt is not None:
                pos = self._clamp_point(
                    self._widget_to_image_point(event.position().toPoint())
                )
                erasing = self.eraser_on or (
                    event.buttons() & Qt.MouseButton.RightButton
                )
                # No pressure with mouse; use mid-range multiplier
                self._draw_line(self.last_pt, pos, pressure=0.6, erasing=erasing)
                self.last_pt = pos

    def mouseReleaseEvent(self, event):
        if self.panning and event.button() == Qt.MouseButton.LeftButton:
            self.panning = False
            self.pan_last_pos = None
            # Restore cursor if space not held
            if not self.space_pan_active:
                self.unsetCursor()
            return
        self.last_pt = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not self.space_pan_active:
            self.space_pan_active = True
            # If not currently dragging, show open hand
            if not self.panning:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.space_pan_active:
            self.space_pan_active = False
            if not self.panning:
                self.unsetCursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    # ---------- utils ----------
    def wheelEvent(self, event):
        if not self.base_image:
            return
        mods = event.modifiers()
        sa = self._scroll_area()
        # Ctrl + wheel -> zoom at cursor
        if mods & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta == 0:
                return
            factor = 1.25 if delta > 0 else 0.8
            anchor_pos = None
            if sa is not None:
                # Map global cursor to viewport coordinates for stable anchor
                global_pos = event.globalPosition().toPoint()
                anchor_pos = sa.viewport().mapFromGlobal(global_pos)
            self._zoom_by(factor, anchor_widget_pos=anchor_pos)
            event.accept()
            return
        # Shift + wheel -> horizontal scroll
        if sa is not None:
            if mods & Qt.KeyboardModifier.ShiftModifier:
                hbar = sa.horizontalScrollBar()
                hbar.setValue(hbar.value() - event.angleDelta().y())
            else:
                vbar = sa.verticalScrollBar()
                vbar.setValue(vbar.value() - event.angleDelta().y())
            event.accept()
            return
        super().wheelEvent(event)

    def _scroll_area(self) -> Optional[QScrollArea]:
        w = self.parent()
        while w is not None:
            if isinstance(w, QScrollArea):
                return w
            w = w.parent()
        return None

    def _zoom_by(self, factor: float, anchor_widget_pos: Optional[QPoint] = None):
        new_scale = max(0.1, min(10.0, self.zoom_scale * factor))
        self._apply_zoom(new_scale, anchor_widget_pos)

    def _apply_zoom(self, new_scale: float, anchor_widget_pos: Optional[QPoint]):
        if not self.base_image:
            return
        sa = self._scroll_area()
        old_scale = self.zoom_scale
        self.zoom_scale = new_scale
        # Update widget size based on new scale
        self._update_widget_size()
        self.updateGeometry()
        self.update()
        if sa is None or anchor_widget_pos is None:
            return
        # Keep the image point under the cursor stationary in the viewport
        hbar = sa.horizontalScrollBar()
        vbar = sa.verticalScrollBar()
        old_origin_x = hbar.value()
        old_origin_y = vbar.value()
        pos_wx = int(anchor_widget_pos.x())
        pos_wy = int(anchor_widget_pos.y())
        # Image coordinate under cursor before zoom
        img_x = (old_origin_x + pos_wx) / max(1e-6, old_scale)
        img_y = (old_origin_y + pos_wy) / max(1e-6, old_scale)
        # Desired widget coords after zoom
        new_wx = int(img_x * self.zoom_scale)
        new_wy = int(img_y * self.zoom_scale)
        new_origin_x = new_wx - pos_wx
        new_origin_y = new_wy - pos_wy
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), new_origin_x)))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), new_origin_y)))

    def _update_widget_size(self):
        if not self.base_image:
            return
        w = max(1, int(self.base_image.width() * self.zoom_scale))
        h = max(1, int(self.base_image.height() * self.zoom_scale))
        self.setMinimumSize(w, h)

    def _widget_to_image_point(self, p: QPoint) -> QPoint:
        if not self.base_image:
            return p
        x = int(p.x() / max(1e-6, self.zoom_scale))
        y = int(p.y() / max(1e-6, self.zoom_scale))
        return QPoint(x, y)

    def _clamp_point(self, p: QPoint) -> QPoint:
        if not self.base_image:
            return p
        x = max(0, min(self.base_image.width() - 1, p.x()))
        y = max(0, min(self.base_image.height() - 1, p.y()))
        return QPoint(x, y)

    def _push_history(self):
        if self.overlay is None:
            return
        self._history.append(self.overlay.copy())
        if len(self._history) > self._history_limit:
            self._history.pop(0)

    def undo_last(self):
        if not self._history:
            return
        last = self._history.pop()
        self.overlay = last
        self.update()

    def merge_result(self) -> Optional[QImage]:
        if not (self.base_image and self.overlay):
            return None
        merged = self.base_image.copy()
        painter = QPainter(merged)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.drawImage(0, 0, self.overlay)
        painter.end()
        return merged


class MainWindow(QMainWindow):
    def __init__(self, init_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("SketchPad — fast tablet drawing")
        self.canvas = Canvas()
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)
        self.setCentralWidget(self.scroll)
        # Ensure we can intercept wheel events for zoom
        self.scroll.viewport().installEventFilter(self)

        self.current_path: Optional[Path] = None

        # Toolbar
        tb = QToolBar("Tools")
        tb.setMovable(False)
        self.addToolBar(tb)

        act_open = QAction("Open…", self)
        act_open.triggered.connect(self.open_image)
        tb.addAction(act_open)

        act_save = QAction("Save", self)
        act_save.triggered.connect(self.save_image)
        tb.addAction(act_save)

        act_save_as = QAction("Save As…", self)
        act_save_as.triggered.connect(self.save_image_as)
        tb.addAction(act_save_as)

        # Undo action (Ctrl+Z)
        act_undo = QAction("Undo", self)
        act_undo.setShortcut(Qt.Modifier.CTRL | Qt.Key.Key_Z)
        act_undo.triggered.connect(self.canvas.undo_last)
        self.addAction(act_undo)

        tb.addSeparator()

        self.btn_pen = QPushButton("Pen")
        self.btn_pen.setCheckable(True)
        self.btn_pen.setChecked(True)
        self.btn_pen.clicked.connect(self._on_pen_clicked)
        tb.addWidget(self.btn_pen)

        self.btn_eraser = QPushButton("Eraser")
        self.btn_eraser.setCheckable(True)
        self.btn_eraser.clicked.connect(self._on_eraser_clicked)
        tb.addWidget(self.btn_eraser)

        tb.addSeparator()

        self.color_btn = QPushButton("Color")
        self.color_btn.clicked.connect(self.pick_color)
        tb.addWidget(self.color_btn)

        # Quick-access pastel colors
        self.quick_colors = [
            ("Pastel Red", QColor(255, 128, 128)),
            ("Pastel Green", QColor(144, 238, 144)),
            ("Pastel Blue", QColor(173, 216, 230)),
            ("Pastel Yellow", QColor(255, 255, 153)),
            ("Pastel Pink", QColor(255, 182, 193)),
        ]
        self.quick_color_buttons: List[QPushButton] = []
        for name, color in self.quick_colors:
            btn = QPushButton()
            btn.setFixedSize(22, 22)
            btn.setToolTip(name)
            btn.setStyleSheet(
                f"background-color: rgba({color.red()},{color.green()},{color.blue()},255);"
                "border: 1px solid #666; margin: 2px;"
            )
            # Use lambda with default arg to capture color
            btn.clicked.connect(lambda _, c=color: self._set_pen_color(c))
            tb.addWidget(btn)
            self.quick_color_buttons.append(btn)

        size_label = QLabel(" Size ")
        tb.addWidget(size_label)

        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(20)
        self.size_slider.setValue(self.canvas.base_pen_size)
        self.size_slider.setFixedWidth(120)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        tb.addWidget(self.size_slider)

        # Shortcuts
        self.btn_pen.setShortcut(Qt.Key.Key_P)
        self.btn_eraser.setShortcut(Qt.Key.Key_E)
        act_open.setShortcut(Qt.Modifier.CTRL | Qt.Key.Key_O)
        act_save.setShortcut(Qt.Modifier.CTRL | Qt.Key.Key_S)

        # Status
        self.statusBar().showMessage("Ready")

        # Load initial image if provided
        if init_path:
            self._try_load(init_path)

    def eventFilter(self, obj, event):
        if obj is self.scroll.viewport() and event.type() == QEvent.Type.Wheel:
            # Forward to canvas handler so Ctrl/Shift behavior applies
            self.canvas.wheelEvent(event)
            return True
        return super().eventFilter(obj, event)

    # -------- toolbar handlers --------
    def _on_pen_clicked(self, checked: bool):
        if checked:
            self.btn_eraser.setChecked(False)
            self.canvas.eraser_on = False
            self.statusBar().showMessage("Pen")
        else:
            # keep at least one active
            self.btn_pen.setChecked(True)

    def _on_eraser_clicked(self, checked: bool):
        if checked:
            self.btn_pen.setChecked(False)
            self.canvas.eraser_on = True
            self.statusBar().showMessage("Eraser")
        else:
            self.btn_eraser.setChecked(True)

    def _on_size_changed(self, v: int):
        self.canvas.base_pen_size = v
        self.statusBar().showMessage(f"Pen size: {v}px", 1500)

    def pick_color(self):
        col = QColorDialog.getColor(self.canvas.pen_color, self, "Pick Pen Color")
        if col.isValid():
            self.canvas.pen_color = col

    def _set_pen_color(self, col: QColor):
        self.canvas.pen_color = col
        # Switch to pen if eraser is active
        if self.btn_eraser.isChecked():
            self.btn_eraser.setChecked(False)
            self.btn_pen.setChecked(True)
            self.canvas.eraser_on = False
        self.statusBar().showMessage(
            f"Color set to RGB({col.red()},{col.green()},{col.blue()})", 1500
        )

    # -------- file ops --------
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp)"
        )
        if path:
            self._try_load(path)

    def _try_load(self, path: str):
        img = load_image_any(path)
        if img is None:
            QMessageBox.warning(self, "Open failed", "Could not load image.")
            return
        self.canvas.set_image(img)
        self.current_path = Path(path)
        self.setWindowTitle(f"SketchPad — {self.current_path.name}")
        self.statusBar().showMessage(f"Loaded: {self.current_path}")

    def _merged_or_warn(self) -> Optional[QImage]:
        if not self.canvas.has_image():
            QMessageBox.information(self, "No image", "Open an image first.")
            return None
        merged = self.canvas.merge_result()
        if merged is None:
            QMessageBox.warning(self, "Save failed", "Could not merge image.")
        return merged

    def save_image(self):
        if self.current_path is None:
            self.save_image_as()
            return
        merged = self._merged_or_warn()
        if merged is None:
            return
        if not merged.save(str(self.current_path)):
            QMessageBox.warning(
                self, "Save failed", f"Could not write {self.current_path}"
            )
            return
        self.statusBar().showMessage(f"Saved: {self.current_path}", 2000)

    def save_image_as(self):
        if not self.canvas.has_image():
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        # Default to same directory/format if possible
        start_dir = str(self.current_path.parent) if self.current_path else ""
        default_name = self.current_path.name if self.current_path else "untitled.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save As",
            os.path.join(start_dir, default_name),
            "PNG (*.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg);;All Files (*)",
        )
        if not path:
            return
        merged = self._merged_or_warn()
        if merged is None:
            return
        fmt = Path(path).suffix.lower()
        ok = merged.save(path)  # Qt infers format from suffix
        if not ok:
            QMessageBox.warning(self, "Save failed", f"Could not write {path}")
            return
        self.current_path = Path(path)
        self.setWindowTitle(f"SketchPad — {self.current_path.name}")
        self.statusBar().showMessage(f"Saved: {self.current_path}", 2000)


def main():
    # High-DPI friendly for crisp pen strokes
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    init_path = sys.argv[1] if len(sys.argv) > 1 else None
    win = MainWindow(init_path)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
