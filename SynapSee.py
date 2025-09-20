
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SynapSee viewer: fast multi-trace viewer with zoom-centric navigation.

Usage:
    python SynapSee.py
    python SynapSee.py path/to/folder
"""

import glob
import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

try:
    import cv2
except Exception:
    cv2 = None


@dataclass
class Series:
    name: str
    t: np.ndarray
    y: np.ndarray


def nice_time_range(t_arrays):
    vals = [(np.nanmin(t), np.nanmax(t)) for t in t_arrays if t is not None and len(t)]
    if not vals:
        return 0.0, 1.0
    return float(min(v[0] for v in vals)), float(max(v[1] for v in vals))


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def find_nearest_frame(frame_times, t):
    if frame_times is None or len(frame_times) == 0:
        return 0
    i = int(np.searchsorted(frame_times, t, "left"))
    if i <= 0:
        return 0
    if i >= len(frame_times):
        return len(frame_times) - 1
    return i - 1 if abs(t - frame_times[i - 1]) <= abs(frame_times[i] - t) else i


def segment_for_window(t, y, t0, t1, max_pts=4000):
    """Return (tx, yx) for the [t0, t1] window using peak-preserving min/max bins."""
    if t1 <= t0:
        return np.empty(0), np.empty(0)

    i0 = max(0, np.searchsorted(t, t0) - 1)
    i1 = min(len(t), np.searchsorted(t, t1) + 1)
    ts = t[i0:i1]
    ys = y[i0:i1]
    n = len(ts)
    if n <= 2:
        return ts, ys
    if n <= max_pts:
        return ts, ys

    bins = max(1, max_pts // 2)
    edges = np.linspace(t0, t1, bins + 1)
    bi = np.clip(np.digitize(ts, edges) - 1, 0, bins - 1)

    order = np.argsort(bi, kind="mergesort")
    bi_s = bi[order]
    ts_s = ts[order]
    ys_s = ys[order]
    starts = np.searchsorted(bi_s, np.arange(bins), "left")
    ends = np.searchsorted(bi_s, np.arange(bins), "right")

    out_t = np.empty(2 * bins, dtype=float)
    out_y = np.empty(2 * bins, dtype=float)
    k = 0
    for b in range(bins):
        s = starts[b]
        e = ends[b]
        if s == e:
            tt = 0.5 * (edges[b] + edges[b + 1])
            out_t[k] = tt
            out_y[k] = np.nan
            k += 1
            out_t[k] = tt
            out_y[k] = np.nan
            k += 1
        else:
            yb = ys_s[s:e]
            tb = ts_s[s:e]
            ymin = float(np.nanmin(yb))
            ymax = float(np.nanmax(yb))
            tmid = float(tb[len(tb) // 2])
            out_t[k] = tmid
            out_y[k] = ymin
            k += 1
            out_t[k] = tmid
            out_y[k] = ymax
            k += 1

    return out_t[:k], out_y[:k]


class VideoWorker(QtCore.QObject):
    frameReady = QtCore.Signal(int, QtGui.QImage)
    opened = QtCore.Signal(bool, str)

    def __init__(self, cache_frames=120):
        super().__init__()
        self.cap = None
        self.cache = OrderedDict()
        self.cache_frames = int(cache_frames)

    @QtCore.Slot(str)
    def open(self, path):
        if cv2 is None:
            self.opened.emit(False, "OpenCV (cv2) not installed.")
            return
        try:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            ok = bool(self.cap.isOpened())
            self.opened.emit(ok, "" if ok else f"Failed to open: {path}")
        except Exception as e:
            self.opened.emit(False, str(e))

    @QtCore.Slot(int)
    def requestFrame(self, idx):
        if self.cap is None:
            return

        idx = int(idx)
        qimg = self.cache.get(idx)
        if qimg is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QtGui.QImage(
                    rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888
                ).copy()
                self.cache[idx] = qimg
                if len(self.cache) > self.cache_frames:
                    self.cache.popitem(last=False)

        if qimg is not None:
            self.frameReady.emit(idx, qimg)

    @QtCore.Slot()
    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cache.clear()


class ZoomViewBox(pg.ViewBox):
    sigDragStart = QtCore.Signal(float)
    sigDragUpdate = QtCore.Signal(float)
    sigDragFinish = QtCore.Signal(float)
    sigCursorRequest = QtCore.Signal(float)
    sigDoubleClick = QtCore.Signal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(x=False, y=True)
        self._dragging = False

    def mouseDragEvent(self, ev, axis=None):
        if (
            ev.button() == QtCore.Qt.LeftButton
            and ev.modifiers() == QtCore.Qt.NoModifier
        ):
            x = float(self.mapSceneToView(ev.scenePos()).x())
            if ev.isStart():
                self._dragging = True
                self.sigDragStart.emit(x)
                ev.accept()
                return
            if ev.isFinish():
                if self._dragging:
                    self.sigDragFinish.emit(x)
                self._dragging = False
                ev.accept()
                return
            if self._dragging:
                self.sigDragUpdate.emit(x)
                ev.accept()
                return
        super().mouseDragEvent(ev, axis=axis)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            x = float(self.mapSceneToView(ev.scenePos()).x())
            if ev.double():
                self.sigDoubleClick.emit(x)
            else:
                self.sigCursorRequest.emit(x)
            ev.accept()
            return
        super().mouseClickEvent(ev)


class SynapSeeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SynapSee — Multi-Trace Viewer")
        self.resize(1400, 900)

        pg.setConfigOptions(
            antialias=False, useOpenGL=True, background="k", foreground="w"
        )

        self.series: list[Series] = []
        self.t_global_min = 0.0
        self.t_global_max = 1.0

        self.plots: list[pg.PlotItem] = []
        self.curves: list[pg.PlotDataItem] = []
        self.plot_cur_lines: list[pg.InfiniteLine] = []
        self.zoom_regions: list[pg.LinearRegionItem] = []
        self._y_controls = []

        self.max_pts_per_plot = 4000

        self.view_start = 0.0
        self.view_end = 1.0
        self.view_span = 1.0
        self.cursor_time = 0.0

        self._zoom_start: Optional[float] = None
        self._zoom_end: Optional[float] = None

        self._updating_view_range = False
        self._block_span = False
        self._block_slider = False

        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._refresh_curves)

        self.video_frame_times = None
        self._video_is_open = False
        self._video_thread = QtCore.QThread(self)
        self._video_worker = VideoWorker(cache_frames=120)
        self._video_worker.moveToThread(self._video_thread)

        self.is_playing = False
        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.timeout.connect(self._advance_playback_frame)
        self.playback_elapsed_timer = QtCore.QElapsedTimer()

        self.master_plot: Optional[pg.PlotItem] = None
        self.master_viewbox: Optional[ZoomViewBox] = None

        self._build_ui()

        self._video_worker.frameReady.connect(self._on_frame_ready)
        self._video_worker.opened.connect(self._on_video_opened)

        self._video_thread.start()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        layout.addLayout(top)
        top.addWidget(QtWidgets.QLabel("Span (s):"))
        self.span_spin = QtWidgets.QDoubleSpinBox()
        self.span_spin.setRange(0.1, 86400.0)
        self.span_spin.setDecimals(2)
        self.span_spin.valueChanged.connect(self._on_span_changed)
        top.addWidget(self.span_spin)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Navigate:"))
        self.nav_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.nav_slider.setRange(0, 10000)
        self.nav_slider.valueChanged.connect(self._on_nav_slider_changed)
        top.addWidget(self.nav_slider, 1)
        top.addSpacing(12)
        self.zoom_out_btn = QtWidgets.QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(self._zoom_out)
        top.addWidget(self.zoom_out_btn)
        self.reset_btn = QtWidgets.QPushButton("Reset Zoom")
        self.reset_btn.clicked.connect(self._reset_view_full)
        top.addWidget(self.reset_btn)
        top.addStretch(1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter, 1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_area = pg.GraphicsLayoutWidget()
        left_layout.addWidget(self.plot_area, 1)
        splitter.addWidget(left)

        right = QtWidgets.QWidget()
        right.setMinimumWidth(170)
        right_layout = QtWidgets.QVBoxLayout(right)
        self.video_label = QtWidgets.QLabel("No video")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumHeight(240)
        self.video_label.setStyleSheet("background-color:#222;border:1px solid #444;")
        right_layout.addWidget(self.video_label, 3)

        cursor_row = QtWidgets.QHBoxLayout()
        cursor_row.addWidget(QtWidgets.QLabel("Cursor:"))
        self.window_cursor_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.window_cursor_slider.setRange(0, 10000)
        self.window_cursor_slider.valueChanged.connect(self._on_window_cursor_changed)
        cursor_row.addWidget(self.window_cursor_slider)
        cursor_widget = QtWidgets.QWidget()
        cursor_widget.setLayout(cursor_row)
        right_layout.addWidget(cursor_widget)

        ygroup = QtWidgets.QGroupBox("Per-trace Y-axis")
        ylayout = QtWidgets.QVBoxLayout(ygroup)
        self.y_controls_container = QtWidgets.QWidget()
        self.y_controls_layout = QtWidgets.QFormLayout(self.y_controls_container)
        ylayout.addWidget(self.y_controls_container)
        ylayout.addStretch(1)
        right_layout.addWidget(ygroup, 2)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.span_spin.setEnabled(False)
        self.nav_slider.setEnabled(False)
        self.window_cursor_slider.setEnabled(False)
        self.zoom_out_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)

        self.status = self.statusBar()
        self._update_status()
        self._build_menu()

    def _build_menu(self):
        mfile = self.menuBar().addMenu("&File")
        load_ts = QtGui.QAction("Load &Time Series…", self)
        load_ts.triggered.connect(self._on_load_time_series)
        mfile.addAction(load_ts)
        load_video = QtGui.QAction("Load &Video && Frame Times…", self)
        load_video.triggered.connect(self._on_load_video)
        mfile.addAction(load_video)
        mfile.addSeparator()
        quit_action = QtGui.QAction("&Quit", self)
        quit_action.triggered.connect(self.close)
        mfile.addAction(quit_action)

        mview = self.menuBar().addMenu("&View")
        reset_zoom = QtGui.QAction("&Reset Zoom", self)
        reset_zoom.triggered.connect(self._reset_view_full)
        mview.addAction(reset_zoom)
        zoom_out = QtGui.QAction("Zoom &Out (2x)", self)
        zoom_out.triggered.connect(self._zoom_out)
        mview.addAction(zoom_out)

        mhelp = self.menuBar().addMenu("&Help")
        help_action = QtGui.QAction("Shortcuts / Help", self)
        help_action.triggered.connect(self._show_help)
        mhelp.addAction(help_action)
    def _on_load_time_series(self):
        self._stop_playback_if_playing()
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder with *_t.npy and *_y.npy"
        )
        if not folder:
            return
        try:
            if not self.load_series_folder(folder):
                QtWidgets.QMessageBox.warning(
                    self, "No data", "No *_t.npy / *_y.npy pairs found."
                )
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load error", str(e))

    def load_series_folder(self, folder):
        series = self._load_series_from_folder(folder)
        if series:
            self.set_series(series)
            return True
        return False

    def _load_series_from_folder(self, folder):
        pairs = []
        for tpath in glob.glob(os.path.join(folder, "*_t.npy")):
            name = os.path.basename(tpath)[:-6]
            ypath = os.path.join(folder, f"{name}_y.npy")
            if os.path.exists(ypath):
                pairs.append((name, tpath, ypath))
        if not pairs:
            return []
        series = []
        for name, tpath, ypath in sorted(pairs):
            t = np.load(tpath).astype(float)
            y = np.load(ypath).astype(float)
            if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
                raise ValueError(f"{name}: t and y must be 1-D and equal length")
            series.append(Series(name, t, y))
        return series

    def set_series(self, series_list):
        self.series = series_list

        self.plot_area.clear()
        self.plots.clear()
        self.curves.clear()
        self.plot_cur_lines.clear()
        self.zoom_regions.clear()
        self.master_plot = None
        self.master_viewbox = None

        self.t_global_min, self.t_global_max = nice_time_range([s.t for s in self.series])
        if self.t_global_max <= self.t_global_min:
            self.t_global_max = self.t_global_min + 1.0

        self.view_start = self.t_global_min
        self.view_end = self.t_global_max
        self.view_span = max(1e-6, self.view_end - self.view_start)
        self.cursor_time = self.view_start

        for idx, s in enumerate(self.series):
            vb = ZoomViewBox()
            plt = self.plot_area.addPlot(row=idx, col=0, viewBox=vb)
            plt.setLabel("left", s.name)
            plt.setLabel(
                "bottom", "Time", units="s" if idx == len(self.series) - 1 else None
            )
            plt.showGrid(x=True, y=True, alpha=0.15)
            plt.enableAutoRange("x", False)

            pen = pg.mkPen((150, 220, 255), width=1)
            curve = pg.PlotDataItem([], [], pen=pen, antialias=False)
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
            plt.addItem(curve)

            cur_line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 120))
            )
            plt.addItem(cur_line)

            zoom_region = pg.LinearRegionItem(
                values=(0, 0), brush=pg.mkBrush(100, 200, 255, 40), movable=False
            )
            zoom_region.setZValue(-10)
            zoom_region.hide()
            plt.addItem(zoom_region)

            self.plots.append(plt)
            self.curves.append(curve)
            self.plot_cur_lines.append(cur_line)
            self.zoom_regions.append(zoom_region)

            if self.master_plot is None:
                self.master_plot = plt
                self.master_viewbox = vb
                vb.sigXRangeChanged.connect(self._on_master_xrange_changed)
            else:
                plt.setXLink(self.master_plot)

            vb.sigDragStart.connect(self._on_zoom_drag_start)
            vb.sigDragUpdate.connect(self._on_zoom_drag_update)
            vb.sigDragFinish.connect(self._on_zoom_drag_finish)
            vb.sigCursorRequest.connect(self._on_cursor_requested)
            vb.sigDoubleClick.connect(self._on_zoom_double_click)

        self.span_spin.setEnabled(True)
        self.nav_slider.setEnabled(True)
        self.window_cursor_slider.setEnabled(True)
        self.zoom_out_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

        self._rebuild_y_controls()
        self._reset_view_full()
        self._update_status(f"Loaded {len(self.series)} series.")
    def _rebuild_y_controls(self):
        layout = self.y_controls_layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self._y_controls = []
        for idx, s in enumerate(self.series):
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            auto = QtWidgets.QCheckBox("Auto")
            auto.setChecked(True)
            mn = QtWidgets.QDoubleSpinBox()
            mx = QtWidgets.QDoubleSpinBox()
            for box in (mn, mx):
                box.setDecimals(3)
                box.setRange(-1e12, 1e12)
                box.setEnabled(False)
            dmin = float(np.nanmin(s.y)) if len(s.y) else -1.0
            dmax = float(np.nanmax(s.y)) if len(s.y) else 1.0
            mn.setValue(dmin)
            mx.setValue(dmax)

            def make_apply(i=idx, auto=auto, mn=mn, mx=mx):
                def _apply(_=None):
                    plt = self.plots[i]
                    if auto.isChecked():
                        plt.enableAutoRange("y", True)
                        mn.setEnabled(False)
                        mx.setEnabled(False)
                    else:
                        plt.enableAutoRange("y", False)
                        lo, hi = mn.value(), mx.value()
                        if hi <= lo:
                            hi = lo + 1e-6
                        plt.setYRange(lo, hi, padding=0.05)
                        mn.setEnabled(True)
                        mx.setEnabled(True)

                return _apply

            auto.stateChanged.connect(make_apply())
            mn.editingFinished.connect(make_apply())
            mx.editingFinished.connect(make_apply())

            row_layout.addWidget(QtWidgets.QLabel(s.name))
            row_layout.addStretch(1)
            row_layout.addWidget(auto)
            row_layout.addWidget(QtWidgets.QLabel("Min"))
            row_layout.addWidget(mn)
            row_layout.addWidget(QtWidgets.QLabel("Max"))
            row_layout.addWidget(mx)

            layout.addRow(row_widget)
            self._y_controls.append((auto, mn, mx))

    def _stop_playback_if_playing(self):
        if self.is_playing:
            self.is_playing = False
            self.playback_timer.stop()
            self._update_status("Playback stopped.")

    def _toggle_playback(self):
        if self.is_playing:
            self._stop_playback_if_playing()
        else:
            if self.video_frame_times is None:
                self._update_status("No video loaded to play.")
                return
            self.is_playing = True
            self.playback_elapsed_timer.start()
            self.playback_timer.start(16)
            self._update_status("Playing...")

    def _advance_playback_frame(self):
        if not self.is_playing:
            return
        dt_ms = self.playback_elapsed_timer.restart()
        dt_sec = dt_ms / 1000.0
        t_start = self.view_start
        t_end = self.view_end
        if t_end <= t_start:
            return
        new_cursor_time = self.cursor_time + dt_sec
        if new_cursor_time >= t_end:
            new_cursor_time = t_start + (new_cursor_time - t_end)
            if new_cursor_time >= t_end:
                new_cursor_time = t_start
        self._set_cursor_time(new_cursor_time, update_slider=True)
    def _page(self, direction: int):
        if not self.series:
            return
        self._stop_playback_if_playing()
        direction = 1 if direction >= 1 else -1
        span = self.view_span
        if span <= 0:
            return
        start = self.view_start + direction * span
        self._set_view_range(start, start + span, ensure_cursor=True)

    def _on_span_changed(self, value):
        if self._block_span or not self.series:
            return
        self._stop_playback_if_playing()
        span = max(0.1, float(value))
        total = max(1e-6, self.t_global_max - self.t_global_min)
        if span >= total:
            self._reset_view_full()
            return
        center = clamp(
            self.cursor_time,
            self.t_global_min + 0.5 * span,
            self.t_global_max - 0.5 * span,
        )
        self._set_view_range(center - 0.5 * span, center + 0.5 * span, ensure_cursor=True)

    def _on_nav_slider_changed(self, value):
        if self._block_slider or not self.series:
            return
        self._stop_playback_if_playing()
        total = max(1e-9, self.t_global_max - self.t_global_min)
        span = min(self.view_span, total)
        if total <= span:
            self._set_view_range(self.t_global_min, self.t_global_max, ensure_cursor=False)
            return
        start = self.t_global_min + (value / 10000.0) * (total - span)
        self._set_view_range(start, start + span, ensure_cursor=False)

    def _on_window_cursor_changed(self, value):
        if not self.series:
            return
        self._stop_playback_if_playing()
        frac = value / 10000.0
        t = self.view_start + frac * self.view_span
        self._set_cursor_time(t, update_slider=False)

    def _update_window_cursor_from_cursor_time(self):
        if self.view_span <= 0:
            frac = 0.0
        else:
            frac = clamp((self.cursor_time - self.view_start) / self.view_span, 0.0, 1.0)
        self.window_cursor_slider.blockSignals(True)
        self.window_cursor_slider.setValue(int(round(frac * 10000)))
        self.window_cursor_slider.blockSignals(False)

    def _update_nav_slider_from_view(self):
        if not self.series:
            self.nav_slider.blockSignals(True)
            self.nav_slider.setValue(0)
            self.nav_slider.blockSignals(False)
            return
        total = max(1e-9, self.t_global_max - self.t_global_min)
        span = min(self.view_span, total)
        if total <= span:
            value = 0
        else:
            frac = clamp(
                (self.view_start - self.t_global_min) / (total - span), 0.0, 1.0
            )
            value = int(round(frac * 10000))
        self.nav_slider.blockSignals(True)
        self.nav_slider.setValue(value)
        self.nav_slider.blockSignals(False)
    def _set_cursor_time(self, t, update_slider=True):
        self.cursor_time = clamp(t, self.t_global_min, self.t_global_max)
        self._update_cursor_lines()
        if update_slider:
            self._update_window_cursor_from_cursor_time()
        if self.video_frame_times is not None and len(self.video_frame_times):
            idx = find_nearest_frame(self.video_frame_times, self.cursor_time)
            QtCore.QMetaObject.invokeMethod(
                self._video_worker,
                "requestFrame",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(int, int(idx)),
            )
        if not self.is_playing:
            self._update_status()

    def _update_cursor_lines(self):
        for ln in self.plot_cur_lines:
            ln.setPos(self.cursor_time)

    def _set_view_range(self, start, end, ensure_cursor=True):
        if not self.series:
            return
        span = max(1e-6, end - start)
        total = max(1e-6, self.t_global_max - self.t_global_min)
        if span >= total:
            start = self.t_global_min
            end = self.t_global_max
            span = total
        else:
            start = clamp(start, self.t_global_min, self.t_global_max - span)
            end = start + span
        self.view_start = start
        self.view_end = end
        self.view_span = span

        self._updating_view_range = True
        for plt in self.plots:
            plt.enableAutoRange("x", False)
            plt.setXRange(start, end, padding=0.0)
        self._updating_view_range = False

        self._update_controls_from_view()

        if ensure_cursor:
            new_cursor = clamp(self.cursor_time, self.view_start, self.view_end)
            self._set_cursor_time(new_cursor, update_slider=True)
        else:
            self._update_cursor_lines()
            self._update_status()

        self._schedule_refresh()

    def _update_controls_from_view(self):
        total = max(0.1, self.t_global_max - self.t_global_min)
        max_span = max(total, self.view_span)
        self._block_span = True
        self.span_spin.setMaximum(max_span)
        self.span_spin.setValue(self.view_span)
        self._block_span = False
        self._update_nav_slider_from_view()

    def _schedule_refresh(self):
        self._refresh_timer.start(0)

    def _refresh_curves(self):
        if not self.series:
            return
        t0, t1 = self.view_start, self.view_end
        max_pts = self._target_pts()
        for s, curve in zip(self.series, self.curves):
            tx, yx = segment_for_window(s.t, s.y, t0, t1, max_pts=max_pts)
            curve.setData(tx, yx, _callSync="off")

    def _target_pts(self):
        if not self.plots:
            return self.max_pts_per_plot
        vb = self.plots[0].getViewBox()
        px = max(300, int(vb.width()))
        return int(min(2 * px, self.max_pts_per_plot))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._schedule_refresh()

    def _on_master_xrange_changed(self, _viewbox, xrange_):
        if self._updating_view_range or not self.series:
            return
        start, end = map(float, xrange_)
        if not (math.isfinite(start) and math.isfinite(end)):
            return
        if end <= start:
            return
        self.view_start = start
        self.view_end = end
        self.view_span = max(1e-6, end - start)
        self._update_controls_from_view()
        self._schedule_refresh()
        self._update_cursor_lines()
        self._update_status()
    def _on_zoom_drag_start(self, x):
        self._stop_playback_if_playing()
        self._zoom_start = x
        self._zoom_end = x
        self._show_zoom_region()

    def _on_zoom_drag_update(self, x):
        self._zoom_end = x
        self._show_zoom_region()

    def _on_zoom_drag_finish(self, x):
        self._zoom_end = x
        self._show_zoom_region()
        if self._zoom_start is not None and self._zoom_end is not None:
            a = float(min(self._zoom_start, self._zoom_end))
            b = float(max(self._zoom_start, self._zoom_end))
            if abs(b - a) >= 1e-3:
                self._set_view_range(a, b, ensure_cursor=True)
            else:
                self._set_cursor_time(clamp(a, self.t_global_min, self.t_global_max), True)
        self._clear_zoom_region()

    def _show_zoom_region(self):
        if self._zoom_start is None or self._zoom_end is None:
            return
        a = min(self._zoom_start, self._zoom_end)
        b = max(self._zoom_start, self._zoom_end)
        for reg in self.zoom_regions:
            reg.setRegion((a, b))
            reg.show()

    def _clear_zoom_region(self):
        self._zoom_start = None
        self._zoom_end = None
        for reg in self.zoom_regions:
            reg.hide()

    def _on_cursor_requested(self, x):
        if not self.series:
            return
        self._stop_playback_if_playing()
        self._set_cursor_time(clamp(x, self.t_global_min, self.t_global_max), True)

    def _on_zoom_double_click(self, x):
        if not self.series:
            return
        mods = QtWidgets.QApplication.keyboardModifiers()
        if mods & QtCore.Qt.ControlModifier:
            self._reset_view_full()
        else:
            self._center_on_time(x)

    def _center_on_time(self, x):
        if not self.series:
            return
        span = self.view_span
        total = max(1e-6, self.t_global_max - self.t_global_min)
        if span >= total:
            self._reset_view_full()
            self._set_cursor_time(clamp(x, self.t_global_min, self.t_global_max), True)
            return
        center = clamp(x, self.t_global_min + 0.5 * span, self.t_global_max - 0.5 * span)
        self._set_view_range(center - 0.5 * span, center + 0.5 * span, ensure_cursor=False)
        self._set_cursor_time(clamp(x, self.t_global_min, self.t_global_max), True)

    def _zoom_out(self):
        if not self.series:
            return
        self._stop_playback_if_playing()
        span = self.view_span * 2.0
        total = max(1e-6, self.t_global_max - self.t_global_min)
        if span >= total:
            self._reset_view_full()
            return
        center = self.view_start + 0.5 * self.view_span
        self._set_view_range(center - 0.5 * span, center + 0.5 * span, ensure_cursor=True)

    def _reset_view_full(self):
        if not self.series:
            return
        self._stop_playback_if_playing()
        self._set_view_range(self.t_global_min, self.t_global_max, ensure_cursor=True)
        self._update_status()

    def keyPressEvent(self, ev):
        key = ev.key()
        txt = ev.text().lower()

        if key == QtCore.Qt.Key_Space:
            self._toggle_playback()
            return
        if key in (QtCore.Qt.Key_BracketRight, QtCore.Qt.Key_PageDown):
            self._page(+1)
            return
        if key in (QtCore.Qt.Key_BracketLeft, QtCore.Qt.Key_PageUp):
            self._page(-1)
            return
        if txt == "o":
            self._zoom_out()
            return
        if txt == "r":
            self._reset_view_full()
            return

        super().keyPressEvent(ev)

    def _update_status(self, msg=None):
        info = []
        if self.series:
            info.append(f"{len(self.series)} traces")
            info.append(f"t=[{self.t_global_min:.2f},{self.t_global_max:.2f}]s")
        info.append(f"view=[{self.view_start:.2f},{self.view_end:.2f}]s")
        info.append(f"cursor={self.cursor_time:.3f}s")
        if self.is_playing and not msg:
            msg = "Playing..."
        if msg:
            info.append("| " + msg)
        self.status.showMessage("  ".join(info))

    def _on_load_video(self):
        self._stop_playback_if_playing()
        if cv2 is None:
            QtWidgets.QMessageBox.warning(
                self, "Video", "OpenCV (cv2) is not installed."
            )
            return
        vpath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select video file",
            filter="Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if not vpath:
            return

        self._video_is_open = False
        self.video_frame_times = None

        QtCore.QMetaObject.invokeMethod(
            self._video_worker,
            "open",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, vpath),
        )
        ft_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select frame_times.npy"
        )
        if not ft_path:
            self.video_frame_times = None
            QtWidgets.QMessageBox.information(
                self, "Video", "No frame_times provided; scrubbing disabled."
            )
            return
        try:
            ft = np.load(ft_path).astype(float)
            if ft.ndim != 1:
                raise ValueError("frame_times.npy must be 1-D")
            self.video_frame_times = ft
            self._update_status(f"Loaded frame_times ({len(ft)} frames).")
            self._request_initial_frame()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Frame times error", str(e))
            self.video_frame_times = None

    def _on_video_opened(self, ok, msg):
        if not ok:
            self._video_is_open = False
            QtWidgets.QMessageBox.warning(self, "Video", msg or "Failed to open.")
        else:
            self._video_is_open = True
            self._request_initial_frame()

    def _request_initial_frame(self):
        if self._video_is_open and self.video_frame_times is not None:
            self._set_cursor_time(self.cursor_time, update_slider=True)

    def _on_frame_ready(self, idx, qimg):
        if qimg is None or qimg.isNull():
            return
        pix = QtGui.QPixmap.fromImage(qimg)
        if pix.isNull():
            return
        scaled = pix.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def _show_help(self):
        self._stop_playback_if_playing()
        QtWidgets.QMessageBox.information(
            self,
            "Help",
            (
                "<b>Interactions</b><br>"
                "Click-drag in any trace to zoom to that interval.<br>"
                "Single click sets the cursor; double click recenters; Ctrl+double click resets view.<br>"
                "Mouse wheel zooms, middle-drag pans.<br><br>"
                "<b>Hotkeys</b><br>"
                "[ / ] (or PageUp/PageDown): page left/right<br>"
                "O: zoom out 2x<br>"
                "R: reset zoom<br>"
                "Space: toggle video playback<br>"
            ),
        )

    def closeEvent(self, ev):
        try:
            self._stop_playback_if_playing()
            QtCore.QMetaObject.invokeMethod(
                self._video_worker, "stop", QtCore.Qt.QueuedConnection
            )
            self._video_thread.quit()
            if not self._video_thread.wait(1000):
                self._video_thread.terminate()
        except Exception as e:
            print(f"ERROR: Exception during closeEvent: {e}")
        super().closeEvent(ev)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = SynapSeeApp()
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        if os.path.isdir(folder):
            try:
                w.load_series_folder(folder)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(w, "Auto load", str(exc))
        elif folder:
            QtWidgets.QMessageBox.warning(w, "Auto load", f"Not a folder: {folder}")
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
