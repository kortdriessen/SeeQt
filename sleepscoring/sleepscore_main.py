#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sleep Scorer (fast): multi-trace viewer with windowed rendering, video scrubbing,
page hotkeys, and cross-page draggable labeling.

pip install PySide6 pyqtgraph opencv-python numpy
"""

import os, sys, glob, csv, math, argparse
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

try:
    import cv2
except Exception:
    cv2 = None


# ---------------- Data containers ----------------


@dataclass
class Series:
    name: str
    t: np.ndarray  # seconds, monotonic
    y: np.ndarray


# ---------------- Utilities ----------------


def nice_time_range(t_arrays):
    vals = [(np.nanmin(t), np.nanmax(t)) for t in t_arrays if t is not None and len(t)]
    return (
        (0.0, 1.0)
        if not vals
        else (float(min(v[0] for v in vals)), float(max(v[1] for v in vals)))
    )


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


def next_pow_two(n: int) -> int:
    n = int(max(1, n))
    return 1 << (n - 1).bit_length()


# ---------------- Peak-preserving window decimator ----------------


def segment_for_window(t, y, t0, t1, max_pts=4000):
    """
    Return (tx, yx) for the [t0, t1] window.
    Uses peak-preserving bin min/max if the window contains too many samples.
    """
    if t1 <= t0:
        return np.empty(0), np.empty(0)

    # 1) slice to window (with 1-sample guard on each side)
    i0 = max(0, np.searchsorted(t, t0) - 1)
    i1 = min(len(t), np.searchsorted(t, t1) + 1)
    ts = t[i0:i1]
    ys = y[i0:i1]
    n = len(ts)
    if n <= 2:
        return ts, ys

    # 2) if already small, return as-is
    if n <= max_pts:
        return ts, ys

    # 3) bin across time into ~max_pts/2 bins; emit min/max per bin
    bins = max(1, max_pts // 2)
    # bin edges across [t0, t1]
    edges = np.linspace(t0, t1, bins + 1)
    # assign each timestamp to a bin index (0..bins-1)
    bi = np.clip(np.digitize(ts, edges) - 1, 0, bins - 1)

    # sort by bin to compute mins/maxes with slice operations
    order = np.argsort(bi, kind="mergesort")  # stable
    bi_s = bi[order]
    ts_s = ts[order]
    ys_s = ys[order]
    # boundaries of each bin in sorted array
    starts = np.searchsorted(bi_s, np.arange(bins), "left")
    ends = np.searchsorted(bi_s, np.arange(bins), "right")

    out_t = np.empty(2 * bins, dtype=float)
    out_y = np.empty(2 * bins, dtype=float)
    k = 0
    for b in range(bins):
        s, e = starts[b], ends[b]
        if s == e:  # empty bin -> fill from edge
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
            # interleave min/max at the same nominal time (gives vertical spikes)
            out_t[k] = tmid
            out_y[k] = ymin
            k += 1
            out_t[k] = tmid
            out_y[k] = ymax
            k += 1

    return out_t[:k], out_y[:k]


# ---------------- Custom UI Components ----------------


class SelectableViewBox(pg.ViewBox):
    sigDragStart = QtCore.Signal(float)
    sigDragUpdate = QtCore.Signal(float)
    sigDragFinish = QtCore.Signal(float)
    sigWheelScrolled = QtCore.Signal(int)
    sigWheelSmoothScrolled = QtCore.Signal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # *** FIX 1: Explicitly disable default mouse pan/zoom behavior ***
        # This allows our custom event handlers to take full control.
        self.setMouseEnabled(x=False, y=False)
        self._drag = False

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            x = float(self.mapSceneToView(ev.scenePos()).x())
            if ev.isStart():
                self._drag = True
                self.sigDragStart.emit(x)
                ev.accept()
                return
            elif ev.isFinish():
                if self._drag:
                    self.sigDragFinish.emit(x)
                self._drag = False
                ev.accept()
                return
            else:
                if self._drag:
                    self.sigDragUpdate.emit(x)
                ev.accept()
                return
        # Do not call super, to prevent default drag (pan) behavior

    def wheelEvent(self, ev, axis=None):
        dy = 0
        if hasattr(ev, "delta"):
            try:
                dy = ev.delta()
            except Exception:
                dy = 0
        else:
            try:
                ad = ev.angleDelta()
                dy = ad.y() if hasattr(ad, "y") else 0
            except Exception:
                dy = 0
        direction = 1 if dy > 0 else -1
        # Use Shift+wheel for smooth scrolling; otherwise page
        try:
            mods = QtWidgets.QApplication.keyboardModifiers()
        except Exception:
            mods = QtCore.Qt.KeyboardModifier.NoModifier
        if mods & QtCore.Qt.KeyboardModifier.ShiftModifier:
            self.sigWheelSmoothScrolled.emit(direction)
        else:
            self.sigWheelScrolled.emit(direction)
        ev.accept()


# *** FIX 2: Create a PlotItem that signals when the mouse enters/leaves it ***
class HoverablePlotItem(pg.PlotItem):
    sigHovered = QtCore.Signal(
        object, bool
    )  # Emits self, True on enter, False on leave

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Required to receive hover events
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, ev):
        self.sigHovered.emit(self, True)
        super().hoverEnterEvent(ev)

    def hoverLeaveEvent(self, ev):
        self.sigHovered.emit(self, False)
        super().hoverLeaveEvent(ev)


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

        qimg = self.cache.get(idx, None)
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


class YAxisControlsDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Y-Axis Controls")
        self.setModal(False)

        self.main_window = parent

        main_layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        main_layout.addLayout(form_layout)

        self.controls = []
        for idx, s in enumerate(self.main_window.series):
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            auto_check = QtWidgets.QCheckBox("Auto")
            auto_check.setChecked(
                self.main_window.plots[idx].getViewBox().autoRangeEnabled()[1]
            )

            min_spin = QtWidgets.QDoubleSpinBox()
            max_spin = QtWidgets.QDoubleSpinBox()

            for spin in (min_spin, max_spin):
                spin.setDecimals(3)
                spin.setRange(-1e12, 1e12)
                spin.setEnabled(not auto_check.isChecked())

            current_y_range = self.main_window.plots[idx].getViewBox().viewRange()[1]
            min_spin.setValue(current_y_range[0])
            max_spin.setValue(current_y_range[1])

            auto_check.stateChanged.connect(
                lambda state, i=idx, ac=auto_check, mn=min_spin, mx=max_spin: self.apply_y_range(
                    i, ac, mn, mx
                )
            )
            min_spin.editingFinished.connect(
                lambda i=idx, ac=auto_check, mn=min_spin, mx=max_spin: self.apply_y_range(
                    i, ac, mn, mx
                )
            )
            max_spin.editingFinished.connect(
                lambda i=idx, ac=auto_check, mn=min_spin, mx=max_spin: self.apply_y_range(
                    i, ac, mn, mx
                )
            )

            row_layout.addStretch(1)
            row_layout.addWidget(auto_check)
            row_layout.addWidget(QtWidgets.QLabel("Min"))
            row_layout.addWidget(min_spin)
            row_layout.addWidget(QtWidgets.QLabel("Max"))
            row_layout.addWidget(max_spin)

            form_layout.addRow(s.name, row_widget)
            self.controls.append((auto_check, min_spin, max_spin))

    def apply_y_range(self, plot_index, auto_check, min_spin, max_spin):
        plot_item = self.main_window.plots[plot_index]
        if auto_check.isChecked():
            plot_item.enableAutoRange("y", True)
            min_spin.setEnabled(False)
            max_spin.setEnabled(False)
        else:
            plot_item.enableAutoRange("y", False)
            lo, hi = min_spin.value(), max_spin.value()
            if hi <= lo:
                hi = lo + 1e-6
            plot_item.setYRange(lo, hi, padding=0.05)
            min_spin.setEnabled(True)
            max_spin.setEnabled(True)


# ---------------- Main window ----------------


class SleepScorerApp(QtWidgets.QMainWindow):
    def __init__(
        self,
        data_dir=None,
        data_files=None,
        colors=None,
        video_path=None,
        frame_times_path=None,
        video2_path=None,
        frame_times2_path=None,
        image_path=None,
        fixed_scale=False,
        low_profile_x=False,
    ):
        super().__init__()
        self.setWindowTitle("Sleep Scorer — Multi-Trace + Video + Labeling")
        self.resize(1400, 900)

        pg.setConfigOptions(
            antialias=False, useOpenGL=True, background="k", foreground="w"
        )

        # Data & plots
        self.series: list[Series] = []
        self.t_global_min = 0.0
        self.t_global_max = 1.0
        self.plots: list[pg.PlotItem] = []
        self.curves: list[pg.PlotDataItem] = []
        self.plot_cur_lines: list[pg.InfiniteLine] = []
        self.plot_sel_regions: list[pg.LinearRegionItem] = []
        self.plot_label_regions: list[list[pg.LinearRegionItem]] = []
        self.hovered_plot = None  # *** FIX 2: Track which plot is hovered ***

        # Rendering budget (per plot)
        self.max_pts_per_plot = 4000

        # Window/cursor & labels
        self.window_len = 10.0
        self.window_start = 0.0
        self.cursor_time = 0.0

        self.keymap = {
            "w": "Wake",
            "q": "Quiet-Wake",
            "b": "Brief-Arousal",
            "2": "NREM-light",
            "1": "NREM",
            "r": "REM",
            "a": "Artifact",
            "t": "Transition-to-REM",
        }
        self.label_colors = {
            "Wake": (0, 209, 40, 60),
            "Quiet-Wake": (79, 255, 168, 60),
            "Brief-Arousal": (188, 255, 45, 60),
            "NREM-light": (79, 247, 255, 60),
            "NREM": (41, 30, 255, 60),
            "Transition-to-REM": (255, 101, 224, 60),
            "REM": (255, 30, 145, 60),
            "Artifact": (255, 0, 0, 80),
        }
        self.labels = []
        self._select_start = None
        self._select_end = None
        self._is_zoom_drag = False
        self.fixed_scale = bool(fixed_scale)
        self.low_profile_x = bool(low_profile_x)

        # Video
        self.video_frame_times = None
        self._video_is_open = False
        self._video_thread = QtCore.QThread(self)
        self._video_worker = VideoWorker(cache_frames=120)
        self._video_worker.moveToThread(self._video_thread)
        self.last_video_pixmap = None

        # Optional second video
        self.video2_frame_times = None
        self._video2_is_open = False
        self._video2_thread = QtCore.QThread(self)
        self._video2_worker = VideoWorker(cache_frames=120)
        self._video2_worker.moveToThread(self._video2_thread)
        self.last_video2_pixmap = None

        # Playback
        self.is_playing = False
        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.timeout.connect(self._advance_playback_frame)
        self.playback_elapsed_timer = QtCore.QElapsedTimer()

        # Smooth scroll settings (fraction of window per wheel step)
        self.smooth_scroll_fraction = 0.10

        # Static Image (hidden if second video is loaded)
        self.static_image_pixmap = None

        # Hypnogram overview
        self.hypnogram_widget = None
        self.hypnogram_plot = None
        self.hypnogram_view_region = None
        self.hypnogram_label_regions = []
        self.hypnogram_zoomed = False
        self.hypnogram_zoom_padding = 30.0

        self._build_ui()

        self.y_axis_dialog = None

        self._video_worker.frameReady.connect(self._on_frame_ready)
        self._video_worker.opened.connect(self._on_video_opened)
        self._video2_worker.frameReady.connect(self._on_frame2_ready)
        self._video2_worker.opened.connect(self._on_video2_opened)

        self._video_thread.start()
        self._video2_thread.start()
        # Prefer explicit file list if provided; otherwise fall back to dir
        if data_files:
            # Allow flexible formats: list[str], comma-separated strings, or "[a,b]".
            def _normalize_file_list(df):
                if not df:
                    return []
                items = [df] if isinstance(df, str) else list(df)
                out = []
                for it in items:
                    s = (it or "").strip()
                    # Strip surrounding list brackets if present
                    if s.startswith("[") and s.endswith("]"):
                        s = s[1:-1]
                    parts = s.split(",") if "," in s else [s]
                    for p in parts:
                        q = p.strip().strip('"').strip("'")
                        # Remove a lingering trailing comma if passed as a token like "file.npy,"
                        if q.endswith(","):
                            q = q[:-1].rstrip()
                        if q:
                            out.append(q)
                return out

            def _normalize_list(raw_list):
                if not raw_list:
                    return []
                items = [raw_list] if isinstance(raw_list, str) else list(raw_list)
                out = []
                for it in items:
                    s = (it or "").strip()
                    if s.startswith("[") and s.endswith("]"):
                        s = s[1:-1]
                    parts = s.split(",") if "," in s else [s]
                    for p in parts:
                        q = p.strip().strip('"').strip("'")
                        if q.endswith(","):
                            q = q[:-1].rstrip()
                        if q:
                            out.append(q)
                return out

            paths = _normalize_file_list(data_files)
            color_list = _normalize_list(colors) if colors else None
            self._load_series_from_files(paths, colors=color_list)
        elif data_dir:
            self._load_series_from_dir(data_dir)
        if video_path and frame_times_path:
            self._load_video_data(video_path, frame_times_path)
        if video2_path and frame_times2_path:
            self._load_video2_data(video2_path, frame_times2_path)
        elif image_path:
            self._load_static_image(image_path)

    def eventFilter(self, obj, ev):
        try:
            if ev.type() == QtCore.QEvent.Type.Resize:
                if obj is self.static_image_label:
                    self._rescale_static_image()
                elif obj is self.video_label:
                    self._rescale_video_frame()
                elif obj is self.video2_label:
                    self._rescale_video2_frame()
        except Exception:
            pass
        return super().eventFilter(obj, ev)

    # ---------- UI ----------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        v.addLayout(top)
        top.addWidget(QtWidgets.QLabel("Window (s):"))
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(0.1, 3600.0)
        self.window_spin.setDecimals(2)
        self.window_spin.setValue(self.window_len)
        self.window_spin.valueChanged.connect(self._on_window_len_changed)
        top.addWidget(self.window_spin)
        top.addSpacing(20)
        top.addWidget(QtWidgets.QLabel("Navigate:"))
        self.nav_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.nav_slider.setRange(0, 10000)
        self.nav_slider.valueChanged.connect(self._on_nav_slider_changed)
        top.addWidget(self.nav_slider, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        v.addWidget(splitter, 1)
        self.splitter = splitter

        # left plots
        left = QtWidgets.QWidget()
        leftl = QtWidgets.QVBoxLayout(left)
        leftl.setContentsMargins(0, 0, 0, 0)
        self.plot_area = pg.GraphicsLayoutWidget()
        leftl.addWidget(self.plot_area, 1)
        splitter.addWidget(left)

        # right side
        right = QtWidgets.QWidget()
        right.setMinimumWidth(150)
        rl = QtWidgets.QVBoxLayout(right)
        self.video_label = QtWidgets.QLabel("No video")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(240)
        self.video_label.setStyleSheet("background-color:#222;border:1px solid #444;")
        rl.addWidget(self.video_label, 3)
        self.video_label.installEventFilter(self)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Cursor:"))
        self.window_cursor_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.window_cursor_slider.setRange(0, 10000)
        self.window_cursor_slider.valueChanged.connect(self._on_window_cursor_changed)
        row.addWidget(self.window_cursor_slider)

        roww = QtWidgets.QWidget()
        roww.setLayout(row)
        rl.addWidget(roww)

        # Second video label (replaces image if video2 is loaded)
        self.video2_label = QtWidgets.QLabel("No video 2")
        self.video2_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video2_label.setMinimumHeight(200)
        self.video2_label.setStyleSheet("background-color:#222;border:1px solid #444;")
        self.video2_label.hide()
        rl.addWidget(self.video2_label, 2)
        self.video2_label.installEventFilter(self)

        self.static_image_label = QtWidgets.QLabel("No image loaded")
        self.static_image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.static_image_label.setStyleSheet(
            "background-color:#222;border:1px solid #444;"
        )
        rl.addWidget(self.static_image_label, 2)
        self.static_image_label.installEventFilter(self)

        # Hypnogram overview plot (full-recording labels with moving window box)
        self.hypnogram_widget = pg.PlotWidget()
        self.hypnogram_widget.setMinimumHeight(90)
        hp = self.hypnogram_widget.getPlotItem()
        hp.showGrid(x=False, y=False)
        hp.hideAxis("left")
        hp.setMenuEnabled(False)
        hp.setMouseEnabled(x=False, y=False)
        hp.enableAutoRange("y", False)
        hp.setYRange(0, 1)
        self.hypnogram_plot = hp
        rl.addWidget(self.hypnogram_widget, 1)

        # Region showing the current window on the hypnogram
        self.hypnogram_view_region = pg.LinearRegionItem(
            values=(self.window_start, self.window_start + self.window_len),
            brush=pg.mkBrush(255, 255, 255, 50),
            movable=False,
        )
        self.hypnogram_view_region.setZValue(20)
        self.hypnogram_plot.addItem(self.hypnogram_view_region)

        # Ensure rescale happens when the splitter is adjusted
        self.splitter.splitterMoved.connect(self._on_splitter_moved)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.status = self.statusBar()
        self._update_status()
        self._build_menu()

    def _build_menu(self):
        mfile = self.menuBar().addMenu("&File")
        a = QtGui.QAction("Load &Time Series…", self)
        a.triggered.connect(self._on_load_time_series)
        mfile.addAction(a)
        b = QtGui.QAction("Load &Video && Frame Times…", self)
        b.triggered.connect(self._on_load_video)
        mfile.addAction(b)
        mfile.addSeparator()

        c = QtGui.QAction("Load &Labels…", self)
        c.triggered.connect(self._on_load_labels)
        mfile.addAction(c)

        d = QtGui.QAction("&Export Labels…", self)
        d.triggered.connect(self._on_export_labels)
        mfile.addAction(d)
        mfile.addSeparator()

        q = QtGui.QAction("&Quit", self)
        q.triggered.connect(self.close)
        mfile.addAction(q)

        medit = self.menuBar().addMenu("&Edit")
        clr = QtGui.QAction("Clear current selection", self)
        clr.triggered.connect(self._clear_selection)
        medit.addAction(clr)
        dl = QtGui.QAction("Delete last label", self)
        dl.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Backspace))
        dl.triggered.connect(self._delete_last_label)
        medit.addAction(dl)

        mview = self.menuBar().addMenu("&View")
        y_axis_action = QtGui.QAction("Y-Axis Controls...", self)
        y_axis_action.setShortcut(QtGui.QKeySequence("Ctrl+D"))
        y_axis_action.triggered.connect(self._show_y_axis_dialog)
        mview.addAction(y_axis_action)

        scroll_speed_action = QtGui.QAction("Adjust Smooth Scroll Speed...", self)
        scroll_speed_action.triggered.connect(self._adjust_scroll_speed)
        mview.addAction(scroll_speed_action)

        mhelp = self.menuBar().addMenu("&Help")
        hh = QtGui.QAction("Shortcuts / Help", self)
        hh.triggered.connect(self._show_help)
        mhelp.addAction(hh)

    def _adjust_scroll_speed(self):
        try:
            val, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Adjust Smooth Scroll Speed",
                "Fraction of window per wheel step (0.001 - 1.0):",
                float(self.smooth_scroll_fraction),
                0.001,
                1.0,
                3,
            )
        except Exception:
            val, ok = (self.smooth_scroll_fraction, False)
        if ok:
            self.smooth_scroll_fraction = float(max(0.001, min(1.0, val)))

    # ---------- Data ----------
    def _load_series_from_dir(self, folder):
        self._stop_playback_if_playing()
        pairs = []
        for tpath in glob.glob(os.path.join(folder, "*_t.npy")):
            name = os.path.basename(tpath)[:-6]
            ypath = os.path.join(folder, f"{name}_y.npy")
            if os.path.exists(ypath):
                pairs.append((name, tpath, ypath))
        if not pairs:
            QtWidgets.QMessageBox.warning(
                self, "No data", "No *_t.npy / *_y.npy pairs found."
            )
            return

        series = []
        for name, tpath, ypath in sorted(pairs):
            try:
                t = np.load(tpath).astype(float)
                y = np.load(ypath).astype(float)
                if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
                    raise ValueError("t and y must be 1-D & equal length")
                series.append(Series(name, t, y))
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Load error", f"{name}: {e}")
        if series:
            self.set_series(series)

    def _load_series_from_files(self, files, colors=None):
        """Load series from an explicit ordered list of *_t.npy / *_y.npy files.

        The display order (top to bottom) follows the order in which distinct
        base names first appear in the provided list. A "base name" is the
        filename without the trailing "_t.npy" or "_y.npy".
        """
        self._stop_playback_if_playing()

        if not files:
            QtWidgets.QMessageBox.warning(self, "No data", "No files provided.")
            return

        # Build mapping base_name -> { 't': path or None, 'y': path or None }
        series_map = {}
        order = []  # first-seen order of base names
        first_index = {}  # base name -> first index in files list

        def base_for(path: str):
            fn = os.path.basename(path)
            if fn.endswith("_t.npy"):
                return fn[:-6], "t"
            if fn.endswith("_y.npy"):
                return fn[:-6], "y"
            return None, None

        for idx, p in enumerate(files):
            if not p:
                continue
            b, kind = base_for(p)
            if b is None:
                QtWidgets.QMessageBox.warning(
                    self, "Skip", f"Not a *_t.npy or *_y.npy file: {p}"
                )
                continue
            if b not in series_map:
                series_map[b] = {"t": None, "y": None}
                order.append(b)
                first_index[b] = idx
            series_map[b][kind] = p

        # Assemble in the order seen; require both t and y
        series = []
        series_colors = []
        for b in order:
            paths = series_map.get(b, {})
            tpath, ypath = paths.get("t"), paths.get("y")
            if not tpath or not ypath:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing pair",
                    f"Skipping '{b}': need both {b}_t.npy and {b}_y.npy",
                )
                continue
            if not (os.path.exists(tpath) and os.path.exists(ypath)):
                QtWidgets.QMessageBox.warning(
                    self, "File not found", f"Missing files for '{b}'."
                )
                continue
            try:
                t = np.load(tpath).astype(float)
                y = np.load(ypath).astype(float)
                if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
                    raise ValueError("t and y must be 1-D & equal length")
                series.append(Series(b, t, y))
                series_colors.append(None)  # placeholder
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Load error", f"{b}: {e}")

        if not series:
            QtWidgets.QMessageBox.warning(
                self, "No data", "No valid *_t.npy / *_y.npy pairs in list."
            )
            return

        # Map provided colors to series order
        def _parse_color(cs: str):
            s = (cs or "").strip()
            try:
                if s.startswith("#"):
                    s = s[1:]
                    if len(s) in (6, 8):
                        r = int(s[0:2], 16)
                        g = int(s[2:4], 16)
                        b = int(s[4:6], 16)
                        a = int(s[6:8], 16) if len(s) == 8 else 255
                        return (r, g, b, a)
                if s.lower().startswith("0x"):
                    v = int(s, 16)
                    r = (v >> 16) & 0xFF
                    g = (v >> 8) & 0xFF
                    b = v & 0xFF
                    return (r, g, b, 255)
                if "," in s:
                    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
                    if len(parts) == 3:
                        return (parts[0], parts[1], parts[2], 255)
                    if len(parts) >= 4:
                        return (parts[0], parts[1], parts[2], parts[3])
            except Exception:
                pass
            return None

        mapped_colors = None
        if colors:
            # If colors count matches series count, map 1:1
            if len(colors) == len(series):
                mapped_colors = [
                    (_parse_color(c) or (255, 255, 255, 255)) for c in colors
                ]
            # If colors matches files count, use first occurrence index mapping
            elif len(colors) == len(files):
                mapped_colors = []
                for b in order:
                    ci = first_index.get(b, 0)
                    col = _parse_color(colors[ci]) or (255, 255, 255, 255)
                    mapped_colors.append(col)
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Colors",
                    (
                        f"Ignoring --colors: count {len(colors)} doesn't match series ({len(series)}) "
                        f"or file count ({len(files)})."
                    ),
                )

        # Store colors aligned with series; default to white if not provided
        self.series_colors = mapped_colors or [(255, 255, 255, 255)] * len(series)
        self.set_series(series)

    def _on_load_time_series(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder with *_t.npy and *_y.npy"
        )
        if folder:
            self._load_series_from_dir(folder)

    def set_series(self, series_list):
        self.series = series_list

        self.plot_area.clear()
        self.plots.clear()
        self.curves.clear()
        self.plot_cur_lines.clear()
        self.plot_sel_regions.clear()
        self.plot_label_regions.clear()

        self.t_global_min, self.t_global_max = nice_time_range(
            [s.t for s in self.series]
        )
        self.window_start = self.t_global_min
        self.cursor_time = self.window_start

        master_plot = None
        for idx, s in enumerate(self.series):
            vb = SelectableViewBox()
            vb.sigWheelScrolled.connect(self._page)
            vb.sigWheelSmoothScrolled.connect(self._on_smooth_scroll)
            vb.sigWheelSmoothScrolled.connect(self._on_smooth_scroll)

            # *** FIX 2: Use HoverablePlotItem to track mouse location ***
            plt = HoverablePlotItem(viewBox=vb)
            plt.sigHovered.connect(self._on_plot_hovered)

            # Manually add the plot to the layout instead of using addPlot
            self.plot_area.addItem(plt, row=idx, col=0)

            plt.setLabel("left", s.name)
            plt.setLabel(
                "bottom", "Time", units="s" if idx == len(self.series) - 1 else None
            )
            # Always enable both grids; for low_profile_x we hide axis visuals but keep grid
            plt.showGrid(x=True, y=True, alpha=0.15)
            plt.addLegend(offset=(10, 10))
            plt.enableAutoRange("x", False)

            if self.low_profile_x and idx != len(self.series) - 1:
                try:
                    # Keep the axis and spine visible, remove text/ticks, keep grid
                    plt.setLabel("bottom", "")
                    bax = plt.getAxis("bottom")
                    bax.setStyle(showValues=True, tickLength=0)
                    bax.setTextPen(pg.mkPen(0, 0, 0, 0))  # hide tick text
                    bax.setHeight(12)
                    try:
                        bax.setGrid(True, alpha=0.15)
                    except Exception:
                        pass
                except Exception:
                    pass

            # Make tick labels slightly smaller for consistency and tighter margin
            try:
                lf = QtGui.QFont()
                lf.setPointSize(9)
                plt.getAxis("left").setStyle(tickFont=lf)
            except Exception:
                pass

            # Choose color: use provided series_colors or default to white
            if getattr(self, "series_colors", None) and idx < len(self.series_colors):
                pen_color = self.series_colors[idx]
            else:
                pen_color = (255, 255, 255)
            pen = pg.mkPen(pen_color, width=1)
            curve = pg.PlotDataItem([], [], pen=pen, antialias=False)
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
            plt.addItem(curve)

            cur_line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 120))
            )
            plt.addItem(cur_line)

            sel_region = pg.LinearRegionItem(
                values=(0, 0), brush=pg.mkBrush(100, 200, 255, 40), movable=True
            )
            sel_region.setZValue(-10)
            sel_region.hide()
            plt.addItem(sel_region)
            sel_region.sigRegionChanged.connect(self._on_active_region_dragged)

            self.plots.append(plt)
            self.curves.append(curve)
            self.plot_cur_lines.append(cur_line)
            self.plot_sel_regions.append(sel_region)
            self.plot_label_regions.append([])

            if master_plot is None:
                master_plot = plt
            else:
                plt.setXLink(master_plot)

            vb.sigDragStart.connect(self._on_drag_start)
            vb.sigDragUpdate.connect(self._on_drag_update)
            vb.sigDragFinish.connect(self._on_drag_finish)

            # Y-axis behavior: either auto or fixed scale from data
            if self.fixed_scale:
                try:
                    y = np.asarray(s.y, dtype=float)
                    # robust limits to avoid outliers
                    lo = float(np.nanpercentile(y, 1.0))
                    hi = float(np.nanpercentile(y, 99.0))
                    if not np.isfinite(lo) or not np.isfinite(hi):
                        raise ValueError("non-finite percentiles")
                    if hi <= lo:
                        hi = lo + 1.0
                    # small pad
                    pad = 0.05 * (hi - lo)
                    plt.enableAutoRange("y", False)
                    plt.setYRange(lo - pad, hi + pad, padding=0)
                except Exception:
                    plt.enableAutoRange("y", False)
            else:
                plt.enableAutoRange("y", True)

        self._apply_x_range()
        self._update_nav_slider_from_window()
        self._update_status(f"Loaded {len(self.series)} series.")
        self._update_hypnogram_extents()
        # Align left axes after layout settles
        QtCore.QTimer.singleShot(0, self._align_left_axes)
        QtCore.QTimer.singleShot(100, self._align_left_axes)

    # ---------- Video & Static Image ----------
    def _load_video_data(self, vpath, ft_path):
        self._stop_playback_if_playing()
        self._video_is_open = False
        self.video_frame_times = None

        if not os.path.exists(vpath) or not os.path.exists(ft_path):
            QtWidgets.QMessageBox.warning(
                self, "File Not Found", "Video or frame times file does not exist."
            )
            return

        QtCore.QMetaObject.invokeMethod(
            self._video_worker,
            "open",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, vpath),
        )
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

    def _on_load_video(self):
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

        ft_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select frame_times.npy"
        )
        if not ft_path:
            return

        self._load_video_data(vpath, ft_path)

    # ---------- Video2 ----------
    def _load_video2_data(self, vpath, ft_path):
        self._stop_playback_if_playing()
        self._video2_is_open = False
        self.video2_frame_times = None

        if not os.path.exists(vpath) or not os.path.exists(ft_path):
            QtWidgets.QMessageBox.warning(
                self, "File Not Found", "Video2 or frame times file does not exist."
            )
            return

        QtCore.QMetaObject.invokeMethod(
            self._video2_worker,
            "open",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, vpath),
        )
        try:
            ft = np.load(ft_path).astype(float)
            if ft.ndim != 1:
                raise ValueError("frame_times.npy must be 1-D")
            self.video2_frame_times = ft
            self._update_status(f"Loaded frame_times2 ({len(ft)} frames).")
            self.static_image_label.hide()
            self.video2_label.show()
            self._request_initial_frame()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Frame times 2 error", str(e))
            self.video2_frame_times = None

    def _on_video2_opened(self, ok, msg):
        if not ok:
            self._video2_is_open = False
            QtWidgets.QMessageBox.warning(self, "Video2", msg or "Failed to open.")
        else:
            self._video2_is_open = True
            self.static_image_label.hide()
            self.video2_label.show()
            self._request_initial_frame()

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

        self.last_video_pixmap = pix
        self._rescale_video_frame()

    def _on_frame2_ready(self, idx, qimg):
        if qimg is None or qimg.isNull():
            return
        pix = QtGui.QPixmap.fromImage(qimg)
        if pix.isNull():
            return
        self.last_video2_pixmap = pix
        self._rescale_video2_frame()

    def _rescale_video_frame(self):
        if self.last_video_pixmap:
            scaled = self.last_video_pixmap.scaled(
                self.video_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.video_label.setPixmap(scaled)

    def _rescale_video2_frame(self):
        if self.last_video2_pixmap:
            scaled = self.last_video2_pixmap.scaled(
                self.video2_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.video2_label.setPixmap(scaled)

    def _load_static_image(self, path):
        if not os.path.exists(path):
            self.static_image_label.setText("Image not found")
            return
        pixmap = QtGui.QPixmap(path)
        if pixmap.isNull():
            self.static_image_label.setText("Failed to load image")
            return
        self.static_image_pixmap = pixmap
        self.static_image_label.setText("")
        self._rescale_static_image()

    def _on_splitter_moved(self, pos, index):
        self._rescale_video_frame()
        self._rescale_static_image()

    def _rescale_static_image(self):
        if self.static_image_pixmap:
            scaled = self.static_image_pixmap.scaled(
                self.static_image_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.static_image_label.setPixmap(scaled)

    # ---------- Selection / labeling ----------
    def _show_y_axis_dialog(self):
        if not self.series:
            QtWidgets.QMessageBox.information(
                self, "Y-Axis Controls", "Load time series data first."
            )
            return
        if self.y_axis_dialog is not None:
            self.y_axis_dialog.deleteLater()

        self.y_axis_dialog = YAxisControlsDialog(self)
        self.y_axis_dialog.show()
        self.y_axis_dialog.raise_()
        self.y_axis_dialog.activateWindow()

    def _on_plot_hovered(self, plot, is_hovered):
        if is_hovered:
            self.hovered_plot = plot
        else:
            if self.hovered_plot is plot:
                self.hovered_plot = None

    def _on_drag_start(self, x):
        self._stop_playback_if_playing()
        self._select_start = x
        self._select_end = x
        # Determine if this drag is a zoom gesture (Shift held)
        try:
            mods = QtWidgets.QApplication.keyboardModifiers()
        except Exception:
            mods = QtCore.Qt.KeyboardModifier.NoModifier
        self._is_zoom_drag = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        self._show_active_selection()

    def _on_drag_update(self, x):
        self._select_end = x
        self._show_active_selection()

    def _on_drag_finish(self, x):
        self._select_end = x
        # If this was a Shift+drag, zoom to the selected time range
        if self._is_zoom_drag:
            a = float(min(self._select_start, self._select_end))
            b = float(max(self._select_start, self._select_end))
            if b > a:
                new_len = max(0.1, b - a)
                self.window_len = new_len
                self.window_start = clamp(
                    a,
                    self.t_global_min,
                    max(self.t_global_min, self.t_global_max - self.window_len),
                )
                # Sync UI without triggering change handler
                self.window_spin.blockSignals(True)
                self.window_spin.setValue(self.window_len)
                self.window_spin.blockSignals(False)
                self._apply_x_range()
                self._update_nav_slider_from_window()
            self._is_zoom_drag = False
            self._clear_selection()
            return
        self._show_active_selection(final=True)

    def _on_active_region_dragged(self):
        self._stop_playback_if_playing()
        if not self.plot_sel_regions:
            return
        a, b = self.plot_sel_regions[0].getRegion()
        self._select_start, self._select_end = float(a), float(b)

    def _show_active_selection(self, final=False):
        if self._select_start is None or self._select_end is None:
            for r in self.plot_sel_regions:
                r.hide()
            return
        a = min(self._select_start, self._select_end)
        b = max(self._select_start, self._select_end)
        for r in self.plot_sel_regions:
            r.setRegion((a, b))
            r.show()

    def _clear_selection(self):
        self._select_start = None
        self._select_end = None
        for r in self.plot_sel_regions:
            r.hide()

    def _add_new_label(self, start, end, label):
        """Adds new label, overwriting/modifying existing ones in the range."""
        updated_labels = []

        for existing in self.labels:
            ex_start, ex_end = existing["start"], existing["end"]

            overlap_start = max(ex_start, start)
            overlap_end = min(ex_end, end)

            if overlap_start >= overlap_end:
                updated_labels.append(existing)
                continue

            if ex_start < start and ex_end > end:
                updated_labels.append(
                    {"start": ex_start, "end": start, "label": existing["label"]}
                )
                updated_labels.append(
                    {"start": end, "end": ex_end, "label": existing["label"]}
                )
            elif ex_start < start:
                updated_labels.append(
                    {"start": ex_start, "end": start, "label": existing["label"]}
                )
            elif ex_end > end:
                updated_labels.append(
                    {"start": end, "end": ex_end, "label": existing["label"]}
                )

        updated_labels.append({"start": start, "end": end, "label": label})
        self.labels = sorted(updated_labels, key=lambda x: x["start"])
        self._merge_adjacent_same_labels()
        self._redraw_all_labels()

    def _merge_adjacent_same_labels(self, adjacency_eps: float = 1e-9):
        if not self.labels:
            return
        merged = []
        for lab in sorted(self.labels, key=lambda x: x["start"]):
            if not merged:
                merged.append(
                    {
                        "start": float(lab["start"]),
                        "end": float(lab["end"]),
                        "label": lab["label"],
                    }
                )
                continue
            prev = merged[-1]
            if (
                lab["label"] == prev["label"]
                and float(lab["start"]) <= float(prev["end"]) + adjacency_eps
            ):
                prev["end"] = max(float(prev["end"]), float(lab["end"]))
            else:
                merged.append(
                    {
                        "start": float(lab["start"]),
                        "end": float(lab["end"]),
                        "label": lab["label"],
                    }
                )
        self.labels = merged

    def _redraw_all_labels(self):
        """Clears and redraws all visual label regions."""
        for plot_regions in self.plot_label_regions:
            for item in plot_regions:
                if item.scene():
                    item.scene().removeItem(item)
            plot_regions.clear()

        for label_data in self.labels:
            a, b, name = label_data["start"], label_data["end"], label_data["label"]
            color = self.label_colors.get(name, (150, 150, 150, 80))

            for i, plt in enumerate(self.plots):
                reg = pg.LinearRegionItem(
                    values=(a, b), brush=pg.mkBrush(*color), movable=False
                )
                reg.setZValue(-20)
                plt.addItem(reg)
                self.plot_label_regions[i].append(reg)

        self._redraw_hypnogram_labels()

    def _redraw_hypnogram_labels(self):
        if self.hypnogram_plot is None:
            return
        # Clear previous regions
        for item in self.hypnogram_label_regions:
            try:
                if item.scene():
                    item.scene().removeItem(item)
            except Exception:
                pass
        self.hypnogram_label_regions.clear()

        # Draw label spans across full height
        for label_data in self.labels:
            a, b, name = label_data["start"], label_data["end"], label_data["label"]
            color = self.label_colors.get(name, (150, 150, 150, 80))
            reg = pg.LinearRegionItem(
                values=(a, b), brush=pg.mkBrush(*color), movable=False
            )
            reg.setZValue(-10)
            self.hypnogram_plot.addItem(reg)
            self.hypnogram_label_regions.append(reg)

    def _update_hypnogram_extents(self):
        if self.hypnogram_plot is None:
            return
        # Keep current zoom mode when extents change
        if not self.hypnogram_zoomed:
            self.hypnogram_plot.enableAutoRange("x", False)
            self.hypnogram_plot.setXRange(
                self.t_global_min, self.t_global_max, padding=0
            )
        else:
            self._update_hypnogram_xrange()
        # Ensure the view region reflects current window
        if self.hypnogram_view_region is not None:
            a = self.window_start
            b = self.window_start + self.window_len
            self.hypnogram_view_region.setRegion((a, b))

    def _delete_last_label(self):
        if not self.labels:
            return

        latest_end_time = -1
        latest_label_index = -1
        for i, lab in enumerate(self.labels):
            if lab["end"] > latest_end_time:
                latest_end_time = lab["end"]
                latest_label_index = i

        if latest_label_index != -1:
            last = self.labels.pop(latest_label_index)
            self._redraw_all_labels()
            self._update_status(
                f"Deleted label: {last['label']} [{last['start']:.3f}, {last['end']:.3f}]"
            )

    def _zoom_active_plot_y(self, factor):
        """Zooms the Y-axis of the currently hovered plot."""
        if self.hovered_plot is None:
            return

        plot = self.hovered_plot
        plot.enableAutoRange("y", False)
        vb = plot.getViewBox()
        y_range = vb.viewRange()[1]
        center = (y_range[0] + y_range[1]) / 2.0
        height = (y_range[1] - y_range[0]) * factor
        vb.setYRange(center - height / 2.0, center + height / 2.0, padding=0)

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        ktxt = ev.text().lower()
        key = ev.key()

        if ev.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            if key == QtCore.Qt.Key.Key_1:
                self._zoom_active_plot_y(0.9)
                return
            if key == QtCore.Qt.Key.Key_2:
                self._zoom_active_plot_y(1.1)
                return

        if key == QtCore.Qt.Key.Key_Space:
            self._toggle_playback()
            return

        if key in (QtCore.Qt.Key.Key_BracketRight, QtCore.Qt.Key.Key_PageDown):
            self._page(+1)
            return
        if key in (QtCore.Qt.Key.Key_BracketLeft, QtCore.Qt.Key.Key_PageUp):
            self._page(-1)
            return

        if (
            ktxt in self.keymap
            and self._select_start is not None
            and self._select_end is not None
        ):
            self._stop_playback_if_playing()
            label = self.keymap[ktxt]
            a = float(min(self._select_start, self._select_end))
            b = float(max(self._select_start, self._select_end))
            if b > a:
                self._add_new_label(a, b, label)
                self._update_status(f"Labeled {label}: [{a:.3f}, {b:.3f}]")
                self._clear_selection()
                return

        # Toggle hypnogram zoom
        if ktxt == "z":
            self._toggle_hypnogram_zoom()
            return

        super().keyPressEvent(ev)

    def _toggle_hypnogram_zoom(self):
        self.hypnogram_zoomed = not self.hypnogram_zoomed
        self._update_hypnogram_xrange()

    def _update_hypnogram_xrange(self):
        if self.hypnogram_plot is None:
            return
        if not self.hypnogram_zoomed:
            # Show full extent
            self.hypnogram_plot.enableAutoRange("x", False)
            self.hypnogram_plot.setXRange(
                self.t_global_min, self.t_global_max, padding=0
            )
        else:
            # Zoom around current window with +/- padding
            pad = float(self.hypnogram_zoom_padding)
            a = max(self.t_global_min, self.window_start - pad)
            b = min(self.t_global_max, self.window_start + self.window_len + pad)
            if b <= a:
                b = min(self.t_global_max, a + 1.0)
            self.hypnogram_plot.enableAutoRange("x", False)
            self.hypnogram_plot.setXRange(a, b, padding=0)

    # ---------- Export / Import ----------

    def _on_load_labels(self):
        self._stop_playback_if_playing()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load labels from CSV", filter="CSV (*.csv)"
        )
        if not path:
            return

        loaded_labels = []
        try:
            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                if header != ["start_s", "end_s", "label"]:
                    raise ValueError("CSV header does not match expected format.")

                for row in reader:
                    if not row:
                        continue
                    loaded_labels.append(
                        {
                            "start": float(row[0]),
                            "end": float(row[1]),
                            "label": str(row[2]),
                        }
                    )

            self.labels = sorted(loaded_labels, key=lambda x: x["start"])
            self._merge_adjacent_same_labels()
            self._redraw_all_labels()
            self._update_status(
                f"Loaded {len(self.labels)} labels from {os.path.basename(path)}"
            )

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Load error", f"Failed to load or parse labels file:\n\n{e}"
            )

    def _on_export_labels(self):
        self._stop_playback_if_playing()
        if not self.labels:
            QtWidgets.QMessageBox.information(self, "Export", "No labels to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export labels to CSV", filter="CSV (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["start_s", "end_s", "label"])
                for lab in self.labels:
                    writer.writerow(
                        [f"{lab['start']:.6f}", f"{lab['end']:.6f}", lab["label"]]
                    )
            self._update_status(f"Exported labels to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export error", str(e))

    # ---------- Navigation / rendering ----------

    def _stop_playback_if_playing(self):
        """Stops playback if it is currently active."""
        if self.is_playing:
            self.is_playing = False
            self.playback_timer.stop()
            self._update_status("Playback stopped.")

    def _toggle_playback(self):
        """Toggles video playback on or off."""
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
        """Called by the QTimer to advance the cursor time."""
        if not self.is_playing:
            return

        dt_ms = self.playback_elapsed_timer.restart()
        dt_sec = dt_ms / 1000.0

        t_start = self.window_start
        t_end = self.window_start + self.window_len
        if t_end <= t_start:
            return

        new_cursor_time = self.cursor_time + dt_sec

        if new_cursor_time >= t_end:
            new_cursor_time = t_start + (new_cursor_time - t_end)
            if new_cursor_time >= t_end:
                new_cursor_time = t_start

        self._set_cursor_time(new_cursor_time, update_slider=True)

    def _page(self, direction: int):
        self._stop_playback_if_playing()
        direction = 1 if direction >= 1 else -1
        total = self.t_global_max - self.t_global_min
        if total <= 0:
            return
        new_start = self.window_start + direction * self.window_len
        new_start = clamp(
            new_start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        rel = (
            0.0
            if self.window_len <= 0
            else (self.cursor_time - self.window_start) / self.window_len
        )
        self.window_start = new_start
        self.cursor_time = self.window_start + rel * self.window_len

        self._apply_x_range()
        self._update_nav_slider_from_window()

    def _on_smooth_scroll(self, direction: int):
        self._stop_playback_if_playing()
        direction = 1 if direction >= 1 else -1
        total = self.t_global_max - self.t_global_min
        if total <= 0:
            return
        delta = direction * float(self.smooth_scroll_fraction) * float(self.window_len)
        new_start = self.window_start + delta
        new_start = clamp(
            new_start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        rel = (
            0.0
            if self.window_len <= 0
            else (self.cursor_time - self.window_start) / self.window_len
        )
        self.window_start = new_start
        self.cursor_time = self.window_start + rel * self.window_len
        self._apply_x_range()
        self._update_nav_slider_from_window()

    def _on_window_len_changed(self, v):
        self._stop_playback_if_playing()
        self.window_len = float(v)
        self.window_start = clamp(
            self.window_start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        self._apply_x_range()
        self._update_nav_slider_from_window()

    def _on_nav_slider_changed(self, value):
        self._stop_playback_if_playing()
        if self.t_global_max <= self.t_global_min:
            return
        total = max(1e-9, self.t_global_max - self.t_global_min)
        span = max(1e-9, total - self.window_len)
        start = self.t_global_min + (value / 10000.0) * span
        self.window_start = clamp(
            start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        self._apply_x_range()

    def _apply_x_range(self):
        xr = (self.window_start, self.window_start + self.window_len)
        for plt in self.plots:
            plt.enableAutoRange("x", False)
            plt.setXRange(*xr, padding=0.0)

        new_cursor_time = clamp(self.cursor_time, xr[0], xr[1])
        self._set_cursor_time(new_cursor_time, update_slider=True)

        self._refresh_curves()

        # Update hypnogram view region to show current window
        if self.hypnogram_view_region is not None:
            self.hypnogram_view_region.setRegion(xr)
        # If zoomed, keep hypnogram centered on the current window +/- padding
        if self.hypnogram_zoomed:
            self._update_hypnogram_xrange()

    def _update_nav_slider_from_window(self):
        if self.t_global_max <= self.t_global_min:
            self.nav_slider.setValue(0)
            return
        total = max(1e-9, self.t_global_max - self.t_global_min)
        span = max(1e-9, total - self.window_len)
        frac = (
            0.0
            if span <= 0
            else clamp((self.window_start - self.t_global_min) / span, 0.0, 1.0)
        )
        self.nav_slider.blockSignals(True)
        self.nav_slider.setValue(int(round(frac * 10000)))
        self.nav_slider.blockSignals(False)

    def _update_cursor_lines(self):
        for ln in self.plot_cur_lines:
            ln.setPos(self.cursor_time)

    def _set_cursor_time(self, t, update_slider=True):
        self.cursor_time = t

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
        if self.video2_frame_times is not None and len(self.video2_frame_times):
            idx2 = find_nearest_frame(self.video2_frame_times, self.cursor_time)
            QtCore.QMetaObject.invokeMethod(
                self._video2_worker,
                "requestFrame",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(int, int(idx2)),
            )

        if not self.is_playing:
            self._update_status()

    def _on_window_cursor_changed(self, value):
        self._stop_playback_if_playing()
        frac = value / 10000.0
        t = self.window_start + frac * self.window_len
        self._set_cursor_time(t, update_slider=False)

    def _update_window_cursor_from_cursor_time(self):
        frac = (
            0.0
            if self.window_len <= 0
            else clamp(
                (self.cursor_time - self.window_start) / self.window_len, 0.0, 1.0
            )
        )
        self.window_cursor_slider.blockSignals(True)
        self.window_cursor_slider.setValue(int(round(frac * 10000)))
        self.window_cursor_slider.blockSignals(False)

    def _target_pts(self):
        if not self.plots:
            return self.max_pts_per_plot
        vb = self.plots[0].getViewBox()
        px = max(300, int(vb.width()))
        return int(min(2 * px, self.max_pts_per_plot))

    def _refresh_curves(self):
        t0, t1 = self.window_start, self.window_start + self.window_len
        max_pts = self._target_pts()
        for s, curve in zip(self.series, self.curves):
            tx, yx = segment_for_window(s.t, s.y, t0, t1, max_pts=max_pts)
            curve.setData(tx, yx, _callSync="off")

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rescale_video_frame()
        self._rescale_static_image()
        QtCore.QTimer.singleShot(50, self._refresh_curves)
        QtCore.QTimer.singleShot(60, self._align_left_axes)

    def _align_left_axes(self):
        try:
            if not self.plots:
                return
            widths = []
            for plt in self.plots:
                ax = plt.getAxis("left")
                widths.append(int(ax.width()))
            if not widths:
                return
            target = max(max(widths), 55)
            for plt in self.plots:
                ax = plt.getAxis("left")
                ax.setWidth(int(target))
        except Exception:
            pass

    # ---------- Help/Status & cleanup ----------

    def _show_help(self):
        self._stop_playback_if_playing()
        QtWidgets.QMessageBox.information(
            self,
            "Help",
            (
                "<b>Hotkeys</b><br>"
                "<b>Spacebar:</b> Toggle window playback<br>"
                "<b>Ctrl+D:</b> Show/hide Y-Axis Controls<br>"
                "<b>Ctrl+1 / Ctrl+2:</b> Zoom Y-Axis In / Out (on hovered plot)<br>"
                "<b>Labels:</b> w=Wake, n=NREM, r=REM, a=Artifact, Backspace=delete last<br>"
                "<b>Paging:</b> [ ] or Scroll Wheel = previous/next page<br><br>"
                "Click-drag in any plot to create selection. Selection stays active across pages; "
                "drag its handles to extend, then press a label hotkey.<br>"
            ),
        )

    def _update_status(self, msg=None):
        info = []
        if self.series:
            info += [
                f"{len(self.series)} traces",
                f"t=[{self.t_global_min:.2f},{self.t_global_max:.2f}]s",
            ]
        info += [
            f"win={self.window_len:.2f}s @ {self.window_start:.2f}s",
            self._format_cursor_with_state(),
        ]
        if self.is_playing and not msg:
            msg = "Playing..."
        if msg:
            info.append("| " + msg)
        self.status.showMessage("  ".join(info))

    def _format_cursor_with_state(self):
        label = self._get_state_at_time(self.cursor_time)
        state_txt = label if label is not None else "Unlabeled"
        return f"cursor={self.cursor_time:.3f}s, state='{state_txt}'"

    def _get_state_at_time(self, t):
        if not self.labels:
            return None
        # labels are kept sorted by start
        for lab in self.labels:
            if lab["start"] <= t < lab["end"]:
                return lab["label"]
            if lab["start"] > t:
                break
        return None

    def closeEvent(self, ev):
        try:
            self._stop_playback_if_playing()
            QtCore.QMetaObject.invokeMethod(
                self._video_worker, "stop", QtCore.Qt.QueuedConnection
            )
            QtCore.QMetaObject.invokeMethod(
                self._video2_worker, "stop", QtCore.Qt.QueuedConnection
            )
            self._video_thread.quit()
            self._video2_thread.quit()
            if not self._video_thread.wait(1000):
                self._video_thread.terminate()
            if not self._video2_thread.wait(1000):
                self._video2_thread.terminate()
        except Exception as e:
            print(f"ERROR: Exception during closeEvent: {e}")
        super().closeEvent(ev)


# ---------------- Main ----------------


def main():
    parser = argparse.ArgumentParser(description="Sleep Scorer GUI")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to directory with time series files (*_t.npy, *_y.npy)",
    )
    parser.add_argument(
        "--data_files",
        nargs="+",
        type=str,
        help=(
            "Ordered list of .npy files (mix of *_t.npy and *_y.npy). "
            "Pairs are matched by basename; display order follows first appearance."
        ),
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        type=str,
        help=(
            "Optional colors matching series order. Accepts hex (#RRGGBB[AA]), 0xRRGGBB, or R,G,B[,A]."
        ),
    )
    parser.add_argument("--video", type=str, help="Path to video file (.mp4)")
    parser.add_argument(
        "--frame_times", type=str, help="Path to video frame times file (.npy)"
    )
    parser.add_argument("--video2", type=str, help="Optional second video file (.mp4)")
    parser.add_argument(
        "--frame_times2", type=str, help="Path to second video frame times (.npy)"
    )
    parser.add_argument(
        "--image", type=str, help="Path to static image file (.png, .jpg, etc.)"
    )
    parser.add_argument(
        "--fixed_scale",
        action="store_true",
        help=(
            "Disable Y auto-scaling and apply fixed initial Y ranges from robust percentiles."
        ),
    )
    parser.add_argument(
        "--low_profile_x",
        action="store_true",
        help=("Hide X-axis labels and ticks for all but the bottom plot."),
    )
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    w = SleepScorerApp(
        data_dir=args.data_dir,
        data_files=args.data_files,
        colors=args.colors,
        video_path=args.video,
        frame_times_path=args.frame_times,
        video2_path=args.video2,
        frame_times2_path=args.frame_times2,
        image_path=args.image,
        fixed_scale=args.fixed_scale,
        low_profile_x=args.low_profile_x,
    )
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
