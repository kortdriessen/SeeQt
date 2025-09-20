#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sleep Scorer (fast): multi-trace viewer with windowed rendering, video scrubbing,
page hotkeys, and cross-page draggable labeling.

pip install PySide6 pyqtgraph opencv-python numpy
"""

import os, sys, glob, csv, math
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


# ---------------- Selectable ViewBox ----------------


class SelectableViewBox(pg.ViewBox):
    sigDragStart = QtCore.Signal(float)
    sigDragUpdate = QtCore.Signal(float)
    sigDragFinish = QtCore.Signal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(x=False, y=True)
        self._drag = False

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.LeftButton:
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
        super().mouseDragEvent(ev, axis=axis)


# ---------------- Video worker with tiny cache ----------------


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


# ---------------- Main window ----------------


class SleepScorerApp(QtWidgets.QMainWindow):
    def __init__(self):
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

        # Rendering budget (per plot)
        self.max_pts_per_plot = 4000

        # Window/cursor & labels
        self.window_len = 10.0
        self.window_start = 0.0
        self.cursor_time = 0.0

        self.keymap = {"w": "Wake", "n": "NREM", "r": "REM", "a": "Artifact"}
        self.label_colors = {
            "Wake": (200, 200, 0, 80),
            "NREM": (0, 200, 255, 80),
            "REM": (255, 0, 200, 80),
            "Artifact": (255, 120, 0, 80),
        }
        self.labels = []
        self._select_start = None
        self._select_end = None

        # Video
        self.video_frame_times = None
        self._video_is_open = False
        self._video_thread = QtCore.QThread(self)
        self._video_worker = VideoWorker(cache_frames=120)
        self._video_worker.moveToThread(self._video_thread)

        # Playback
        self.is_playing = False
        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.timeout.connect(self._advance_playback_frame)
        self.playback_elapsed_timer = QtCore.QElapsedTimer()

        self._build_ui()

        self._video_worker.frameReady.connect(self._on_frame_ready)
        self._video_worker.opened.connect(self._on_video_opened)

        self._video_thread.start()

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
        self.nav_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.nav_slider.setRange(0, 10000)
        self.nav_slider.valueChanged.connect(self._on_nav_slider_changed)
        top.addWidget(self.nav_slider, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        v.addWidget(splitter, 1)

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
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumHeight(240)
        self.video_label.setStyleSheet("background-color:#222;border:1px solid #444;")
        rl.addWidget(self.video_label, 3)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Cursor:"))
        self.window_cursor_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.window_cursor_slider.setRange(0, 10000)
        self.window_cursor_slider.valueChanged.connect(self._on_window_cursor_changed)
        row.addWidget(self.window_cursor_slider)

        roww = QtWidgets.QWidget()
        roww.setLayout(row)
        rl.addWidget(roww)

        ygroup = QtWidgets.QGroupBox("Per-trace Y-axis")
        ylay = QtWidgets.QVBoxLayout(ygroup)
        self.y_controls_container = QtWidgets.QWidget()
        self.y_controls_layout = QtWidgets.QFormLayout(self.y_controls_container)
        ylay.addWidget(self.y_controls_container)
        ylay.addStretch(1)
        rl.addWidget(ygroup, 2)

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

        # *** FIX: Add Load Labels action to menu ***
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

        mhelp = self.menuBar().addMenu("&Help")
        hh = QtGui.QAction("Shortcuts / Help", self)
        hh.triggered.connect(self._show_help)
        mhelp.addAction(hh)

    # ---------- Data ----------

    def _on_load_time_series(self):
        self._stop_playback_if_playing()
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder with *_t.npy and *_y.npy"
        )
        if not folder:
            return
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
            plt = self.plot_area.addPlot(row=idx, col=0, viewBox=vb)
            plt.setLabel("left", s.name)
            plt.setLabel(
                "bottom", "Time", units="s" if idx == len(self.series) - 1 else None
            )
            plt.showGrid(x=True, y=True, alpha=0.15)
            plt.addLegend(offset=(10, 10))
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

        self._rebuild_y_controls()
        self._apply_x_range()
        self._update_nav_slider_from_window()
        self._update_status(f"Loaded {len(self.series)} series.")

    def _rebuild_y_controls(self):
        lay = self.y_controls_layout
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self._y_controls = []
        for idx, s in enumerate(self.series):
            roww = QtWidgets.QWidget()
            rowl = QtWidgets.QHBoxLayout(roww)
            rowl.setContentsMargins(0, 0, 0, 0)
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

            rowl.addWidget(QtWidgets.QLabel(s.name))
            rowl.addStretch(1)
            rowl.addWidget(auto)
            rowl.addWidget(QtWidgets.QLabel("Min"))
            rowl.addWidget(mn)
            rowl.addWidget(QtWidgets.QLabel("Max"))
            rowl.addWidget(mx)

            lay.addRow(roww)
            self._y_controls.append((auto, mn, mx))

    # ---------- Video ----------

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

    # ---------- Selection / labeling ----------

    def _on_drag_start(self, x):
        self._stop_playback_if_playing()
        self._select_start = x
        self._select_end = x
        self._show_active_selection()

    def _on_drag_update(self, x):
        self._select_end = x
        self._show_active_selection()

    def _on_drag_finish(self, x):
        self._select_end = x
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

        # Process every existing label against the new one
        for existing in self.labels:
            ex_start, ex_end = existing["start"], existing["end"]

            # Check for overlap: max of starts must be less than min of ends
            overlap_start = max(ex_start, start)
            overlap_end = min(ex_end, end)

            if overlap_start >= overlap_end:  # No overlap
                updated_labels.append(existing)
                continue

            # Old label is split into two by the new one
            if ex_start < start and ex_end > end:
                updated_labels.append(
                    {"start": ex_start, "end": start, "label": existing["label"]}
                )
                updated_labels.append(
                    {"start": end, "end": ex_end, "label": existing["label"]}
                )
            # Old label is truncated at its end
            elif ex_start < start:
                updated_labels.append(
                    {"start": ex_start, "end": start, "label": existing["label"]}
                )
            # Old label is truncated at its beginning
            elif ex_end > end:
                updated_labels.append(
                    {"start": end, "end": ex_end, "label": existing["label"]}
                )
            # Else, old label is completely covered and is not added

        # Add the new label itself
        updated_labels.append({"start": start, "end": end, "label": label})

        # Update the main label list and redraw everything
        self.labels = sorted(updated_labels, key=lambda x: x["start"])
        self._redraw_all_labels()

    def _redraw_all_labels(self):
        """Clears and redraws all visual label regions."""
        # Clear existing visual items
        for plot_regions in self.plot_label_regions:
            for item in plot_regions:
                if item.scene():
                    item.scene().removeItem(item)
            plot_regions.clear()

        # Redraw from the self.labels list
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

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        ktxt = ev.text().lower()
        key = ev.key()

        if key == QtCore.Qt.Key.Key_Space:
            self._toggle_playback()
            return

        # Page hotkeys
        if key in (QtCore.Qt.Key.Key_BracketRight, QtCore.Qt.Key.Key_PageDown):
            self._page(+1)
            return
        if key in (QtCore.Qt.Key.Key_BracketLeft, QtCore.Qt.Key.Key_PageUp):
            self._page(-1)
            return

        # Label hotkeys
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

        super().keyPressEvent(ev)

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
                header = next(reader)  # Skip header
                # Optional: check header for correctness
                if header != ["start_s", "end_s", "label"]:
                    raise ValueError("CSV header does not match expected format.")

                for row in reader:
                    if not row:
                        continue  # Skip empty rows
                    loaded_labels.append(
                        {
                            "start": float(row[0]),
                            "end": float(row[1]),
                            "label": str(row[2]),
                        }
                    )

            # Sort and assign the newly loaded labels
            self.labels = sorted(loaded_labels, key=lambda x: x["start"])
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
            self.playback_timer.start(16)  # ~60fps timer
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
        QtCore.QTimer.singleShot(0, self._refresh_curves)

    # ---------- Help/Status & cleanup ----------

    def _show_help(self):
        self._stop_playback_if_playing()
        QtWidgets.QMessageBox.information(
            self,
            "Help",
            (
                "<b>Hotkeys</b><br>"
                "<b>Spacebar:</b> Toggle window playback<br>"
                "<b>Labels:</b> w=Wake, n=NREM, r=REM, a=Artifact, Backspace=delete last<br>"
                "<b>Paging:</b> [ = previous page, ] = next page (also PageUp/PageDown)<br><br>"
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
            f"cursor={self.cursor_time:.3f}s",
        ]
        if self.is_playing and not msg:
            msg = "Playing..."
        if msg:
            info.append("| " + msg)
        self.status.showMessage("  ".join(info))

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


# ---------------- Main ----------------


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = SleepScorerApp()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
