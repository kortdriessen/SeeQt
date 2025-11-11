from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap

from .configuration import VideoSpec


def _as_pixmap(frame: np.ndarray) -> QPixmap:
    """Convert an OpenCV frame (BGR) to a QPixmap."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    bytes_per_line = 3 * w
    image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class VideoClip:
    """Small wrapper around OpenCV capture that knows about frame times."""

    def __init__(self, spec: VideoSpec):
        self.spec = spec
        self.id = spec.id
        self.name = spec.name or spec.id
        self.video_path = Path(spec.video_path)
        self.frame_times = np.array(
            np.load(spec.frame_times_path, mmap_mode="r"), dtype=np.float64
        )
        if self.frame_times.ndim != 1:
            raise ValueError(f"{spec.frame_times_path} must be 1-D array of frame times")
        self.start_time = float(self.frame_times[0])
        self.end_time = float(self.frame_times[-1])
        self.duration = self.end_time - self.start_time
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video {self.video_path}")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) or len(self.frame_times)

        self._cache_index: int = -1
        self._cache_pixmap: Optional[QPixmap] = None
        self._next_frame_index: int = 0
        self._lock = Lock()

    def index_for_time(self, timepoint: float) -> Optional[int]:
        if timepoint < self.start_time or timepoint > self.end_time:
            return None
        idx = int(np.searchsorted(self.frame_times, timepoint, side="right") - 1)
        if idx < 0 or idx >= len(self.frame_times):
            return None
        return idx

    def pixmap_at(self, timepoint: float) -> Optional[QPixmap]:
        with self._lock:
            idx = self.index_for_time(timepoint)
            if idx is None:
                return None
            if idx == self._cache_index and self._cache_pixmap is not None:
                return self._cache_pixmap
            frame = self._read_frame(idx)
            if frame is None:
                return None
            pixmap = _as_pixmap(frame)
            self._cache_index = idx
            self._cache_pixmap = pixmap
            return pixmap

    def _read_frame(self, index: int) -> Optional[np.ndarray]:
        if self.capture is None:
            return None
        if index != self._next_frame_index:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._next_frame_index = index
        ok, frame = self.capture.read()
        if not ok:
            return None
        self._next_frame_index = index + 1
        return frame

    def release(self) -> None:
        with self._lock:
            if self.capture:
                self.capture.release()
                self.capture = None
            self._cache_pixmap = None
