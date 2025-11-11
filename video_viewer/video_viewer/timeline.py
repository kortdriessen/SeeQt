from __future__ import annotations

import time
from typing import Tuple

from PySide6.QtCore import QObject, QTimer, Signal


class TimelineController(QObject):
    """Global timeline that keeps all videos in sync."""

    time_changed = Signal(float)
    playing_changed = Signal(bool)
    range_changed = Signal(float, float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._start = 0.0
        self._end = 10.0
        self._current = 0.0
        self._playing = False
        self._last_tick = time.perf_counter()
        self._timer = QTimer(self)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._on_timeout)

    @property
    def current_time(self) -> float:
        return self._current

    @property
    def range(self) -> Tuple[float, float]:
        return self._start, self._end

    def set_range(self, start: float, end: float) -> None:
        end = max(end, start + 1e-6)
        self._start, self._end = start, end
        if self._current < start or self._current > end:
            self.set_time(start, emit=False)
        self.range_changed.emit(start, end)

    def set_time(self, value: float, emit: bool = True) -> None:
        value = min(max(value, self._start), self._end)
        self._current = value
        if emit:
            self.time_changed.emit(value)

    def play(self) -> None:
        if self._playing:
            return
        self._playing = True
        self._last_tick = time.perf_counter()
        self._timer.start()
        self.playing_changed.emit(True)

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._timer.stop()
        self.playing_changed.emit(False)

    def toggle(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def stop(self) -> None:
        self.pause()
        self.set_time(self._start)

    def _on_timeout(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_tick
        self._last_tick = now
        next_time = self._current + elapsed
        if next_time >= self._end:
            self.set_time(self._end)
            self.pause()
        else:
            self.set_time(next_time)
