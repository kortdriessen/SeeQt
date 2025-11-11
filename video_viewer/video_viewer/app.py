from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Optional

from PySide6.QtCore import Qt, QRect
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .configuration import (
    ContainerSpec,
    VideoSpec,
    ViewerConfig,
    write_template,
)
from .media import VideoClip
from .timeline import TimelineController
from .widgets import VideoContainerWidget, styled_subwindow_title


class VideoContainerWindow(QMdiSubWindow):
    """Wrapper around a QMdiSubWindow that hosts a VideoContainerWidget."""

    def __init__(self, container_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.container_id = container_id
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self._widget = VideoContainerWidget(container_id)
        self.setWidget(self._widget)
        self.setWindowTitle(styled_subwindow_title(container_id))
        self.clip: Optional[VideoClip] = None
        self.video_id: Optional[str] = None

    @property
    def selector(self) -> VideoContainerWidget:
        return self._widget

    def set_available_videos(self, videos: Iterable[VideoSpec]) -> None:
        self._widget.populate_videos(videos)
        if self.video_id:
            self._widget.set_selected_video(self.video_id, emit=False)

    def set_clip(self, clip: Optional[VideoClip]) -> None:
        self.clip = clip
        self.video_id = clip.id if clip else None
        self._widget.set_selected_video(self.video_id, emit=False)
        if clip:
            self._widget.set_status(
                f"{clip.name} [{clip.start_time:.3f}s - {clip.end_time:.3f}s]"
            )
        else:
            self._widget.set_status("No video assigned")
        if clip is None:
            self._widget.show_pixmap(None, message="No video")

    def render_time(self, timepoint: float) -> None:
        if not self.clip:
            self._widget.show_pixmap(None, message="No video")
            return
        pixmap = self.clip.pixmap_at(timepoint)
        if pixmap is None:
            if timepoint < self.clip.start_time:
                msg = f"Starts in {self.clip.start_time - timepoint:.2f}s"
            elif timepoint > self.clip.end_time:
                msg = "Finished"
            else:
                msg = "Frame unavailable"
            self._widget.show_pixmap(None, message=msg)
        else:
            self._widget.show_pixmap(pixmap)


class MainWindow(QMainWindow):
    """Primary window for the synchronized video viewer."""

    SLIDER_STEPS = 10_000

    def __init__(self, config_path: Optional[Path] = None) -> None:
        super().__init__()
        self.setWindowTitle("SeeQt video viewer")
        self.resize(1400, 900)

        self.mdi = QMdiArea()
        self.timeline = TimelineController()
        self.timeline.time_changed.connect(self._on_time_changed)
        self.timeline.playing_changed.connect(self._on_play_state_changed)
        self.timeline.range_changed.connect(self._on_range_changed)
        self._slider_active = False

        self.video_specs: Dict[str, VideoSpec] = {}
        self.video_clips: Dict[str, VideoClip] = {}
        self.containers: Dict[str, VideoContainerWindow] = {}
        self._container_counter = 1
        self._config_path: Optional[Path] = None

        self._build_ui()

        if config_path:
            self.load_config(config_path)

    # ---------- UI construction -------------------------------------------------
    def _build_ui(self) -> None:
        self._build_menus()
        self._build_toolbar()
        self.setStatusBar(QStatusBar())

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.mdi, 1)
        layout.addWidget(self._build_control_panel())
        self.setCentralWidget(central)

    def _build_menus(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        open_action = file_menu.addAction("Load configuration...")
        open_action.triggered.connect(self.load_config_dialog)

        save_action = file_menu.addAction("Save configuration...")
        save_action.triggered.connect(self.save_config_dialog)

        file_menu.addSeparator()

        template_action = file_menu.addAction("Write config template...")
        template_action.triggered.connect(self.write_template_dialog)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Tools", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        add_video_btn = QPushButton("Add Video")
        add_video_btn.clicked.connect(self.add_video_dialog)
        toolbar.addWidget(add_video_btn)

        add_container_btn = QPushButton("Add Container")
        add_container_btn.clicked.connect(self.add_container_dialog)
        toolbar.addWidget(add_container_btn)

        toolbar.addSeparator()

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.reset_viewer)
        toolbar.addWidget(clear_btn)

    def _build_control_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("controlPanel")
        panel.setStyleSheet(
            "#controlPanel {border-top: 1px solid #333; padding: 6px; background: #1f1f1f;}"
        )

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.timeline.toggle)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.timeline.stop)

        self.time_label = QLabel("0.000 s")
        self.time_label.setMinimumWidth(120)
        self.time_label.setAlignment(Qt.AlignCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.SLIDER_STEPS)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_value_changed)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(6)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.time_label)

        layout.addLayout(button_layout)
        layout.addWidget(self.slider)
        return panel

    # ---------- Menu/toolbar callbacks -----------------------------------------
    def load_config_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select configuration", "", "Config Files (*.json *.yaml *.yml);;All Files (*)"
        )
        if path:
            self.load_config(Path(path))

    def save_config_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save configuration", "", "Config Files (*.json *.yaml *.yml);;All Files (*)"
        )
        if path:
            self.save_config(Path(path))

    def write_template_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Write template", "", "YAML (*.yaml);;All Files (*)"
        )
        if path:
            write_template(Path(path))

    def add_video_dialog(self) -> None:
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Select video file", "", "Video Files (*.mp4 *.mov *.avi);;All Files (*)"
        )
        if not video_path:
            return
        frame_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select frame times npy",
            str(Path(video_path).parent),
            "NumPy arrays (*.npy);;All Files (*)",
        )
        if not frame_path:
            return
        base_id = Path(video_path).stem
        video_id = self._next_unique_video_id(base_id)
        spec = VideoSpec(
            id=video_id,
            name=base_id,
            video_path=Path(video_path),
            frame_times_path=Path(frame_path),
        )
        self._register_video(spec)

    def add_container_dialog(self) -> None:
        suggested = f"container-{self._container_counter}"
        text, ok = QInputDialog.getText(self, "Container ID", "Enter container id:", text=suggested)
        if not ok or not text:
            return
        container_id = text.strip()
        if container_id in self.containers:
            QMessageBox.warning(self, "Duplicate", f"Container '{container_id}' already exists.")
            return
        self._container_counter += 1
        self._create_container(container_id, QRect(0, 0, 320, 240))

    # ---------- Core logic ------------------------------------------------------
    def load_config(self, path: Path) -> None:
        try:
            config = ViewerConfig.load(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"Failed to load config:\n{exc}")
            return
        self._config_path = path
        self.reset_viewer()
        for spec in config.videos:
            try:
                self._register_video(spec, allow_replace=True)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Video error",
                    f"Failed to load video '{spec.id}':\n{exc}",
                )
        for container_spec in config.containers:
            rect = QRect(*container_spec.geometry)
            window = self._create_container(container_spec.id, rect)
            if container_spec.video_id:
                self._assign_video_to_container(container_spec.id, container_spec.video_id)
            window.show()
        if config.canvas_size:
            self.resize(*config.canvas_size)
        self._refresh_container_video_lists()
        self._update_timeline_range(config.timeline_start, config.timeline_end)

    def save_config(self, path: Path) -> None:
        configs = ViewerConfig(
            videos=list(self.video_specs.values()),
            containers=[
                ContainerSpec(
                    id=cid,
                    geometry=self._container_geometry(window),
                    video_id=window.video_id,
                )
                for cid, window in self.containers.items()
            ],
        )
        configs.timeline_start, configs.timeline_end = self.timeline.range
        configs.save(path)
        self._config_path = path

    def reset_viewer(self) -> None:
        for window in list(self.containers.values()):
            window.close()
        self.containers.clear()
        for clip in self.video_clips.values():
            clip.release()
        self.video_clips.clear()
        self.video_specs.clear()
        self._container_counter = 1
        self.mdi.closeAllSubWindows()
        self.timeline.set_range(0.0, 10.0)
        self.timeline.stop()

    def _register_video(self, spec: VideoSpec, allow_replace: bool = False) -> None:
        if spec.id in self.video_specs and not allow_replace:
            QMessageBox.warning(self, "Duplicate id", f"Video id '{spec.id}' already exists.")
            return
        if spec.id in self.video_specs:
            old = self.video_clips.pop(spec.id, None)
            if old:
                old.release()
        clip = VideoClip(spec)
        self.video_specs[spec.id] = spec
        self.video_clips[spec.id] = clip
        self._refresh_container_video_lists()
        self._update_timeline_range()

    def _refresh_container_video_lists(self) -> None:
        videos = sorted(self.video_specs.values(), key=lambda spec: spec.name or spec.id)
        for container in self.containers.values():
            container.set_available_videos(videos)

    def _update_timeline_range(
        self, forced_start: Optional[float] = None, forced_end: Optional[float] = None
    ) -> None:
        if self.video_clips:
            start = forced_start if forced_start is not None else min(
                clip.start_time for clip in self.video_clips.values()
            )
            end = forced_end if forced_end is not None else max(
                clip.end_time for clip in self.video_clips.values()
            )
        else:
            start, end = 0.0, 10.0
        self.timeline.set_range(start, end)
        self.time_label.setText(f"{self.timeline.current_time:.3f} s")

    def _create_container(self, container_id: str, geometry: QRect) -> VideoContainerWindow:
        window = VideoContainerWindow(container_id)
        window.selector.video_changed.connect(
            partial(self._assign_video_to_container, container_id)
        )
        self.mdi.addSubWindow(window)
        window.setGeometry(geometry)
        window.show()
        self.containers[container_id] = window
        self._refresh_container_video_lists()
        return window

    def _assign_video_to_container(self, container_id: str, video_id: Optional[str]) -> None:
        window = self.containers.get(container_id)
        if not window:
            return
        if not video_id:
            window.set_clip(None)
            return
        clip = self.video_clips.get(video_id)
        if clip is None:
            QMessageBox.warning(
                self,
                "Missing video",
                f"Video '{video_id}' is not loaded.",
            )
            window.set_clip(None)
            return
        window.set_clip(clip)
        window.render_time(self.timeline.current_time)

    def _on_time_changed(self, value: float) -> None:
        for window in self.containers.values():
            window.render_time(value)
        self.time_label.setText(f"{value:.3f} s")
        if not self._slider_active:
            slider_value = self._time_to_slider(value)
            self.slider.blockSignals(True)
            self.slider.setValue(slider_value)
            self.slider.blockSignals(False)

    def _on_play_state_changed(self, playing: bool) -> None:
        self.play_button.setText("Pause" if playing else "Play")

    def _on_range_changed(self, start: float, end: float) -> None:
        self.statusBar().showMessage(f"Timeline: {start:.3f}s -> {end:.3f}s")
        if not self._slider_active:
            self.slider.setValue(self._time_to_slider(self.timeline.current_time))

    def _on_slider_pressed(self) -> None:
        self._slider_active = True

    def _on_slider_released(self) -> None:
        self._slider_active = False
        self.timeline.set_time(self._slider_to_time(self.slider.value()))

    def _on_slider_value_changed(self, value: int) -> None:
        if self._slider_active:
            self.timeline.set_time(self._slider_to_time(value))

    def _time_to_slider(self, timepoint: float) -> int:
        start, end = self.timeline.range
        if end <= start:
            return 0
        fraction = (timepoint - start) / (end - start)
        return int(fraction * self.SLIDER_STEPS)

    def _slider_to_time(self, slider_value: int) -> float:
        start, end = self.timeline.range
        return start + (end - start) * (slider_value / self.SLIDER_STEPS)

    def _container_geometry(self, window: VideoContainerWindow) -> tuple[int, int, int, int]:
        rect = window.geometry()
        return rect.x(), rect.y(), rect.width(), rect.height()

    def _next_unique_video_id(self, base: str) -> str:
        candidate = base
        suffix = 1
        while candidate in self.video_specs:
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.timeline.stop()
        for clip in self.video_clips.values():
            clip.release()
        super().closeEvent(event)


def run_viewer(config_path: Optional[Path] = None) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(config_path=config_path)
    window.show()
    app.exec()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronized video viewer")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a JSON/YAML configuration file",
    )
    parser.add_argument(
        "--write-template",
        type=Path,
        help="Write an example configuration to the given path and exit",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    if args.write_template:
        write_template(args.write_template)
        return
    run_viewer(config_path=args.config)


