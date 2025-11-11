## SeeQt Video Viewer

The `video_viewer/` package now ships a PySide6 application that keeps any number of MP4 clips in sync by following per-frame timestamps stored in companion `video_frame_times.npy` files. Each clip lives inside a movable, resizable container on a shared canvas so you can create arbitrary layouts and scrub, play, or pause all feeds together.

### Launching

```bash
cd /path/to/SeeQt
python -m video_viewer                    # start empty workspace
python -m video_viewer --config layout.yaml
seeqt-video-viewer --config layout.json   # via pyproject script
```

The dependencies already appear in `pyproject.toml` (`PySide6`, `numpy`, `opencv-python`, `PyYAML`).

### UI Basics

- **Canvas** - each video sits inside a QMdi subwindow. Drag the title bar to reposition and drag the edges to resize.
- **Container widget** - every subwindow exposes a dropdown listing all loaded clips; pick a clip to feed that container. The status row shows the clip name and temporal coverage.
- **Transport controls** - the bottom panel provides Play or Pause, Stop, and a high-resolution scrub slider that moves the shared timeline. All containers refresh from this global clock.
- **Toolbar** - quick buttons let you add new videos (choose the MP4 and its `.npy` frame-times), spawn fresh containers, or clear the workspace.
- **Menus** - File -> Load/Save configuration to persist the current layout, timeline range, and assignments, plus an option to write an example template.

### Configuration Format

Configs can be JSON or YAML. Each entry lists the clips, the containers, and optional canvas or timeline defaults. Paths are resolved relative to the config file.

```yaml
videos:
  - id: camera-1
    name: Top
    video_path: ./cam1.mp4
    frame_times_path: ./cam1_frame_times.npy
  - id: camera-2
    video_path: ./cam2.mp4
    frame_times_path: ./cam2_frame_times.npy

containers:
  - id: large
    geometry: [0, 0, 900, 600]      # x, y, width, height (canvas coordinates)
    video_id: camera-1
  - id: inset
    geometry: [950, 50, 400, 300]
    video_id: camera-2

timeline:
  start: 0.0                        # optional - defaults to min clip start
  end: null                         # optional - defaults to max clip end
canvas_size: [1600, 900]            # optional - initial main-window size
```

Load the file with `--config` or via File -> Load configuration. After arranging your workspace, choose File -> Save configuration to emit a new JSON or YAML definition that includes all loaded videos and container geometries.

### Manual Workflow

1. Launch the app with no config.
2. Click **Add Video** for each MP4/`video_frame_times.npy` pair you want to view.
3. Click **Add Container** as many times as needed, arranging the windows by dragging/resizing them on the canvas.
4. Use the dropdown inside each container to assign the desired clip.
5. Hit **Play** to watch everything advance in lockstep or grab the scrubber to inspect specific timestamps.

This setup gives you a flexible, scriptable way to inspect synchronized video feeds while keeping the layout reproducible through config files.
