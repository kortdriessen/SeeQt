# AGENTS.md

## Build & Run
- **Package manager**: `uv` (pyproject.toml, uv.lock)
- **Run video viewer**: `uv run seeqt-video-viewer` or `uv run python -m video_viewer`
- **Run sleep scorer**: `uv run python sleepscoring/sleepscore_main.py`
- **No test framework detected** — tests would need to be added

## Architecture
- **PySide6/Qt6** desktop GUI apps for viewing multimodal neuroscience data
- Key subprojects:
  - `video_viewer/` — synchronized multi-video viewer with timeline control
  - `sleepscoring/` — multi-trace EEG/LFP viewer with video + state labeling
  - `synapse_roladex/` — synapse annotation tool (napari-based)
  - `lab_notebook_gui/` — lab notebook interface
- **pyqtgraph** for high-performance plotting; **OpenCV** for video; **numpy/polars** for data

## Code Style
- Use `from __future__ import annotations` for type hints
- PySide6 imports: `from PySide6.QtCore import ...`, `from PySide6.QtWidgets import ...`
- Type hints on function signatures (`def foo(x: int) -> None:`)
- Classes use PascalCase; functions/variables use snake_case
- Config files in YAML/JSON; video frame times in `.npy` files
