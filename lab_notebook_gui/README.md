# Lab Notebook GUI

This directory hosts a PySide6 application for browsing experiment materials directories.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

When the application starts, choose the root directory that contains your subject/experiment folders. Select a terminal materials directory to view its canvases, metadata, notes, and attachments.

## TIFF Viewer

You can quickly inspect interleaved TIFF stacks straight from the Lab Notebook via the “View TIFFs” button. To run the viewer standalone:

```powershell
python -m tiffviewer.viewer path\to\stack1.tif [stack2.tif ...]
```

The viewer assumes two interleaved channels (c1p1, c2p1, c1p2, ...), overlays magenta/cyan, and provides per-channel visibility, opacity, gamma, and contrast controls. Use the mouse wheel to move planes, `Ctrl+I` / `Ctrl+T` to zoom, and hold `Space` + drag to pan.


