#!/usr/bin/env python3
"""
two_image_viewer.py

A tiny GUI to view two images either side-by-side or stacked, with automatic
fit-to-window so each image is as large as possible while preserving aspect ratio.

Usage:
  python two_image_viewer.py /path/to/img1.png /path/to/img2.png [-o horizontal|vertical]

Shortcuts:
  Space / O     Toggle orientation (horizontal <-> vertical)
  F / F11       Toggle fullscreen
  Esc           Exit fullscreen (or quit if not fullscreen)
  Cmd/Ctrl+Q    Quit
"""
import sys
import argparse
from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QScrollArea,
    QLabel,
    QSplitter,
    QWidget,
    QToolBar,
    QMessageBox,
)


# ------------------------- Utility widgets ------------------------- #
class ImagePane(QScrollArea):
    """ScrollArea that auto-scales an image to fill the available viewport."""

    def __init__(self, path: Path):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QScrollArea.NoFrame)
        self.viewport().setStyleSheet("background-color: black;")

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("background: transparent;")
        self.setWidget(self._label)

        self._orig_pix: QPixmap | None = None
        self.load_image(path)

    def load_image(self, path: Path):
        pix = QPixmap(str(path))
        if pix.isNull():
            raise ValueError(f"Failed to load image: {path}")
        self._orig_pix = pix
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if not self._orig_pix:
            return
        # Fit the pixmap to the *viewport* while keeping aspect ratio
        max_size: QSize = self.viewport().size()
        if max_size.width() <= 0 or max_size.height() <= 0:
            return
        scaled = self._orig_pix.scaled(
            max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._label.setPixmap(scaled)

    # Keep image fitted while the pane resizes
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_scaled_pixmap()

    def showEvent(self, e):
        super().showEvent(e)
        self._update_scaled_pixmap()


# --------------------------- Main Window --------------------------- #
class TwoImageViewer(QMainWindow):
    def __init__(self, path1: Path, path2: Path, orientation: str):
        super().__init__()
        self.setWindowTitle("Two Image Viewer")
        self.setMinimumSize(640, 400)

        # Splitter lets the user drag the divider if they want unequal sizes
        self.splitter = QSplitter(
            Qt.Horizontal if orientation.startswith("h") else Qt.Vertical
        )
        self.splitter.setChildrenCollapsible(False)

        # Remove margins for true edge-to-edge fitting
        cw = QWidget()
        cw.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(cw)
        layout = self.splitter
        cw.setLayout(
            None
        )  # QMainWindow needs a central widget; we directly set splitter as central via setCentralWidget is not allowed, so we embed using setCentralWidget + set the splitter as the only child
        # Workaround: put splitter directly as central widget by reparenting
        self.setCentralWidget(self.splitter)

        # Panes
        try:
            self.pane1 = ImagePane(path1)
            self.pane2 = ImagePane(path2)
        except ValueError as e:
            QMessageBox.critical(self, "Load Error", str(e))
            sys.exit(1)

        self.splitter.addWidget(self.pane1)
        self.splitter.addWidget(self.pane2)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        self._build_toolbar()
        self._apply_dark_chrome()

    # Simple toolbar with orientation + fullscreen
    def _build_toolbar(self):
        tb = QToolBar("Controls")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        act_toggle = QAction("Toggle Orientation (O)", self)
        act_toggle.setShortcut("O")
        act_toggle.triggered.connect(self.toggle_orientation)
        tb.addAction(act_toggle)

        act_full = QAction("Fullscreen (F)", self)
        act_full.setShortcut("F")
        act_full.triggered.connect(self.toggle_fullscreen)
        tb.addAction(act_full)

        # Quit
        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q" if sys.platform != "darwin" else "Meta+Q")
        act_quit.triggered.connect(self.close)
        tb.addAction(act_quit)

    def _apply_dark_chrome(self):
        # Dark window chrome to emphasize images
        self.setStyleSheet(
            """
            QMainWindow { background: #111; }
            QToolBar { background: #1a1a1a; border: none; padding: 4px; }
            QToolBar QToolButton { color: #ddd; }
            QToolBar QToolButton:hover { color: white; }
            """
        )

    def toggle_orientation(self):
        orient = self.splitter.orientation()
        self.splitter.setOrientation(
            Qt.Vertical if orient == Qt.Horizontal else Qt.Horizontal
        )

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # Key handling for convenience
    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_O, Qt.Key_Space):
            self.toggle_orientation()
            e.accept()
            return
        if e.key() in (Qt.Key_F, Qt.Key_F11):
            self.toggle_fullscreen()
            e.accept()
            return
        if e.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.close()
            e.accept()
            return
        super().keyPressEvent(e)


# ------------------------------ Main ------------------------------ #
def parse_args(argv):
    p = argparse.ArgumentParser(
        description="View two images side-by-side or stacked with auto fit."
    )
    p.add_argument("path1", type=Path, help="First image path (PNG/JPG/etc.)")
    p.add_argument("path2", type=Path, help="Second image path")
    p.add_argument(
        "-o",
        "--orientation",
        choices=["horizontal", "vertical", "h", "v", "side", "stack"],
        default="horizontal",
        help="Initial layout orientation",
    )
    return p.parse_args(argv)


def normalize_orientation(s: str) -> str:
    s = s.lower()
    if s in ("h", "horizontal", "side"):
        return "horizontal"
    return "vertical"


def main():
    args = parse_args(sys.argv[1:])
    for p in (args.path1, args.path2):
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(2)

    app = QApplication(sys.argv)
    # High-DPI is automatic in Qt6, but ensure pixmaps follow device pixel ratio
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    viewer = TwoImageViewer(
        args.path1, args.path2, normalize_orientation(args.orientation)
    )
    viewer.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
