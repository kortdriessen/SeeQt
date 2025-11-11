from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import json
import os
from textwrap import dedent

import yaml


@dataclass
class VideoSpec:
    """Configuration for an individual video clip."""

    id: str
    video_path: Path
    frame_times_path: Path
    name: Optional[str] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["video_path"] = str(self.video_path)
        data["frame_times_path"] = str(self.frame_times_path)
        return data

    @classmethod
    def from_dict(cls, data: dict, base_dir: Optional[Path] = None) -> "VideoSpec":
        video_path = Path(data["video_path"])
        frame_times_path = Path(data["frame_times_path"])
        if base_dir and not video_path.is_absolute():
            video_path = (base_dir / video_path).resolve()
        if base_dir and not frame_times_path.is_absolute():
            frame_times_path = (base_dir / frame_times_path).resolve()
        return cls(
            id=str(data["id"]),
            video_path=video_path,
            frame_times_path=frame_times_path,
            name=data.get("name"),
        )


@dataclass
class ContainerSpec:
    """Placement information for a single video container."""

    id: str
    geometry: Tuple[int, int, int, int]  # x, y, width, height
    video_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "geometry": list(self.geometry),
            "video_id": self.video_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContainerSpec":
        geom = data.get("geometry", [0, 0, 320, 240])
        if len(geom) != 4:
            raise ValueError(f"Invalid geometry {geom!r}; expected 4 numbers")
        return cls(
            id=str(data["id"]),
            geometry=tuple(int(v) for v in geom),
            video_id=data.get("video_id"),
        )


@dataclass
class ViewerConfig:
    """Full configuration for the viewer."""

    videos: List[VideoSpec] = field(default_factory=list)
    containers: List[ContainerSpec] = field(default_factory=list)
    canvas_size: Optional[Tuple[int, int]] = None
    timeline_start: Optional[float] = None
    timeline_end: Optional[float] = None

    def to_dict(self, relative_to: Optional[Path] = None) -> dict:
        def maybe_rel(path: Path) -> str:
            if relative_to:
                try:
                    return str(path.relative_to(relative_to))
                except ValueError:
                    pass
            return str(path)

        videos = []
        for v in self.videos:
            entry = v.to_dict()
            entry["video_path"] = maybe_rel(Path(entry["video_path"]))
            entry["frame_times_path"] = maybe_rel(Path(entry["frame_times_path"]))
            videos.append(entry)

        data = {
            "videos": videos,
            "containers": [c.to_dict() for c in self.containers],
        }
        if self.canvas_size:
            data["canvas_size"] = list(self.canvas_size)
        if self.timeline_start is not None:
            data["timeline"] = {
                "start": float(self.timeline_start),
                "end": float(self.timeline_end) if self.timeline_end is not None else None,
            }
        return data

    def save(self, path: Path) -> None:
        path = Path(path)
        data = self.to_dict(relative_to=path.parent)
        if path.suffix.lower() in {".yml", ".yaml"}:
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        else:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ViewerConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yml", ".yaml"}:
            raw = yaml.safe_load(text) or {}
        else:
            raw = json.loads(text or "{}")
        base_dir = path.parent
        videos = [VideoSpec.from_dict(v, base_dir) for v in raw.get("videos", [])]
        containers = [ContainerSpec.from_dict(c) for c in raw.get("containers", [])]
        canvas_size = raw.get("canvas_size")
        timeline_data = raw.get("timeline") or {}
        return cls(
            videos=videos,
            containers=containers,
            canvas_size=tuple(canvas_size) if canvas_size else None,
            timeline_start=timeline_data.get("start"),
            timeline_end=timeline_data.get("end"),
        )

    def video_by_id(self, video_id: str) -> Optional[VideoSpec]:
        for video in self.videos:
            if video.id == video_id:
                return video
        return None


DEFAULT_CONFIG_TEMPLATE = dedent(
    """
    # Example SeeQt video viewer configuration
    videos:
      - id: camera-1
        name: Top Camera
        video_path: ./camera1.mp4
        frame_times_path: ./camera1_frame_times.npy
      - id: camera-2
        name: Side Camera
        video_path: ./camera2.mp4
        frame_times_path: ./camera2_frame_times.npy
    containers:
      - id: large
        geometry: [0, 0, 900, 600]
        video_id: camera-1
      - id: small
        geometry: [920, 0, 480, 360]
        video_id: camera-2
    timeline:
      start: 0.0
      end: null
    """
).strip()


def write_template(path: Path) -> None:
    """Create a starter configuration file."""
    Path(path).write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
