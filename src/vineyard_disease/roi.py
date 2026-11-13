"""Regions of interest for leaf-crop inference.

The classifier only sees single leaf crops. A crop comes from either

* a *known ROI config*: fixed pixel (or normalized) boxes for a fixed camera,
  e.g. produced by the simulation layout or a calibration step; or
* *full-frame mode*: the whole image is used and the caller asserts that it
  is already a single-leaf crop (``FULL_FRAME_MODE``). Whole-canopy frames
  are out of scope; there is no detector in this pipeline.

ROI config (JSON; YAML if PyYAML is installed)::

    {"image_width": 640, "image_height": 480, "normalized": false,
     "rois": [{"id": "leaf_0", "x": 100, "y": 80, "width": 160, "height": 160}]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import InputError

FULL_FRAME_MODE = "full_frame_single_leaf_crop_only"
ROI_MODE = "roi_config"


class RoiError(ValueError):
    """An ROI cannot be applied to the given frame."""


@dataclass(frozen=True)
class Roi:
    id: str
    x: float
    y: float
    width: float
    height: float
    normalized: bool = False

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0 or self.x < 0 or self.y < 0:
            raise InputError(f"ROI '{self.id}' must have x,y >= 0 and width,height > 0")
        if self.normalized and (self.x + self.width > 1.0 + 1e-9 or self.y + self.height > 1.0 + 1e-9):
            raise InputError(f"normalized ROI '{self.id}' exceeds the unit square")

    def box(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Pixel box (left, top, right, bottom); raises RoiError if out of frame."""
        if self.normalized:
            left, top = round(self.x * frame_w), round(self.y * frame_h)
            right, bottom = round((self.x + self.width) * frame_w), round((self.y + self.height) * frame_h)
        else:
            left, top = int(self.x), int(self.y)
            right, bottom = int(self.x + self.width), int(self.y + self.height)
        if right > frame_w or bottom > frame_h or right <= left or bottom <= top:
            raise RoiError(f"ROI '{self.id}' {left, top, right, bottom} is outside the {frame_w}x{frame_h} frame")
        return left, top, right, bottom

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "normalized": self.normalized,
        }


@dataclass(frozen=True)
class RoiConfig:
    rois: tuple[Roi, ...]
    image_width: int | None = None
    image_height: int | None = None

    def check_frame(self, frame_w: int, frame_h: int) -> None:
        if self.image_width and self.image_height and (frame_w, frame_h) != (self.image_width, self.image_height):
            raise RoiError(
                f"frame is {frame_w}x{frame_h} but the ROI config was made for {self.image_width}x{self.image_height}"
            )


def parse_roi_string(text: str, roi_id: str = "roi_0") -> Roi:
    """Parse ``x,y,w,h`` (pixels) or ``x,y,w,h@norm`` (fractions)."""
    normalized = text.endswith("@norm")
    body = text[: -len("@norm")] if normalized else text
    try:
        x, y, w, h = (float(v) for v in body.split(","))
    except ValueError as exc:
        raise InputError(f"invalid ROI '{text}': expected x,y,width,height") from exc
    return Roi(roi_id, x, y, w, h, normalized)


def roi_config_from_dict(obj: dict) -> RoiConfig:
    if not isinstance(obj, dict) or not isinstance(obj.get("rois"), list) or not obj["rois"]:
        raise InputError("ROI config needs a non-empty 'rois' list")
    normalized = bool(obj.get("normalized", False))
    rois = []
    seen = set()
    for i, r in enumerate(obj["rois"]):
        try:
            roi = Roi(
                str(r.get("id", f"roi_{i}")),
                float(r["x"]),
                float(r["y"]),
                float(r["width"]),
                float(r["height"]),
                bool(r.get("normalized", normalized)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InputError(f"ROI #{i} needs numeric x, y, width, height") from exc
        if roi.id in seen:
            raise InputError(f"duplicate ROI id '{roi.id}'")
        seen.add(roi.id)
        rois.append(roi)
    w, h = obj.get("image_width"), obj.get("image_height")
    if (w is None) != (h is None):
        raise InputError("set both image_width and image_height, or neither")
    return RoiConfig(tuple(rois), int(w) if w else None, int(h) if h else None)


def load_roi_config(path: str | Path) -> RoiConfig:
    path = Path(path)
    if not path.is_file():
        raise InputError(f"ROI config not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise InputError("YAML ROI configs need PyYAML; use JSON instead") from exc
        obj = yaml.safe_load(text)
    else:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InputError(f"invalid ROI config JSON: {exc}") from exc
    return roi_config_from_dict(obj)


def crop(image, roi: Roi | None):
    """Crop a PIL image to ``roi`` (``None`` returns the image unchanged)."""
    if roi is None:
        return image
    return image.crop(roi.box(*image.size))
