"""``predict``: run the classifier on leaf-crop images and gate the outputs."""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from . import DISCLAIMER
from .errors import InputError, require
from .gating import GateConfig, LabelSpace, decide
from .manifest import read_manifest
from .roi import FULL_FRAME_MODE, Roi, RoiError, crop
from .sources.local_dir import IMAGE_SUFFIXES

PREDICTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PredictInput:
    id: str
    path: str  # filesystem path to open
    image: str  # path as reported in outputs (as given / manifest-relative)


def _file_id(path: str) -> str:
    return "file-" + hashlib.sha256(path.encode()).hexdigest()[:16]


def collect_inputs(
    paths: Sequence[str] = (),
    *,
    manifest: str | None = None,
    split: str | None = None,
    recursive: bool = False,
    limit: int | None = None,
) -> list[PredictInput]:
    """Collect images from explicit paths/directories or from a manifest.

    Manifest labels are deliberately not read here: predictions never see
    ground truth (simulation or dataset labels).
    """
    if bool(paths) == bool(manifest):
        raise InputError("give image paths/directories or --manifest (exactly one)")
    out: list[PredictInput] = []
    if manifest:
        for rec in read_manifest(manifest, split=split):
            out.append(PredictInput(rec["id"], rec["_image_path"], rec["image"]))
    else:
        for p in paths:
            path = Path(p)
            if path.is_dir():
                it = path.rglob("*") if recursive else path.iterdir()
                files = sorted(
                    f for f in it if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES and not f.name.startswith(".")
                )
                out.extend(PredictInput(_file_id(str(f)), str(f), str(f)) for f in files)
            elif path.is_file():
                out.append(PredictInput(_file_id(str(path)), str(path), str(path)))
            else:
                raise InputError(f"input not found: {p}")
    if limit is not None:
        out = out[:limit]
    if not out:
        raise InputError("no input images found")
    return out


def _load_image(path: str, roi: Roi | None):
    image_mod = require("PIL.Image", "loading images", "Install Pillow.")
    ops = require("PIL.ImageOps", "loading images", "Install Pillow.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", image_mod.DecompressionBombWarning)
        with image_mod.open(path) as img:
            img.load()
            img = ops.exif_transpose(img)
            return crop(img.convert("RGB"), roi)


def _bomb_errors() -> tuple[type[BaseException], ...]:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - _load_image raises MissingDependencyError first
        return ()
    return (Image.DecompressionBombError, Image.DecompressionBombWarning)


def _batches(items: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def predict_iter(
    inputs: Sequence[PredictInput],
    classifier,
    cfg: GateConfig,
    *,
    roi: Roi | None = None,
    batch_size: int = 8,
    all_scores: bool = True,
) -> Iterator[dict]:
    """Yield one prediction row per input, in input order."""
    if batch_size < 1:
        raise InputError("--batch-size must be >= 1")
    space = LabelSpace(classifier.labels)
    space.require_supported()
    model = {k: classifier.info.get(k) for k in ("model_id", "revision", "device", "num_labels")}
    crop_mode = "roi" if roi is not None else FULL_FRAME_MODE
    for batch in _batches(list(inputs), batch_size):
        loaded, errors = [], {}
        for item in batch:
            try:
                loaded.append((item, _load_image(item.path, roi)))
            except (OSError, RoiError, ValueError, *_bomb_errors()) as exc:
                errors[item.id] = f"{type(exc).__name__}: {exc}"
        probs = classifier.predict_proba([img for _, img in loaded]) if loaded else []
        by_id = {item.id: p for (item, _), p in zip(loaded, probs)}
        for item in batch:
            row = {
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "id": item.id,
                "image": item.image,
                "crop_mode": crop_mode,
                "roi": roi.as_dict() if roi else None,
                "model": model,
                "disclaimer": DISCLAIMER,
            }
            if item.id in errors:
                row.update({"status": "input_error", "predicted_class": None, "score": None, "error": errors[item.id]})
            else:
                p = by_id[item.id]
                row.update(decide(p, space, cfg))
                if all_scores:
                    row["all_scores"] = {label: round(float(s), 6) for label, s in zip(space.labels, p)}
            yield row


def write_jsonl(rows: Iterable[dict], out: IO[str]) -> int:
    n = 0
    for row in rows:
        out.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        n += 1
    out.flush()
    return n


def open_output(path: str | None):
    if path in (None, "-"):
        return _NoClose(sys.stdout)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.open("w", encoding="utf-8")


class _NoClose:
    def __init__(self, fh):
        self.fh = fh

    def __enter__(self):
        return self.fh

    def __exit__(self, *exc):
        self.fh.flush()
        return False
