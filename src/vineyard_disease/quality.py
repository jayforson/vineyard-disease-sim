"""Image quality checks used during data preparation.

Hard failures exclude an image from a manifest; soft issues are kept as
flags on the record so downstream consumers can decide.
"""

from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import dataclass, field

from .errors import require

HARD_ISSUES = frozenset({"undecodable", "too_small", "decompression_bomb"})


@dataclass(frozen=True)
class QualityConfig:
    min_side: int = 32  # hard reject below this
    warn_side: int = 128  # soft flag below this (model input is 224 px)
    min_gray_stddev: float = 4.0  # soft flag: near-constant image
    max_aspect: float = 4.0  # soft flag: extreme aspect ratio


@dataclass
class QualityReport:
    sha256: str
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    format: str | None = None
    issues: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return not any(i in HARD_ISSUES for i in self.issues)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "issues": sorted(set(self.issues))}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_image_bytes(data: bytes, config: QualityConfig | None = None) -> QualityReport:
    """Decode ``data`` with Pillow and report size/mode/quality issues."""
    config = config or QualityConfig()
    report = QualityReport(sha256=sha256_bytes(data))
    image_mod = require("PIL.Image", "image quality checks", "Install Pillow.")
    stat_mod = require("PIL.ImageStat", "image quality checks", "Install Pillow.")
    try:
        with warnings.catch_warnings():
            # Treat Pillow's oversize-image warning as a hard failure too.
            warnings.simplefilter("error", image_mod.DecompressionBombWarning)
            with image_mod.open(io.BytesIO(data)) as probe:
                probe.verify()
            with image_mod.open(io.BytesIO(data)) as img:
                img.load()
                report.width, report.height = img.size
                report.mode = img.mode
                report.format = img.format
                gray = img.convert("L")
                stddev = stat_mod.Stat(gray).stddev[0]
    except (image_mod.DecompressionBombError, image_mod.DecompressionBombWarning) as exc:
        report.issues.append("decompression_bomb")
        report.error = f"{type(exc).__name__}: {exc}"
        return report
    except Exception as exc:  # noqa: BLE001 - Pillow raises many exception types for corrupt data
        report.issues.append("undecodable")
        report.error = f"{type(exc).__name__}: {exc}"
        return report

    short, long_ = sorted((report.width, report.height))
    if short < config.min_side:
        report.issues.append("too_small")
    elif short < config.warn_side:
        report.issues.append("small")
    if short > 0 and long_ / short > config.max_aspect:
        report.issues.append("extreme_aspect")
    if stddev < config.min_gray_stddev:
        report.issues.append("low_contrast")
    if report.mode in ("1", "L", "LA", "I", "I;16", "F"):
        report.issues.append("not_color")
    return report


def flag_duplicates(records: list[dict]) -> dict:
    """Flag records sharing image bytes; returns a summary dict.

    Adds ``duplicate_sha256`` to every member of a duplicate set and
    ``duplicate_across_splits`` when a duplicate set spans splits (leakage).
    """
    by_hash: dict[str, list[dict]] = {}
    for rec in records:
        by_hash.setdefault(rec["sha256"], []).append(rec)
    dup_sets = 0
    cross = 0
    for members in by_hash.values():
        if len(members) < 2:
            continue
        dup_sets += 1
        splits = {m.get("split") for m in members}
        for m in members:
            issues = set(m["quality"]["issues"])
            issues.add("duplicate_sha256")
            if len(splits) > 1:
                issues.add("duplicate_across_splits")
            m["quality"]["issues"] = sorted(issues)
        if len(splits) > 1:
            cross += 1
    return {"duplicate_sets": dup_sets, "duplicate_sets_across_splits": cross}
