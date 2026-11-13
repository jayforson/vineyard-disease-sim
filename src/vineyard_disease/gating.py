"""Decision gating over the full model label distribution.

The classifier scores all of its labels (38 for the default checkpoint).
A grape class is reported only when the *top-1 label overall* is a grape
label and its unmodified softmax probability clears ``min_score``. Grape
scores are never renormalized over the four grape classes, because that
would turn "probably not a grape leaf" into fake grape confidence.

Statuses:

* ``accepted``          top-1 is a grape label and passes the gates
* ``non_grape``         top-1 label is not one of the grape labels
* ``low_score``         top-1 is grape but below ``min_score`` / ``min_grape_mass``
* ``unsupported_model`` the model label space misses a target grape class
* ``input_error``       the image could not be read (set by callers)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .errors import UnsupportedModelError
from .labels import CANONICAL_CLASSES, duplicate_target_classes, missing_target_classes, model_label_map

STATUSES = ("accepted", "non_grape", "low_score", "unsupported_model", "input_error")
SCORE_SEMANTICS = (
    "uncalibrated softmax probability over the model's full label set (not renormalized to grape "
    "classes); screening signal only, not a diagnosis"
)


@dataclass(frozen=True)
class GateConfig:
    min_score: float = 0.5
    min_grape_mass: float = 0.0
    top_k: int = 5

    def __post_init__(self):
        if not 0.0 <= self.min_score <= 1.0 or not 0.0 <= self.min_grape_mass <= 1.0:
            raise ValueError("thresholds must be within [0, 1]")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")


class LabelSpace:
    """Model labels plus their grape mapping and coverage diagnostics."""

    def __init__(self, labels: Sequence[str]):
        self.labels = list(labels)
        self.grape = model_label_map(self.labels)
        self.missing = missing_target_classes(self.grape)
        self.duplicates = duplicate_target_classes(self.grape)

    @property
    def supported(self) -> bool:
        return not self.missing and not self.duplicates

    def describe(self) -> dict:
        return {
            "num_labels": len(self.labels),
            "grape_labels": {self.labels[i]: c for i, c in sorted(self.grape.items())},
            "missing_target_classes": self.missing,
            "ambiguous_target_classes": self.duplicates,
        }

    def require_supported(self) -> None:
        if not self.supported:
            raise UnsupportedModelError(
                "model label space does not cover the grape classes exactly once: "
                f"missing={self.missing} ambiguous={self.duplicates}"
            )


def decide(probs: Sequence[float], space: LabelSpace, cfg: GateConfig) -> dict:
    """Gate one probability vector; returns a JSON-serializable dict."""
    if len(probs) != len(space.labels):
        raise ValueError(f"got {len(probs)} scores for {len(space.labels)} labels")
    if any(not math.isfinite(p) or p < 0 for p in probs):
        raise ValueError("scores must be finite and non-negative")
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    top = order[0]
    grape_scores = {c: 0.0 for c in CANONICAL_CLASSES}
    for idx, canonical in space.grape.items():
        grape_scores[canonical] += float(probs[idx])
    grape_mass = sum(grape_scores.values())
    top_class = space.grape.get(top)
    out = {
        "status": None,
        "predicted_class": None,
        "score": None,
        "top_label": space.labels[top],
        "top_score": float(probs[top]),
        "top_grape_candidate": (max(CANONICAL_CLASSES, key=lambda c: grape_scores[c]) if grape_mass > 0 else None),
        "grape_scores": grape_scores,
        "grape_mass": grape_mass,
        "top_k": [{"label": space.labels[i], "score": float(probs[i])} for i in order[: cfg.top_k]],
        "score_semantics": SCORE_SEMANTICS,
        "thresholds": {"min_score": cfg.min_score, "min_grape_mass": cfg.min_grape_mass},
        "prob_sum": float(sum(probs)),
    }
    if not space.supported:
        out["status"] = "unsupported_model"
    elif top_class is None:
        out["status"] = "non_grape"
    elif probs[top] < cfg.min_score or grape_mass < cfg.min_grape_mass:
        out["status"] = "low_score"
    else:
        out["status"] = "accepted"
        out["predicted_class"] = top_class
        out["score"] = float(probs[top])
    return out


def status_counts(rows: Sequence[Mapping]) -> dict[str, int]:
    counts = {s: 0 for s in STATUSES}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts
