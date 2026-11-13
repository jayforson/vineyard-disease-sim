"""Canonical grape-leaf class labels and label normalization.

Canonical classes (fixed order)::

    black_rot, esca, isariopsis_leaf_spot, healthy

Source names (PlantVillage folder names such as ``Grape___Black_rot``) and
classifier label strings (such as ``Grape with Black Rot``) are mapped onto
these names. Anything else, including other crops, normalizes to ``None``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

CANONICAL_CLASSES: tuple[str, ...] = (
    "black_rot",
    "esca",
    "isariopsis_leaf_spot",
    "healthy",
)

# Official PlantVillage class folder names (color/grayscale/segmented configs).
PLANTVILLAGE_GRAPE_CLASSES: Mapping[str, str] = {
    "Grape___Black_rot": "black_rot",
    "Grape___Esca_(Black_Measles)": "esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "isariopsis_leaf_spot",
    "Grape___healthy": "healthy",
}

# Grape labels of the default pretrained classifier (38-label head).
DEFAULT_MODEL_GRAPE_LABELS: Mapping[str, str] = {
    "Grape with Black Rot": "black_rot",
    "Grape with Esca (Black Measles)": "esca",
    "Grape with Isariopsis Leaf Spot": "isariopsis_leaf_spot",
    "Healthy Grape Plant": "healthy",
}

# Aliases after _key() normalization. Entries without "grape" are accepted
# only when require_crop=False (for class directories that are already
# grape-only, e.g. ``black_rot/``).
_GRAPE_ALIASES: Mapping[str, str] = {
    "grape black rot": "black_rot",
    "grape with black rot": "black_rot",
    "grape esca": "esca",
    "grape esca black measles": "esca",
    "grape with esca": "esca",
    "grape with esca black measles": "esca",
    "grape black measles": "esca",
    "grape leaf blight isariopsis leaf spot": "isariopsis_leaf_spot",
    "grape isariopsis leaf spot": "isariopsis_leaf_spot",
    "grape with isariopsis leaf spot": "isariopsis_leaf_spot",
    "grape leaf blight": "isariopsis_leaf_spot",
    "grape healthy": "healthy",
    "healthy grape": "healthy",
    "healthy grape plant": "healthy",
    "healthy grape leaf": "healthy",
    "grape leaf healthy": "healthy",
}

_CROP_FREE_ALIASES: Mapping[str, str] = {
    "black rot": "black_rot",
    "esca": "esca",
    "esca black measles": "esca",
    "black measles": "esca",
    "isariopsis leaf spot": "isariopsis_leaf_spot",
    "leaf blight isariopsis leaf spot": "isariopsis_leaf_spot",
    "isariopsis": "isariopsis_leaf_spot",
    "healthy": "healthy",
}


def _key(raw: str) -> str:
    text = raw.strip().lower().replace("___", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(raw: str | None, *, require_crop: bool = False) -> str | None:
    """Map a source or model label onto a canonical class, else ``None``.

    ``require_crop=True`` only accepts labels that name grape explicitly;
    use it for multi-crop label spaces (e.g. classifier outputs) so that
    ``Healthy Apple`` or ``Apple___Black_rot`` never map to a grape class.
    """
    if raw is None:
        return None
    if raw in PLANTVILLAGE_GRAPE_CLASSES:
        return PLANTVILLAGE_GRAPE_CLASSES[raw]
    if raw in DEFAULT_MODEL_GRAPE_LABELS:
        return DEFAULT_MODEL_GRAPE_LABELS[raw]
    if raw in CANONICAL_CLASSES and not require_crop:
        return raw
    key = _key(raw)
    if key in _GRAPE_ALIASES:
        return _GRAPE_ALIASES[key]
    if not require_crop:
        if key in _CROP_FREE_ALIASES:
            return _CROP_FREE_ALIASES[key]
        if key.replace(" ", "_") in CANONICAL_CLASSES:
            return key.replace(" ", "_")
    return None


def model_label_map(labels: Iterable[str]) -> dict[int, str]:
    """Return {label index: canonical class} for grape labels of a model."""
    mapping: dict[int, str] = {}
    for idx, label in enumerate(labels):
        canonical = normalize_label(label, require_crop=True)
        if canonical is not None:
            mapping[idx] = canonical
    return mapping


def missing_target_classes(mapping: Mapping[int, str]) -> list[str]:
    """Canonical classes that no model label maps to."""
    covered = set(mapping.values())
    return [c for c in CANONICAL_CLASSES if c not in covered]


def duplicate_target_classes(mapping: Mapping[int, str]) -> list[str]:
    """Canonical classes that more than one model label maps to."""
    seen: dict[str, int] = {}
    for canonical in mapping.values():
        seen[canonical] = seen.get(canonical, 0) + 1
    return sorted(c for c, n in seen.items() if n > 1)
