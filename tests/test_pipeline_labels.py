"""Label normalization and model label-space coverage."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest
from test_pipeline_support import MODEL_LABELS

from vineyard_disease.labels import (
    CANONICAL_CLASSES,
    PLANTVILLAGE_GRAPE_CLASSES,
    duplicate_target_classes,
    missing_target_classes,
    model_label_map,
    normalize_label,
)


def test_canonical_order_is_stable():
    assert CANONICAL_CLASSES == ("black_rot", "esca", "isariopsis_leaf_spot", "healthy")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Grape___Black_rot", "black_rot"),
        ("Grape___Esca_(Black_Measles)", "esca"),
        ("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "isariopsis_leaf_spot"),
        ("Grape___healthy", "healthy"),
        ("Grape with Black Rot", "black_rot"),
        ("Grape with Esca (Black Measles)", "esca"),
        ("Grape with Isariopsis Leaf Spot", "isariopsis_leaf_spot"),
        ("Healthy Grape Plant", "healthy"),
        ("black_rot", "black_rot"),
        ("Black Rot", "black_rot"),
        ("ESCA", "esca"),
        ("black-measles", "esca"),
        ("isariopsis_leaf_spot", "isariopsis_leaf_spot"),
        ("Leaf blight (Isariopsis leaf spot)", "isariopsis_leaf_spot"),
        ("healthy", "healthy"),
        ("  grape__healthy ", "healthy"),
    ],
)
def test_normalize_known_labels(raw, expected):
    assert normalize_label(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "Apple___Black_rot",
        "Apple with Black Rot",
        "Healthy Apple",
        "Tomato___healthy",
        "leaf blight",
        "powdery_mildew",
        "",
        None,
        "Grape___Powdery_mildew",
    ],
)
def test_unknown_or_other_crop_labels_are_none(raw):
    assert normalize_label(raw) is None


def test_require_crop_rejects_crop_free_aliases():
    assert normalize_label("healthy", require_crop=True) is None
    assert normalize_label("black_rot", require_crop=True) is None
    assert normalize_label("Healthy Grape Plant", require_crop=True) == "healthy"
    assert normalize_label("Grape___Black_rot", require_crop=True) == "black_rot"


def test_plantvillage_table_covers_all_classes():
    assert sorted(PLANTVILLAGE_GRAPE_CLASSES.values()) == sorted(CANONICAL_CLASSES)


def test_default_model_labels_map_exactly_four_grape_indices():
    mapping = model_label_map(MODEL_LABELS)
    assert len(MODEL_LABELS) == 38
    assert mapping == {11: "black_rot", 12: "esca", 13: "isariopsis_leaf_spot", 14: "healthy"}
    assert missing_target_classes(mapping) == []
    assert duplicate_target_classes(mapping) == []


def test_missing_and_duplicate_coverage_detected():
    labels = [lab for lab in MODEL_LABELS if lab != "Healthy Grape Plant"] + ["Grape___Black_rot"]
    mapping = model_label_map(labels)
    assert missing_target_classes(mapping) == ["healthy"]
    assert duplicate_target_classes(mapping) == ["black_rot"]
