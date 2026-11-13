"""Quality checks, gating, predict and evaluate with a mock classifier."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest

pytest.importorskip("PIL")

from test_pipeline_support import MODEL_LABELS, MockClassifier, dist, png_bytes

from vineyard_disease.errors import InputError, UnsupportedModelError
from vineyard_disease.evaluate import is_synthetic, run_evaluate
from vineyard_disease.gating import GateConfig, LabelSpace, decide
from vineyard_disease.manifest import read_manifest, write_manifest
from vineyard_disease.metrics import compute_metrics
from vineyard_disease.predict import collect_inputs, predict_iter
from vineyard_disease.quality import QualityConfig, flag_duplicates, inspect_image_bytes
from vineyard_disease.roi import parse_roi_string

SPACE = LabelSpace(MODEL_LABELS)


# -- quality ------------------------------------------------------------------------
def test_quality_ok_and_soft_issues():
    ok = inspect_image_bytes(png_bytes(size=(256, 256)))
    assert ok.ok and ok.issues == [] and (ok.width, ok.height) == (256, 256)
    soft = inspect_image_bytes(png_bytes(size=(64, 400), noise=False))
    assert soft.ok and {"small", "low_contrast", "extreme_aspect"} <= set(soft.issues)
    gray = inspect_image_bytes(png_bytes(size=(200, 200), mode="L"))
    assert "not_color" in gray.issues


def test_quality_hard_failures():
    bad = inspect_image_bytes(b"not an image")
    assert not bad.ok and bad.issues == ["undecodable"] and bad.error
    tiny = inspect_image_bytes(png_bytes(size=(8, 8)))
    assert not tiny.ok and "too_small" in tiny.issues
    assert inspect_image_bytes(png_bytes(size=(8, 8)), QualityConfig(min_side=4, warn_side=4)).ok


def test_duplicate_flags_and_cross_split_leakage():
    recs = [
        {"sha256": "x", "split": "train", "quality": {"ok": True, "issues": []}},
        {"sha256": "x", "split": "test", "quality": {"ok": True, "issues": []}},
        {"sha256": "y", "split": "test", "quality": {"ok": True, "issues": []}},
    ]
    summary = flag_duplicates(recs)
    assert summary == {"duplicate_sets": 1, "duplicate_sets_across_splits": 1}
    assert recs[0]["quality"]["issues"] == ["duplicate_across_splits", "duplicate_sha256"]
    assert recs[2]["quality"]["issues"] == []


# -- gating ----------------------------------------------------------------------------
def test_accepted_uses_unrenormalized_full_softmax():
    probs = dist({"Grape with Black Rot": 0.6, "Grape with Esca (Black Measles)": 0.1})
    out = decide(probs, SPACE, GateConfig(min_score=0.5))
    assert out["status"] == "accepted" and out["predicted_class"] == "black_rot"
    assert out["score"] == pytest.approx(0.6)
    assert out["grape_mass"] < 1.0
    assert sum(out["grape_scores"].values()) == pytest.approx(out["grape_mass"])
    assert out["prob_sum"] == pytest.approx(1.0)
    assert len(out["top_k"]) == 5 and out["top_k"][0]["label"] == "Grape with Black Rot"
    assert "not a diagnosis" in out["score_semantics"]


def test_non_grape_top1_is_never_mapped_to_grape():
    # Grape labels hold most of the *grape* mass, but the top-1 overall is tomato.
    probs = dist({"Tomato with Septoria Leaf Spot": 0.5, "Grape with Esca (Black Measles)": 0.3})
    out = decide(probs, SPACE, GateConfig(min_score=0.1))
    assert out["status"] == "non_grape" and out["predicted_class"] is None and out["score"] is None
    assert out["top_label"] == "Tomato with Septoria Leaf Spot"
    assert out["top_grape_candidate"] == "esca"
    assert out["grape_scores"]["esca"] == pytest.approx(0.3)  # not renormalized to 1.0


def test_low_score_and_grape_mass_gates():
    probs = dist({"Grape with Esca (Black Measles)": 0.3, "Grape with Black Rot": 0.25})
    assert decide(probs, SPACE, GateConfig(min_score=0.5))["status"] == "low_score"
    assert decide(probs, SPACE, GateConfig(min_score=0.2))["status"] == "accepted"
    assert decide(probs, SPACE, GateConfig(min_score=0.2, min_grape_mass=0.9))["status"] == "low_score"


def test_unsupported_model_label_space():
    labels = [lab for lab in MODEL_LABELS if lab != "Grape with Isariopsis Leaf Spot"]
    space = LabelSpace(labels)
    assert not space.supported and space.missing == ["isariopsis_leaf_spot"]
    out = decide(dist({"Grape with Black Rot": 0.9}, labels), space, GateConfig())
    assert out["status"] == "unsupported_model" and out["predicted_class"] is None
    with pytest.raises(UnsupportedModelError):
        space.require_supported()
    assert space.describe()["missing_target_classes"] == ["isariopsis_leaf_spot"]


def test_decide_validates_vectors_and_config():
    with pytest.raises(ValueError):
        decide([0.5, 0.5], SPACE, GateConfig())
    with pytest.raises(ValueError):
        decide([float("nan")] * 38, SPACE, GateConfig())
    with pytest.raises(ValueError):
        GateConfig(min_score=1.5)
    with pytest.raises(ValueError):
        GateConfig(top_k=0)


def test_zero_grape_mass_has_no_candidate():
    probs = [0.0] * 38
    probs[0] = 1.0
    out = decide(probs, SPACE, GateConfig())
    assert out["status"] == "non_grape" and out["top_grape_candidate"] is None


# -- predict ---------------------------------------------------------------------------
def _img(tmp_path, name, color, size=(64, 64)):
    p = tmp_path / name
    p.write_bytes(png_bytes(color, size))
    return p


def test_predict_files_dirs_errors_and_batches(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    _img(d, "a.png", (200, 0, 0))
    _img(d, "b.png", (0, 0, 200))
    (d / "c.png").write_bytes(b"broken")
    (d / "readme.txt").write_text("x")
    clf = MockClassifier()
    rows = list(predict_iter(collect_inputs([str(d)]), clf, GateConfig(), batch_size=2))
    assert [r["status"] for r in rows] == ["accepted", "non_grape", "input_error"]
    assert rows[0]["predicted_class"] == "black_rot" and rows[0]["crop_mode"] == "full_frame_single_leaf_crop_only"
    assert len(rows[0]["all_scores"]) == 38
    assert "error" in rows[2] and rows[2]["predicted_class"] is None
    assert clf.calls == [2]  # broken image excluded from the second batch
    assert all("not a plant-health diagnosis" in r["disclaimer"] for r in rows)


def test_predict_roi_crop(tmp_path):
    p = _img(tmp_path, "frame.png", (0, 0, 200), size=(100, 80))
    from PIL import Image

    img = Image.open(p).convert("RGB")
    img.paste((0, 200, 0), (10, 10, 50, 50))
    img.save(p)
    clf = MockClassifier()
    roi = parse_roi_string("10,10,40,40")
    [row] = predict_iter(collect_inputs([str(p)]), clf, GateConfig(), roi=roi, all_scores=False)
    assert row["predicted_class"] == "healthy" and clf.sizes == [(40, 40)]
    assert "all_scores" not in row
    [bad] = predict_iter(collect_inputs([str(p)]), clf, GateConfig(), roi=parse_roi_string("90,70,40,40"))
    assert bad["status"] == "input_error"
    norm = parse_roi_string("0.1,0.125,0.4,0.5@norm")
    assert norm.box(100, 80) == (10, 10, 50, 50)


def test_predict_refuses_unsupported_model(tmp_path):
    p = _img(tmp_path, "a.png", (200, 0, 0))
    clf = MockClassifier(labels=MODEL_LABELS[:11])
    with pytest.raises(UnsupportedModelError):
        list(predict_iter(collect_inputs([str(p)]), clf, GateConfig()))


def test_collect_inputs_validation(tmp_path):
    with pytest.raises(InputError):
        collect_inputs([])
    with pytest.raises(InputError):
        collect_inputs([str(tmp_path / "missing.png")])
    with pytest.raises(InputError):
        collect_inputs([str(tmp_path)], manifest="m.jsonl")
    (tmp_path / "sub").mkdir()
    _img(tmp_path / "sub", "x.png", (1, 1, 1))
    with pytest.raises(InputError, match="no input"):
        collect_inputs([str(tmp_path)])
    assert len(collect_inputs([str(tmp_path)], recursive=True)) == 1


# -- manifest + evaluate ------------------------------------------------------------
COLORS = {
    "black_rot": (200, 0, 0),
    "healthy": (0, 200, 0),
    "esca": (128, 128, 128),
    "isariopsis_leaf_spot": (200, 200, 0),
}


def make_manifest(tmp_path, labels, split="test", synthetic=False):
    recs = []
    for i, label in enumerate(labels):
        rel = f"images/{i}.png"
        (tmp_path / "images").mkdir(exist_ok=True)
        (tmp_path / rel).write_bytes(png_bytes(COLORS[label]))
        recs.append(
            {
                "schema_version": 1,
                "id": f"r{i:03d}",
                "image": rel,
                "label": label,
                "split": split,
                "source_split": split,
                "group_id": None,
                "grouping_source": "none",
                "source": {"kind": "simulation" if synthetic else "local", "path": rel},
                "synthetic": synthetic,
                "quality": {"ok": True, "issues": []},
            }
        )
    write_manifest(tmp_path / "manifest.jsonl", recs)
    return tmp_path / "manifest.jsonl"


def test_manifest_roundtrip_and_validation(tmp_path):
    m = make_manifest(tmp_path, ["esca", "healthy"])
    recs = read_manifest(m)
    assert [r["id"] for r in recs] == ["r000", "r001"]
    assert os.path.isabs(recs[0]["_image_path"])
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x", "image": "a", "label": "powdery", "split": "test"}) + "\n")
    with pytest.raises(InputError, match="canonical"):
        read_manifest(bad)
    bad.write_text("{not json\n")
    with pytest.raises(InputError, match="invalid JSON"):
        read_manifest(bad)
    with pytest.raises(InputError, match="duplicate"):
        write_manifest(tmp_path / "dup.jsonl", [recs[0], recs[0]])


def test_evaluate_with_mock_classifier(tmp_path):
    labels = ["black_rot", "black_rot", "healthy", "esca", "isariopsis_leaf_spot"]
    m = make_manifest(tmp_path, labels)
    report = run_evaluate(
        str(m), str(tmp_path / "eval"), split="test", classifier=MockClassifier(), gate=GateConfig(min_score=0.5)
    )
    met = report["metrics"]
    assert met["n"] == 5
    assert report["status_counts"]["low_score"] == 1  # esca colour -> 0.3 top score
    assert met["per_class"]["black_rot"]["recall"] == 1.0
    assert met["per_class"]["esca"]["recall"] == 0.0
    assert met["per_class"]["esca"]["precision"] is None  # never predicted: undefined, not 0
    assert met["coverage"] == pytest.approx(0.8)
    assert met["selective_accuracy"] == 1.0
    assert (tmp_path / "eval" / "predictions.jsonl").is_file()
    saved = json.loads((tmp_path / "eval" / "metrics.json").read_text())
    assert saved["claims"].startswith("in-distribution")
    assert any("New Plant Diseases" in c for c in saved["caveats"])
    # Re-evaluate from saved predictions (no classifier).
    again = run_evaluate(str(m), str(tmp_path / "eval2"), predictions=str(tmp_path / "eval" / "predictions.jsonl"))
    assert again["metrics"]["confusion"] == met["confusion"]


def test_evaluate_missing_predictions_and_synthetic_claims(tmp_path):
    m = make_manifest(tmp_path, ["healthy", "esca"], synthetic=True)
    preds = tmp_path / "p.jsonl"
    preds.write_text(
        json.dumps({"id": "r000", "status": "accepted", "predicted_class": "healthy"})
        + "\n"
        + json.dumps({"id": "other", "status": "non_grape"})
        + "\n"
    )
    report = run_evaluate(str(m), str(tmp_path / "e"), predictions=str(preds))
    assert report["claims"] == "none"
    assert report["caveats"][0].startswith("2 of 2 images are synthetic")
    assert report["metrics"]["confusion"]["esca"]["missing"] == 1
    assert report["unmatched_predictions"] == 1
    assert is_synthetic({"source": {"kind": "gazebo"}})


def test_evaluate_input_validation(tmp_path):
    m = make_manifest(tmp_path, ["healthy"], split="train")
    with pytest.raises(InputError, match="no manifest records"):
        run_evaluate(str(m), str(tmp_path / "e"), split="test", classifier=MockClassifier())
    with pytest.raises(InputError, match="--predictions or a classifier"):
        run_evaluate(str(m), str(tmp_path / "e"), split="train")
    dup = tmp_path / "d.jsonl"
    dup.write_text(
        json.dumps({"id": "r000", "status": "low_score"})
        + "\n"
        + json.dumps({"id": "r000", "status": "low_score"})
        + "\n"
    )
    with pytest.raises(InputError, match="duplicate"):
        run_evaluate(str(m), str(tmp_path / "e"), split="train", predictions=str(dup))


def test_compute_metrics_definitions():
    rows = [
        ("esca", "accepted", "esca"),
        ("esca", "accepted", "healthy"),
        ("healthy", "non_grape", None),
        ("healthy", "accepted", "healthy"),
        ("black_rot", "low_score", None),
    ]
    m = compute_metrics(rows)
    assert m["per_class"]["esca"] == {
        "support": 2,
        "predicted": 1,
        "tp": 1,
        "fp": 0,
        "fn": 1,
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "coverage": 1.0,
    }
    assert m["per_class"]["healthy"]["precision"] == 0.5
    assert m["per_class"]["black_rot"]["f1"] is None  # no predictions -> undefined precision
    assert m["per_class"]["isariopsis_leaf_spot"]["support"] == 0
    assert m["accuracy"] == pytest.approx(2 / 5) and m["selective_accuracy"] == pytest.approx(2 / 3)
    assert m["macro_f1_classes"] == ["esca", "healthy"]
    with pytest.raises(ValueError):
        compute_metrics([("powdery", "accepted", "esca")])
    with pytest.raises(ValueError):
        compute_metrics([("esca", "accepted", None)])
    assert compute_metrics([])["accuracy"] is None
