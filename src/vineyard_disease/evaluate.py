"""``evaluate``: score predictions against manifest labels."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import DISCLAIMER, __version__
from .classifier import MODEL_CAVEATS
from .errors import InputError
from .gating import GateConfig, status_counts
from .manifest import read_jsonl, read_manifest, write_json
from .metrics import compute_metrics
from .predict import collect_inputs, predict_iter, write_jsonl

SYNTHETIC_KINDS = frozenset({"synthetic", "simulation", "sim", "gazebo"})


def is_synthetic(rec: dict) -> bool:
    src = rec.get("source") or {}
    return bool(rec.get("synthetic")) or str(src.get("kind", "")).lower() in SYNTHETIC_KINDS


def _load_predictions(path: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for lineno, row in read_jsonl(path):
        if "id" not in row or "status" not in row:
            raise InputError(f"{path}:{lineno}: prediction rows need 'id' and 'status'")
        if row["id"] in rows:
            raise InputError(f"{path}:{lineno}: duplicate prediction id {row['id']}")
        rows[row["id"]] = row
    return rows


def run_evaluate(
    manifest: str,
    output_dir: str,
    *,
    split: str | None = "test",
    predictions: str | None = None,
    classifier=None,
    gate: GateConfig | None = None,
    batch_size: int = 8,
    limit: int | None = None,
) -> dict:
    gate = gate or GateConfig()
    records = read_manifest(manifest, split=split)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise InputError(f"no manifest records for split={split!r}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_info = None
    if predictions:
        preds = _load_predictions(predictions)
        predictions_path = predictions
    else:
        if classifier is None:
            raise InputError("evaluate needs --predictions or a classifier")
        inputs = collect_inputs(manifest=manifest, split=split)
        wanted = {r["id"] for r in records}
        inputs = [i for i in inputs if i.id in wanted]
        predictions_path = str(out_dir / "predictions.jsonl")
        rows = list(predict_iter(inputs, classifier, gate, batch_size=batch_size))
        with open(predictions_path, "w", encoding="utf-8") as fh:
            write_jsonl(rows, fh)
        preds = {r["id"]: r for r in rows}
        model_info = classifier.info

    triples = []
    joined = []
    for rec in records:
        row = preds.get(rec["id"])
        if row is None:
            triples.append((rec["label"], "missing", None))
            continue
        status = row["status"]
        pred = row.get("predicted_class") if status == "accepted" else None
        triples.append((rec["label"], status, pred))
        joined.append(row)
        if model_info is None and isinstance(row.get("model"), dict):
            model_info = row["model"]
    manifest_ids = {r["id"] for r in records}
    metrics = compute_metrics(triples)

    synthetic = sum(1 for r in records if is_synthetic(r))
    provenance = _sibling_provenance(manifest)
    source = (provenance or {}).get("source", {})
    caveats = list(MODEL_CAVEATS) + list(source.get("caveats", []))
    if synthetic:
        caveats.insert(
            0,
            f"{synthetic} of {len(records)} images are synthetic/simulated: these metrics check the "
            "pipeline only and support no performance claim.",
        )
    report = {
        "schema_version": 1,
        "tool": {"name": "vineyard-disease", "version": __version__},
        "manifest": Path(manifest).name,
        "split": split,
        "predictions": Path(predictions_path).name,
        "model": model_info,
        "thresholds": {"min_score": gate.min_score, "min_grape_mass": gate.min_grape_mass}
        if not predictions
        else "as recorded in predictions",
        "status_counts": status_counts([{"status": s} for _, s, _ in triples]),
        "label_counts": dict(Counter(t for t, _, _ in triples)),
        "unmatched_predictions": sum(1 for pid in preds if pid not in manifest_ids),
        "data": {
            "source_kind": source.get("kind"),
            "dataset": source.get("dataset"),
            "revision": source.get("revision"),
            "license": source.get("license"),
            "synthetic_images": synthetic,
        },
        "claims": "none" if synthetic else "in-distribution pipeline check only; not field performance",
        "metrics": metrics,
        "caveats": caveats,
        "disclaimer": DISCLAIMER,
    }
    write_json(out_dir / "metrics.json", report)
    return report


def _sibling_provenance(manifest: str) -> dict | None:
    p = Path(manifest).with_name("provenance.json")
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None
