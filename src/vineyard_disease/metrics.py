"""Classification metrics for the four grape classes with abstention.

Only ``accepted`` rows carry a predicted class. Abstentions (``non_grape``,
``low_score``, errors, missing predictions) count against recall but not
against precision, and are reported separately via coverage.
Undefined ratios are ``None`` rather than a misleading 0.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .labels import CANONICAL_CLASSES

ABSTAIN_COLUMNS = ("non_grape", "low_score", "input_error", "unsupported_model", "missing")


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def compute_metrics(rows: Iterable[tuple[str, str, str | None]], classes: Sequence[str] = CANONICAL_CLASSES) -> dict:
    """``rows`` are ``(true_label, status, predicted_class)`` triples."""
    rows = list(rows)
    columns = list(classes) + list(ABSTAIN_COLUMNS)
    confusion = {c: {col: 0 for col in columns} for c in classes}
    for true, status, pred in rows:
        if true not in confusion:
            raise ValueError(f"unknown true label '{true}'")
        if status == "accepted":
            if pred not in classes:
                raise ValueError(f"accepted row with invalid predicted class '{pred}'")
            confusion[true][pred] += 1
        else:
            col = status if status in ABSTAIN_COLUMNS else "missing"
            confusion[true][col] += 1

    per_class = {}
    f1s = []
    for c in classes:
        tp = confusion[c][c]
        support = sum(confusion[c].values())
        predicted = sum(confusion[t][c] for t in classes)
        accepted = sum(confusion[c][k] for k in classes)
        precision = _ratio(tp, predicted)
        recall = _ratio(tp, support)
        if precision is None or recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if f1 is not None:
            f1s.append(f1)
        per_class[c] = {
            "support": support,
            "predicted": predicted,
            "tp": tp,
            "fp": predicted - tp,
            "fn": support - tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "coverage": _ratio(accepted, support),
        }
    n = len(rows)
    accepted_total = sum(1 for _, s, _ in rows if s == "accepted")
    correct = sum(1 for t, s, p in rows if s == "accepted" and p == t)
    return {
        "n": n,
        "accepted": accepted_total,
        "coverage": _ratio(accepted_total, n),
        "accuracy": _ratio(correct, n),
        "selective_accuracy": _ratio(correct, accepted_total),
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "macro_f1_classes": [c for c in classes if per_class[c]["f1"] is not None],
        "per_class": per_class,
        "confusion": confusion,
        "definitions": {
            "accuracy": "correct accepted predictions / all evaluated images (abstentions count as wrong)",
            "selective_accuracy": "correct / accepted",
            "coverage": "accepted / all evaluated images",
            "recall": "tp / support (abstentions count as misses)",
            "precision": "tp / accepted predictions of that class",
        },
    }
