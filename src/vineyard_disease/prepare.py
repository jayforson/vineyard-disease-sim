"""``prepare``: bounded, reproducible manifest creation from a data source."""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from . import __version__
from .errors import InputError
from .labels import CANONICAL_CLASSES
from .manifest import SCHEMA_VERSION, ensure_ignored_data_root, write_json, write_manifest
from .quality import QualityConfig, flag_duplicates, inspect_image_bytes
from .sources import EntryRejected, SourceEntry

MANIFEST_NAME = "manifest.jsonl"
PROVENANCE_NAME = "provenance.json"
REAL_GROUPING = ("leaf_map", "groups_file")


def _rank(seed: int, *parts: str) -> str:
    return hashlib.sha256("|".join((str(seed), *parts)).encode()).hexdigest()


def select_entries(entries: Iterable[SourceEntry], max_per_class: int | None, seed: int) -> list[SourceEntry]:
    """Deterministically select at most ``max_per_class`` images per (split, class).

    Sampling never moves an image between splits (source splits are kept, so
    group separation across splits is preserved). Within a (split, class)
    bucket, groups (physical leaves) are ranked by a seeded SHA-256 and taken
    round-robin: one image of each ranked group first, then second images,
    and so on. This maximizes distinct leaves under the bound and does not
    depend on Python's RNG or input order. ``max_per_class=None`` keeps all.
    """
    if max_per_class is not None and max_per_class < 1:
        raise InputError("--max-per-class must be >= 1")
    buckets: dict[tuple[str, str], dict[str, list[SourceEntry]]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        gkey = e.group_id if e.group_id is not None else f"__image__:{e.key}"
        buckets[(e.split, e.label)][gkey].append(e)
    selected: list[SourceEntry] = []
    for (split, label), groups in sorted(buckets.items()):
        order = sorted(groups, key=lambda g: (_rank(seed, split, label, g), g))
        members = {g: sorted(groups[g], key=lambda e: (_rank(seed, e.key), e.key)) for g in order}
        limit = max_per_class if max_per_class is not None else sum(len(m) for m in members.values())
        picked: list[SourceEntry] = []
        depth = 0
        while len(picked) < limit:
            layer = [members[g][depth] for g in order if depth < len(members[g])]
            if not layer:
                break
            picked.extend(layer[: limit - len(picked)])
            depth += 1
        selected.extend(picked)
    return sorted(selected, key=lambda e: (e.split, e.label, e.key))


def group_leakage(records: list[dict]) -> list[str]:
    """Group ids with real grouping info that occur in more than one split."""
    splits: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r.get("group_id") and r.get("grouping_source") in REAL_GROUPING:
            splits[r["group_id"]].add(r["split"])
    return sorted(g for g, s in splits.items() if len(s) > 1)


def _safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._") or "image"
    return stem[-80:]


def _record_id(prefix: str, key: str) -> str:
    return f"{prefix}-{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def run_prepare(
    source,
    output_dir: str | Path,
    *,
    max_per_class: int | None = 25,
    seed: int = 0,
    quality: QualityConfig | None = None,
    overwrite: bool = False,
    command: str | None = None,
) -> dict:
    """Prepare a manifest from ``source``; returns a summary dict.

    Any source failure raises before a manifest is written; nothing is
    synthesized to stand in for missing data.
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        raise InputError(f"{manifest_path} exists; pass --overwrite to replace it")
    ensure_ignored_data_root(output_dir)
    quality = quality or QualityConfig()

    entries = source.load_entries()
    available = Counter((e.split, e.label) for e in entries)
    selected = select_entries(entries, max_per_class, seed)

    images_dir = output_dir / "images"
    staging = output_dir / "images.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    records: list[dict] = []
    rejected: list[dict] = []
    with source.open_reader() as read:
        for e in selected:
            try:
                data = read(e)
            except EntryRejected as exc:
                rejected.append({"key": e.key, "reasons": [exc.reason], "detail": exc.detail})
                continue
            except OSError as exc:
                rejected.append({"key": e.key, "reasons": ["unreadable"], "detail": f"{type(exc).__name__}"})
                continue
            report = inspect_image_bytes(data, quality)
            if not report.ok:
                rejected.append({"key": e.key, "reasons": sorted(report.issues), "detail": report.error})
                continue
            rel = Path(e.split) / e.label / f"{report.sha256[:16]}_{_safe_name(e.key)}"
            (staging / rel).parent.mkdir(parents=True, exist_ok=True)
            (staging / rel).write_bytes(data)
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "id": _record_id(source.id_prefix, e.key),
                    "image": (Path("images") / rel).as_posix(),
                    "label": e.label,
                    "source_label": e.source_label,
                    "split": e.split,
                    "source_split": e.source_split,
                    "group_id": e.group_id,
                    "grouping_source": e.grouping_source,
                    "source": source.source_fields(e),
                    "sha256": report.sha256,
                    "width": report.width,
                    "height": report.height,
                    "quality": report.as_dict(),
                    "synthetic": False,
                }
            )

    if not records:
        shutil.rmtree(staging)
        reasons = Counter(reason for r in rejected for reason in r["reasons"])
        raise InputError(
            f"no images survived preparation ({len(selected)} selected, rejected: {dict(reasons)}); "
            "no manifest was written"
        )
    if images_dir.exists():
        shutil.rmtree(images_dir)
    staging.replace(images_dir)

    dup_summary = flag_duplicates(records)
    leakage = group_leakage(records)
    counts = Counter((r["split"], r["label"]) for r in records)
    splits = sorted({e.split for e in entries})
    warnings = list(source.provenance().get("warnings", []))
    for split in splits:
        for cls in CANONICAL_CLASSES:
            if counts[(split, cls)] == 0:
                warnings.append(f"split '{split}' has no '{cls}' images: per-class metrics will be undefined")
    if leakage:
        warnings.append(f"{len(leakage)} group(s) occur in more than one split (possible leakage)")
    if dup_summary["duplicate_sets_across_splits"]:
        warnings.append("identical image bytes occur in more than one split (leakage)")

    n = write_manifest(manifest_path, records)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "vineyard-disease", "version": __version__, "command": command},
        "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "manifest": MANIFEST_NAME,
        "records": n,
        "source": source.provenance(),
        "selection": {
            "max_per_class_per_split": max_per_class,
            "seed": seed,
            "method": "round-robin over leaf groups ranked by sha256(seed|split|class|group_id); source splits kept",
        },
        "quality_config": quality.__dict__,
        "available": _nested(available),
        "selected": _nested(counts),
        "rejected": rejected,
        "duplicates": dup_summary,
        "group_leakage": leakage[:100],
        "grouping_sources": dict(Counter(r["grouping_source"] for r in records)),
        "warnings": warnings,
        "synthetic": False,
    }
    write_json(output_dir / PROVENANCE_NAME, provenance)
    return {
        "manifest": str(manifest_path),
        "provenance": str(output_dir / PROVENANCE_NAME),
        "records": n,
        "rejected": len(rejected),
        "selected": provenance["selected"],
        "warnings": warnings,
    }


def _nested(counter: Counter) -> dict:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for (split, label), n in sorted(counter.items()):
        out[split][label] = n
    return dict(out)
