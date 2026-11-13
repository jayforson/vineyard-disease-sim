"""Manifest JSONL schema, reading and writing.

One JSON object per line, sorted by ``id``. Image paths are relative to the
manifest's directory. Example record::

    {"schema_version": 1, "id": "pv-color-1a2b...", "image": "images/test/esca/...jpg",
     "label": "esca", "source_label": "Grape___Esca_(Black_Measles)",
     "split": "test", "source_split": "test",
     "group_id": "Grape___Esca_(Black_Measles):::111.0", "grouping_source": "leaf_map",
     "source": {"kind": "huggingface", ...}, "sha256": "...", "width": 256,
     "height": 256, "quality": {"ok": true, "issues": []}, "synthetic": false}
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from .errors import InputError
from .labels import CANONICAL_CLASSES

SCHEMA_VERSION = 1
REQUIRED_FIELDS = ("id", "image", "label", "split")
GROUPING_SOURCES = (
    "leaf_map",  # official PlantVillage leaf grouping file
    "filename_fallback",  # no grouping entry; one group per image identifier
    "groups_file",  # user-supplied grouping for local data
    "none",  # no grouping information
)


def validate_record(rec: dict, *, line: int | None = None) -> None:
    where = f" (line {line})" if line is not None else ""
    for key in REQUIRED_FIELDS:
        if key not in rec:
            raise InputError(f"manifest record missing '{key}'{where}")
    if rec["label"] not in CANONICAL_CLASSES:
        raise InputError(f"manifest label '{rec['label']}' is not canonical{where}")
    gs = rec.get("grouping_source", "none")
    if gs not in GROUPING_SOURCES:
        raise InputError(f"unknown grouping_source '{gs}'{where}")


def write_manifest(path: str | os.PathLike, records: Iterable[dict]) -> int:
    """Atomically write records sorted by id; returns the record count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(records, key=lambda r: r["id"])
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise InputError("duplicate record ids in manifest")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in rows:
            validate_record(rec)
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return len(rows)


def read_jsonl(path: str | os.PathLike) -> Iterator[tuple[int, dict]]:
    path = Path(path)
    if not path.is_file():
        raise InputError(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise InputError(f"{path}:{lineno}: expected a JSON object")
            yield lineno, obj


def read_manifest(path: str | os.PathLike, *, split: str | None = None) -> list[dict]:
    """Read and validate a manifest, optionally filtering by ``split``.

    Each returned record gets ``_image_path``: the absolute image path.
    """
    path = Path(path)
    base = path.resolve().parent
    out = []
    for lineno, rec in read_jsonl(path):
        validate_record(rec, line=lineno)
        if split is not None and rec["split"] != split:
            continue
        rec["_image_path"] = str((base / rec["image"]).resolve())
        out.append(rec)
    return out


def write_json(path: str | os.PathLike, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def ensure_ignored_data_root(output_dir: str | os.PathLike) -> None:
    """Make sure a ``data/`` ancestor of ``output_dir`` is git-ignored.

    Writes ``data/.gitignore`` containing ``*`` when the output lives under a
    directory named ``data`` and no ignore file exists there yet.
    """
    p = Path(output_dir).resolve()
    for candidate in (p, *p.parents):
        if candidate.name == "data":
            gi = candidate / ".gitignore"
            if not gi.exists():
                candidate.mkdir(parents=True, exist_ok=True)
                gi.write_text("# Generated datasets, caches and logs are never committed.\n*\n")
            return
