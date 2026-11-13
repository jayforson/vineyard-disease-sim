"""Local class-directory source.

Accepted layouts (directory names are normalized with
:func:`vineyard_disease.labels.normalize_label`)::

    <input>/<class>/<image>              # no source split -> "unspecified"
    <input>/<split>/<class>/<image>      # split in {train, test, val, validation, valid}

Only files directly inside a class directory are read (no deeper recursion).
Unrecognized class directories (e.g. other crops) are skipped and reported,
or rejected with ``strict=True``. Optional grouping comes from a CSV file with
columns ``path,group_id`` (``path`` relative to the input directory).
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

from ..errors import InputError, SourceUnavailableError
from ..labels import normalize_label
from . import MAX_IMAGE_BYTES, EntryRejected, SourceEntry

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
SPLIT_DIRS = {"train": "train", "test": "test", "val": "val", "validation": "val", "valid": "val"}


def _visible_dirs(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))


def hash_split(key: str, seed: int, test_fraction: float) -> str:
    """Deterministic split assignment from a stable hash of ``key``."""
    digest = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "test" if value < test_fraction else "train"


class LocalClassDirs:
    kind = "local"
    id_prefix = "local"

    def __init__(
        self,
        input_dir: str | os.PathLike,
        *,
        groups_file: str | os.PathLike | None = None,
        strict: bool = False,
        assign_split: str = "none",
        test_fraction: float = 0.2,
        seed: int = 0,
        splits: tuple[str, ...] | None = None,
    ):
        self.input_dir = Path(input_dir)
        if not self.input_dir.is_dir():
            raise SourceUnavailableError(f"--input-dir {self.input_dir} is not a directory")
        if assign_split not in ("none", "hash"):
            raise InputError("--assign-split must be 'none' or 'hash'")
        if not 0.0 < test_fraction < 1.0:
            raise InputError("--test-fraction must be in (0, 1)")
        self.groups_file = Path(groups_file) if groups_file else None
        self.strict = strict
        self.assign_split = assign_split
        self.test_fraction = test_fraction
        self.seed = seed
        self.splits = tuple(splits) if splits else None
        self.skipped_dirs: list[str] = []
        self.skipped_files = 0
        self.caveats: list[str] = []

    def _load_groups(self) -> dict[str, str]:
        if self.groups_file is None:
            return {}
        if not self.groups_file.is_file():
            raise SourceUnavailableError(f"--groups-file {self.groups_file} not found")
        groups: dict[str, str] = {}
        with self.groups_file.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or not {"path", "group_id"} <= set(reader.fieldnames):
                raise InputError("--groups-file must be a CSV with columns path,group_id")
            for row in reader:
                path = (row.get("path") or "").strip().replace("\\", "/")
                gid = (row.get("group_id") or "").strip()
                if path and gid:
                    groups[path] = gid
        return groups

    def _class_dirs(self) -> list[tuple[str, Path]]:
        """Return [(source_split, class_dir)]."""
        top = _visible_dirs(self.input_dir)
        split_dirs = [d for d in top if d.name.lower() in SPLIT_DIRS]
        if split_dirs and len(split_dirs) != len(top):
            others = sorted(d.name for d in top if d not in split_dirs)
            raise InputError(f"mixed split and class directories in {self.input_dir}: {others}")
        if split_dirs:
            return [(SPLIT_DIRS[s.name.lower()], c) for s in split_dirs for c in _visible_dirs(s)]
        return [("unspecified", c) for c in top]

    def load_entries(self) -> list[SourceEntry]:
        groups = self._load_groups()
        entries: list[SourceEntry] = []
        for source_split, class_dir in self._class_dirs():
            canonical = normalize_label(class_dir.name)
            rel_dir = class_dir.relative_to(self.input_dir).as_posix()
            if canonical is None:
                if self.strict:
                    raise InputError(f"unrecognized class directory '{rel_dir}' (use canonical grape class names)")
                self.skipped_dirs.append(rel_dir)
                continue
            for f in sorted(class_dir.iterdir()):
                if not f.is_file() or f.name.startswith(".") or f.suffix.lower() not in IMAGE_SUFFIXES:
                    self.skipped_files += 1
                    continue
                rel = f.relative_to(self.input_dir).as_posix()
                gid = groups.get(rel)
                entries.append(
                    SourceEntry(
                        key=rel,
                        source_label=class_dir.name,
                        label=canonical,
                        split=source_split,
                        source_split=source_split,
                        group_id=gid,
                        grouping_source="groups_file" if gid else "none",
                    )
                )
        entries = self._apply_split_policy(entries)
        if self.splits:
            entries = [e for e in entries if e.split in self.splits]
        if not entries:
            raise SourceUnavailableError(f"no recognizable grape images under {self.input_dir}")
        if any(e.grouping_source == "none" for e in entries):
            self.caveats.append(
                "Some images have no grouping information (group_id=null); images of the same "
                "physical leaf may appear in different splits."
            )
        return entries

    def _apply_split_policy(self, entries: list[SourceEntry]) -> list[SourceEntry]:
        has_source_split = any(e.source_split != "unspecified" for e in entries)
        if self.assign_split == "none":
            if not has_source_split:
                self.caveats.append("No source split supplied; all records have split='unspecified'.")
            return entries
        if has_source_split:
            raise InputError("--assign-split hash is only allowed when the input has no split directories")
        out = []
        for e in entries:
            key = e.group_id if e.group_id else e.key
            out.append(
                SourceEntry(
                    e.key,
                    e.source_label,
                    e.label,
                    hash_split(key, self.seed, self.test_fraction),
                    e.source_split,
                    e.group_id,
                    e.grouping_source,
                )
            )
        self.caveats.append(
            f"Split assigned by this pipeline (hash, seed={self.seed}, test_fraction={self.test_fraction}); "
            "group-aware only for images with a group_id."
        )
        return out

    @contextlib.contextmanager
    def open_reader(self) -> Iterator:
        def read(entry: SourceEntry) -> bytes:
            path = self.input_dir / entry.key
            size = path.stat().st_size
            if size > MAX_IMAGE_BYTES:
                raise EntryRejected("file_too_large", f"{size} bytes > {MAX_IMAGE_BYTES}")
            return path.read_bytes()

        yield read

    def source_fields(self, entry: SourceEntry) -> dict:
        return {"kind": self.kind, "path": entry.key}

    def provenance(self) -> dict:
        return {
            "kind": self.kind,
            "input_dir_name": self.input_dir.name,  # no absolute paths in provenance
            "groups_file": self.groups_file.name if self.groups_file else None,
            "assign_split": self.assign_split,
            "test_fraction": self.test_fraction if self.assign_split == "hash" else None,
            "skipped_class_dirs": self.skipped_dirs,
            "skipped_files": self.skipped_files,
            "license": "unknown (local data; record the license of your images)",
            "caveats": self.caveats,
            "warnings": [],
        }
