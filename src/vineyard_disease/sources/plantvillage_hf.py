"""Official PlantVillage (``mohanty/PlantVillage``, config ``color``) source.

The Hugging Face dataset repository ships a loading script, the image
archive ``data.zip``, the official 80/20 split lists ``splits/color_*.txt``
and a leaf grouping file ``leaf_grouping/leaf-map.json``. This module reads
those files directly at a pinned revision instead of executing the loading
script (no remote code), reimplements the script's ``leaf_id`` derivation so
the official grouping is preserved, and fetches only the selected archive
members via HTTP range requests (or from a local copy of ``data.zip``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path

from ..errors import InputError, SourceUnavailableError
from ..labels import PLANTVILLAGE_GRAPE_CLASSES
from . import MAX_IMAGE_BYTES, EntryRejected, SourceEntry
from .http import HttpClient, HttpRangeReader

HF_REPO = "mohanty/PlantVillage"
HF_REVISION = "9e97599868962bd0079b8db4b7f1efa9185fa1e7"
HF_CONFIG = "color"
SOURCE_SPLITS = ("train", "test")
LICENSE = "CC-BY-SA-3.0"
HOMEPAGE = "https://github.com/spMohanty/PlantVillage-Dataset"
DATASET_URL = f"https://huggingface.co/datasets/{HF_REPO}"
CITATION = (
    "Mohanty, S. P., Hughes, D. P., & Salathe, M. (2016). Using deep learning for "
    "image-based plant disease detection. Frontiers in Plant Science, 7. "
    "doi:10.3389/fpls.2016.01419"
)
SPLIT_FILES = {split: f"splits/{HF_CONFIG}_{split}.txt" for split in SOURCE_SPLITS}
LEAF_MAP_FILE = "leaf_grouping/leaf-map.json"
DATA_ZIP_FILE = "data.zip"

# SHA-256 of the small metadata files at HF_REVISION (verified on every run).
PINNED_SHA256: Mapping[str, str] = {
    "splits/color_train.txt": "c43e0205b321d1c929116f5df1842348ac1e5c4861306e081416785d3c439dbd",
    "splits/color_test.txt": "7a7257dcb5f9456feae6ae81248d29c28f45dc4cd41a00c13a5c5c198c5f2f8b",
    "leaf_grouping/leaf-map.json": "3b4b253683c6911744959a3870a92509b1faa1ee34e9f8585e21d0fe6337a25b",
}
# Git LFS object id (= SHA-256) and size of data.zip at HF_REVISION.
DATA_ZIP_SHA256 = "fba30c6a7965e49be94b47a62f8aff6cfb1c35c27f475f22092b56db41745e84"
DATA_ZIP_SIZE = 2184723441

CAVEATS = (
    (
        "PlantVillage images are lab photographs of detached single leaves on plain "
        "backgrounds; they do not represent in-canopy vineyard imagery."
    ),
    (
        "Official train/test splits are preserved (no image changes split); per-class "
        "sampling is bounded and spreads round-robin over leaf groups (see selection)."
    ),
    (
        "Grouping comes from the official leaf-map.json; images without an entry get a "
        "per-image 'filename_fallback' group (grouping unknown, possible near-duplicates). "
        "At the pinned revision all Grape___healthy images are fallback-grouped."
    ),
    (
        "License CC BY-SA 3.0 (dataset card); attribute the PlantVillage authors and "
        "share derived datasets under the same license."
    ),
)


def hf_url(path: str, revision: str = HF_REVISION) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/datasets/{HF_REPO}/resolve/{revision}/{path}"


def derive_leaf_id(file_name: str, class_name: str, leaf_map: Mapping[str, list]) -> tuple[str, str]:
    """Reproduce the dataset script's ``leaf_id`` logic.

    Returns ``(leaf_id, grouping_source)`` where grouping_source is
    ``leaf_map`` for a resolved entry and ``filename_fallback`` otherwise.
    """
    ident = file_name.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG"):
        ident = ident.replace(ext, "")
    ident = ident.strip()
    suggestions = leaf_map.get(ident.lower().strip())
    if suggestions:
        if len(suggestions) == 1:
            return str(suggestions[0]), "leaf_map"
        for suggestion in suggestions:
            if class_name in suggestion:
                return str(suggestion), "leaf_map"
    return f"fallback_{ident}", "filename_fallback"


def parse_split_file(text: str, split: str, leaf_map: Mapping[str, list]) -> tuple[list[SourceEntry], dict]:
    """Parse an official split list into grape entries plus parse stats."""
    entries: list[SourceEntry] = []
    stats = {"lines": 0, "malformed": 0, "non_grape": 0, "wrong_config": 0}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        stats["lines"] += 1
        parts = line.split("/")
        if len(parts) < 4:
            stats["malformed"] += 1
            continue
        if parts[0] != "raw" or parts[1] != HF_CONFIG:
            stats["wrong_config"] += 1
            continue
        class_name, file_name = parts[2], parts[3]
        canonical = PLANTVILLAGE_GRAPE_CLASSES.get(class_name)
        if canonical is None:
            stats["non_grape"] += 1
            continue
        leaf_id, grouping = derive_leaf_id(file_name, class_name, leaf_map)
        entries.append(
            SourceEntry(
                key=line,
                source_label=class_name,
                label=canonical,
                split=split,
                source_split=split,
                group_id=leaf_id,
                grouping_source=grouping,
            )
        )
    return entries, stats


class PlantVillageHF:
    """Pinned PlantVillage color source (metadata + archive members)."""

    kind = "huggingface"
    id_prefix = "pv-color"

    def __init__(
        self,
        client: HttpClient,
        *,
        cache_dir: str | os.PathLike,
        revision: str = HF_REVISION,
        meta_dir: str | os.PathLike | None = None,
        zip_path: str | os.PathLike | None = None,
        verify_zip: bool = True,
        splits: tuple[str, ...] = SOURCE_SPLITS,
        max_member_bytes: int = MAX_IMAGE_BYTES,
    ):
        unknown = set(splits) - set(SOURCE_SPLITS)
        if unknown:
            raise InputError(f"unknown PlantVillage split(s): {sorted(unknown)}; choose from {SOURCE_SPLITS}")
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.revision = revision
        self.meta_dir = Path(meta_dir) if meta_dir else None
        self.zip_path = Path(zip_path) if zip_path else None
        self.verify_zip = verify_zip
        self.splits = tuple(splits)
        self.max_member_bytes = max_member_bytes
        self.files: dict[str, dict] = {}
        self.parse_stats: dict[str, dict] = {}
        self.archive_info: dict = {}
        self.warnings: list[str] = []
        if revision != HF_REVISION:
            self.warnings.append(
                f"revision {revision} differs from the pinned {HF_REVISION}; metadata hash pins are not enforced"
            )

    # -- metadata -----------------------------------------------------
    def _metadata(self, rel: str) -> bytes:
        expected = PINNED_SHA256.get(rel) if self.revision == HF_REVISION else None
        if self.meta_dir is not None:
            candidates = [self.meta_dir / rel, self.meta_dir / Path(rel).name]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise SourceUnavailableError(f"{rel} not found under --hf-meta-dir {self.meta_dir}")
            data, origin = path.read_bytes(), "local"
        else:
            cached = self.cache_dir / self.revision / rel
            if cached.is_file() and (expected is None or hashlib.sha256(cached.read_bytes()).hexdigest() == expected):
                data, origin = cached.read_bytes(), "cache"
            else:
                data, origin = self.client.get_bytes(hf_url(rel, self.revision)), "download"
                cached.parent.mkdir(parents=True, exist_ok=True)
                tmp = cached.with_suffix(cached.suffix + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(cached)
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise SourceUnavailableError(
                f"{rel}: sha256 {digest} does not match pinned {expected}; refusing to continue"
            )
        self.files[rel] = {"sha256": digest, "bytes": len(data), "origin": origin, "url": hf_url(rel, self.revision)}
        return data

    def load_entries(self) -> list[SourceEntry]:
        try:
            leaf_map = json.loads(self._metadata(LEAF_MAP_FILE).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceUnavailableError(f"{LEAF_MAP_FILE} is not valid JSON: {exc}") from exc
        if not isinstance(leaf_map, dict):
            raise SourceUnavailableError(f"{LEAF_MAP_FILE}: expected a JSON object")
        entries: list[SourceEntry] = []
        for split in self.splits:
            text = self._metadata(SPLIT_FILES[split]).decode("utf-8")
            split_entries, stats = parse_split_file(text, split, leaf_map)
            self.parse_stats[split] = stats
            entries.extend(split_entries)
        if not entries:
            raise SourceUnavailableError("no grape entries found in the official split files")
        return entries

    # -- images ---------------------------------------------------------
    @contextlib.contextmanager
    def open_reader(self) -> Iterator:
        """Yield ``read(entry) -> bytes`` backed by the archive."""
        if self.zip_path is not None:
            if not self.zip_path.is_file():
                raise SourceUnavailableError(f"--hf-zip {self.zip_path} does not exist")
            size = self.zip_path.stat().st_size
            info = {"origin": "local", "bytes": size}
            if self.revision == HF_REVISION and size != DATA_ZIP_SIZE:
                raise SourceUnavailableError(
                    f"{self.zip_path}: size {size} != pinned {DATA_ZIP_SIZE}; not the pinned data.zip"
                )
            if self.verify_zip:
                h = hashlib.sha256()
                with self.zip_path.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                        h.update(chunk)
                info["sha256"] = h.hexdigest()
                if self.revision == HF_REVISION and info["sha256"] != DATA_ZIP_SHA256:
                    raise SourceUnavailableError(f"{self.zip_path}: sha256 does not match pinned data.zip")
            fh = self.zip_path.open("rb")
        else:
            reader = HttpRangeReader(hf_url(DATA_ZIP_FILE, self.revision), self.client)
            etag = (reader.headers.get("x-linked-etag") or reader.headers.get("etag") or "").strip('"')
            info = {
                "origin": "http-range",
                "bytes": reader.size,
                "etag": etag,
                "url": hf_url(DATA_ZIP_FILE, self.revision),
                "pinned_lfs_sha256": DATA_ZIP_SHA256,
                "note": "whole-archive hash not verified for range reads; size, revision and per-image sha256 recorded",
            }
            if self.revision == HF_REVISION:
                if reader.size != DATA_ZIP_SIZE:
                    raise SourceUnavailableError(f"remote data.zip size {reader.size} != pinned {DATA_ZIP_SIZE}")
                if len(etag) == 64 and etag != DATA_ZIP_SHA256:
                    raise SourceUnavailableError("remote data.zip LFS oid does not match the pinned sha256")
            fh = reader
        self.archive_info = info
        try:
            try:
                zf = zipfile.ZipFile(fh)
            except zipfile.BadZipFile as exc:
                raise SourceUnavailableError(f"data.zip could not be opened: {exc}") from exc
            names = set(zf.namelist())
            prefix = _detect_prefix(names)
            info["member_prefix"] = prefix

            def read(entry: SourceEntry) -> bytes:
                name = prefix + entry.key
                if name not in names:
                    raise EntryRejected("missing_in_source", f"{entry.key} not in data.zip")
                info = zf.getinfo(name)
                # Bound before reading: zipfile stops decompressing at the declared size.
                if info.file_size > self.max_member_bytes or info.is_dir():
                    raise EntryRejected("member_too_large", f"{info.file_size} bytes > {self.max_member_bytes}")
                return zf.read(info)

            yield read
            if isinstance(fh, HttpRangeReader):
                info["range_requests"] = fh.requests
        finally:
            fh.close()

    def source_fields(self, entry: SourceEntry) -> dict:
        return {"kind": self.kind, "repo": HF_REPO, "revision": self.revision, "config": HF_CONFIG, "path": entry.key}

    def provenance(self) -> dict:
        return {
            "kind": self.kind,
            "dataset": HF_REPO,
            "dataset_url": DATASET_URL,
            "homepage": HOMEPAGE,
            "revision": self.revision,
            "pinned_revision": HF_REVISION,
            "config": HF_CONFIG,
            "splits": list(self.splits),
            "license": LICENSE,
            "citation": CITATION,
            "files": self.files,
            "archive": {k: v for k, v in self.archive_info.items()},
            "parse_stats": self.parse_stats,
            "loader": "split lists + leaf-map read directly; dataset loading script not executed",
            "caveats": list(CAVEATS),
            "warnings": list(self.warnings),
        }


def _detect_prefix(names: set[str]) -> str:
    """Archive members may sit under a top-level folder; find the prefix of ``raw/``."""
    for name in names:
        if name.startswith("raw/"):
            return ""
    for name in sorted(names):
        idx = name.find("/raw/")
        if idx >= 0:
            return name[: idx + 1]
    raise SourceUnavailableError("data.zip does not contain the expected raw/<config>/ layout")
