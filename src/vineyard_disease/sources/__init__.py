"""Data sources that yield grape-leaf entries for :mod:`vineyard_disease.prepare`."""

from __future__ import annotations

from dataclasses import dataclass

# Upper bound on the uncompressed size of one source image (PlantVillage
# images are ~20-40 KB); larger members are rejected before being read.
MAX_IMAGE_BYTES = 20 * 1024 * 1024


class EntryRejected(Exception):
    """A single source entry cannot be used; ``reason`` is recorded in provenance."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class SourceEntry:
    """One candidate image before download/quality checks."""

    key: str  # stable source-relative identifier (used for the record id)
    source_label: str  # label as named by the source
    label: str  # canonical class
    split: str  # split used by this pipeline
    source_split: str  # split as supplied by the source ("unspecified" if none)
    group_id: str | None  # grouping key (physical leaf) when supplied
    grouping_source: str  # see manifest.GROUPING_SOURCES
