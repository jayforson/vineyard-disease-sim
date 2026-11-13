"""PlantVillage HF source, local class directories, selection and prepare."""

import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import pytest

pytest.importorskip("PIL")

from test_pipeline_support import REV, FakeHub, build_fake_dataset, client_for, png_bytes

from vineyard_disease.errors import InputError, PipelineError, SourceUnavailableError
from vineyard_disease.manifest import read_manifest
from vineyard_disease.prepare import group_leakage, run_prepare, select_entries
from vineyard_disease.sources import SourceEntry
from vineyard_disease.sources.http import ByteBudget, HttpRangeReader
from vineyard_disease.sources.local_dir import LocalClassDirs, hash_split
from vineyard_disease.sources.plantvillage_hf import (
    HF_REVISION,
    PINNED_SHA256,
    PlantVillageHF,
    derive_leaf_id,
    hf_url,
    parse_split_file,
)


def hf_source(tmp_path, hub, **kw):
    kw.setdefault("revision", REV)
    return PlantVillageHF(client_for(hub, budget=kw.pop("budget", None)), cache_dir=tmp_path / "cache", **kw)


# -- leaf grouping (mirrors the dataset script) ---------------------------------
def test_derive_leaf_id_variants():
    lm = {
        "fam_b.rot 1": ["Grape___Black_rot:::1.0"],
        "fam_b.rot 2": ["Apple___Black_rot:::7.0", "Grape___Black_rot:::2.0"],
        "fam_b.rot 3": ["Apple___Black_rot:::7.0", "Peach___healthy:::8.0"],
    }
    assert derive_leaf_id("u___FAM_B.Rot 1.JPG", "Grape___Black_rot", lm) == ("Grape___Black_rot:::1.0", "leaf_map")
    assert derive_leaf_id("u___FAM_B.Rot 1 copy.jpg", "Grape___Black_rot", lm)[1] == "leaf_map"
    assert derive_leaf_id("u___FAM_B.Rot 1_final_masked.png", "Grape___Black_rot", lm)[0] == "Grape___Black_rot:::1.0"
    assert derive_leaf_id("u___FAM_B.Rot 2.JPG", "Grape___Black_rot", lm) == ("Grape___Black_rot:::2.0", "leaf_map")
    assert derive_leaf_id("u___FAM_B.Rot 3.JPG", "Grape___Black_rot", lm) == (
        "fallback_FAM_B.Rot 3",
        "filename_fallback",
    )
    assert derive_leaf_id("u___Unknown 9.JPG", "Grape___healthy", lm) == ("fallback_Unknown 9", "filename_fallback")


def test_parse_split_file_filters_non_grape_and_malformed():
    text = "raw/color/Grape___Black_rot/a___FAM_B.Rot 1.JPG\nraw/color/Apple___healthy/b.JPG\nraw/grayscale/Grape___Black_rot/c.JPG\ngarbage\n"
    entries, stats = parse_split_file(text, "test", {})
    assert [e.label for e in entries] == ["black_rot"]
    assert entries[0].source_split == "test" and entries[0].source_label == "Grape___Black_rot"
    assert stats == {"lines": 4, "malformed": 1, "non_grape": 1, "wrong_config": 1}


# -- HTTP range reader ------------------------------------------------------------
def test_range_reader_reads_zip_members_without_full_download(tmp_path):
    files, _ = build_fake_dataset(extra_members={"raw/color/pad.bin": os.urandom(600_000)})
    hub = FakeHub(files)
    client = client_for(hub)
    reader = HttpRangeReader(hf_url("data.zip", REV), client, block_size=4096)
    with zipfile.ZipFile(reader) as zf:
        name = "raw/color/Grape___Black_rot/a1___FAM_B.Rot 0001.JPG"
        assert zf.read(name)[:8] == b"\x89PNG\r\n\x1a\n"
    assert all(r is not None for _, r in hub.log)  # every request was a Range request
    assert client.budget.used < len(files["data.zip"]) / 2


def test_range_reader_reresolves_expired_signed_url():
    files, _ = build_fake_dataset()
    hub = FakeHub(files, expire_once=True)
    reader = HttpRangeReader(hf_url("data.zip", REV), client_for(hub), block_size=1024)
    reader.seek(-10, 2)
    assert len(reader.read(10)) == 10
    assert sum(1 for url, _ in hub.log if "resolve" in url) == 2


def test_range_reader_requires_range_support():
    files, _ = build_fake_dataset()
    with pytest.raises(SourceUnavailableError, match="Range"):
        HttpRangeReader(hf_url("data.zip", REV), client_for(FakeHub(files, no_range=True)))


def test_no_credentials_are_sent_even_with_ambient_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    files, _ = build_fake_dataset()
    hub = FakeHub(files)
    reader = HttpRangeReader(hf_url("data.zip", REV), client_for(hub), block_size=1024)
    reader.read(16)
    client_for(hub).get_bytes(hf_url("leaf_grouping/leaf-map.json", REV))
    assert hub.auth_seen == []


def test_byte_budget_aborts():
    budget = ByteBudget(10)
    budget.charge(10)
    with pytest.raises(PipelineError, match="budget"):
        budget.charge(1)


# -- HF source end to end ----------------------------------------------------------
def test_hf_prepare_preserves_splits_grouping_and_provenance(tmp_path):
    files, _ = build_fake_dataset()
    hub = FakeHub(files)
    out = tmp_path / "data" / "pv"
    summary = run_prepare(hf_source(tmp_path, hub), out, max_per_class=10, seed=0)
    recs = read_manifest(out / "manifest.jsonl")
    assert summary["records"] == len(recs) == 13
    assert {r["label"] for r in recs} == {"black_rot", "esca", "isariopsis_leaf_spot", "healthy"}
    assert all(r["split"] == r["source_split"] for r in recs)
    by_path = {r["source"]["path"].split("/")[-1]: r for r in recs}
    assert by_path["a1___FAM_B.Rot 0001.JPG"]["group_id"] == "Grape___Black_rot:::1.0"
    assert by_path["a4___FAM_B.Rot 0004.JPG"]["group_id"] == "Grape___Black_rot:::3.0"
    assert by_path["h1___Mt.N.V_HL 0001.JPG"]["grouping_source"] == "filename_fallback"
    assert by_path["l2___FAM_L.Blight 0101.JPG"]["grouping_source"] == "filename_fallback"
    assert by_path["e2___FAM_B.Msls 0101.JPG"]["split"] == "test"
    for r in recs:
        assert os.path.isfile(r["_image_path"])
        assert hashlib.sha256(Path(r["_image_path"]).read_bytes()).hexdigest() == r["sha256"]
        assert r["source"]["repo"] == "mohanty/PlantVillage" and r["source"]["config"] == "color"
        assert r["synthetic"] is False and r["quality"]["ok"]
        assert not os.path.isabs(r["image"])
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["source"]["license"] == "CC-BY-SA-3.0"
    assert prov["source"]["revision"] == REV
    assert prov["source"]["parse_stats"]["train"]["non_grape"] == 1
    assert any("differs from the pinned" in w for w in prov["warnings"])
    assert any("fallback" in c for c in prov["source"]["caveats"])
    assert prov["source"]["archive"]["origin"] == "http-range"
    assert (tmp_path / "data" / ".gitignore").read_text().strip().endswith("*")
    assert str(tmp_path) not in json.dumps(prov)


def test_hf_prepare_bounded_and_deterministic(tmp_path):
    files, _ = build_fake_dataset()
    a = run_prepare(hf_source(tmp_path, FakeHub(files)), tmp_path / "a", max_per_class=1, seed=3)
    b = run_prepare(hf_source(tmp_path, FakeHub(files)), tmp_path / "b", max_per_class=1, seed=3)
    assert a["selected"] == b["selected"]
    assert all(n == 1 for split in a["selected"].values() for n in split.values())
    ma = [(r["id"], r["sha256"]) for r in read_manifest(tmp_path / "a" / "manifest.jsonl")]
    mb = [(r["id"], r["sha256"]) for r in read_manifest(tmp_path / "b" / "manifest.jsonl")]
    assert ma == mb


def test_hf_split_filter(tmp_path):
    files, _ = build_fake_dataset()
    run_prepare(hf_source(tmp_path, FakeHub(files), splits=("test",)), tmp_path / "t", max_per_class=5)
    assert {r["split"] for r in read_manifest(tmp_path / "t" / "manifest.jsonl")} == {"test"}
    with pytest.raises(InputError):
        hf_source(tmp_path, FakeHub(files), splits=("validation",))


def test_hf_network_failure_writes_nothing_and_synthesizes_nothing(tmp_path):
    files, _ = build_fake_dataset()
    hub = FakeHub(files, fail={"splits/color_train.txt"})
    out = tmp_path / "data" / "pv"
    with pytest.raises(SourceUnavailableError, match="No data was synthesized"):
        run_prepare(hf_source(tmp_path, hub), out, max_per_class=2)
    assert not (out / "manifest.jsonl").exists()


def test_hf_missing_archive_is_an_error(tmp_path):
    files, _ = build_fake_dataset()
    del files["data.zip"]
    with pytest.raises(SourceUnavailableError):
        run_prepare(hf_source(tmp_path, FakeHub(files)), tmp_path / "pv", max_per_class=2)
    assert not (tmp_path / "pv" / "manifest.jsonl").exists()


def test_pinned_revision_enforces_metadata_hashes(tmp_path):
    _files, _ = build_fake_dataset()

    def transport(url, headers, timeout):
        from vineyard_disease.sources.http import Response

        return Response(200, {}, b"tampered", url)

    src = PlantVillageHF(client_for(transport), cache_dir=tmp_path / "cache", revision=HF_REVISION)
    with pytest.raises(SourceUnavailableError, match="does not match pinned"):
        src.load_entries()
    assert set(PINNED_SHA256) == {"splits/color_train.txt", "splits/color_test.txt", "leaf_grouping/leaf-map.json"}


def test_metadata_cache_is_reused(tmp_path):
    files, _ = build_fake_dataset()
    hub = FakeHub(files)
    hf_source(tmp_path, hub).load_entries()
    hub2 = FakeHub(files)
    src = hf_source(tmp_path, hub2)
    src.load_entries()
    assert hub2.log == []
    assert {f["origin"] for f in src.files.values()} == {"cache"}


def test_offline_meta_dir_and_local_zip(tmp_path):
    files, _ = build_fake_dataset(prefix="PlantVillage/")
    meta = tmp_path / "meta"
    for rel in ("splits/color_train.txt", "splits/color_test.txt", "leaf_grouping/leaf-map.json"):
        (meta / rel).parent.mkdir(parents=True, exist_ok=True)
        (meta / rel).write_bytes(files[rel])
    zpath = tmp_path / "data.zip"
    zpath.write_bytes(files["data.zip"])

    def offline(url, headers, timeout):  # any network use fails the test
        raise AssertionError(f"unexpected network access {url}")

    src = PlantVillageHF(client_for(offline), cache_dir=tmp_path / "cache", revision=REV, meta_dir=meta, zip_path=zpath)
    summary = run_prepare(src, tmp_path / "pv", max_per_class=2)
    assert summary["records"] > 0
    prov = json.loads((tmp_path / "pv" / "provenance.json").read_text())
    assert prov["source"]["archive"]["member_prefix"] == "PlantVillage/"
    assert prov["source"]["archive"]["sha256"] == hashlib.sha256(files["data.zip"]).hexdigest()


def test_local_zip_must_match_pinned_size(tmp_path):
    zpath = tmp_path / "data.zip"
    zpath.write_bytes(b"PK")
    src = PlantVillageHF(client_for(FakeHub({})), cache_dir=tmp_path, zip_path=zpath)
    with pytest.raises(SourceUnavailableError, match="pinned"), src.open_reader():
        pass


def test_member_missing_from_archive_is_rejected_not_faked(tmp_path):
    files, _split_lines = build_fake_dataset()
    files["splits/color_test.txt"] += b"raw/color/Grape___healthy/zz___ghost 1.JPG\n"
    out = tmp_path / "pv"
    run_prepare(hf_source(tmp_path, FakeHub(files)), out, max_per_class=10)
    prov = json.loads((out / "provenance.json").read_text())
    assert any(r["reasons"] == ["missing_in_source"] for r in prov["rejected"])
    assert all("ghost" not in r["source"]["path"] for r in read_manifest(out / "manifest.jsonl"))


def test_existing_manifest_needs_overwrite(tmp_path):
    files, _ = build_fake_dataset()
    run_prepare(hf_source(tmp_path, FakeHub(files)), tmp_path / "pv", max_per_class=1)
    with pytest.raises(InputError, match="overwrite"):
        run_prepare(hf_source(tmp_path, FakeHub(files)), tmp_path / "pv", max_per_class=1)
    run_prepare(hf_source(tmp_path, FakeHub(files)), tmp_path / "pv", max_per_class=1, overwrite=True)


# -- selection --------------------------------------------------------------------
def _e(key, label="esca", split="train", group=None, gs="leaf_map"):
    return SourceEntry(key, label, label, split, split, group, gs if group else "none")


def test_select_round_robin_over_groups_and_bounds():
    entries = [_e(f"g{g}-{i}", group=f"G{g}") for g in range(3) for i in range(4)]
    picked = select_entries(entries, 3, seed=0)
    assert len(picked) == 3
    assert len({e.group_id for e in picked}) == 3  # one image per leaf first
    assert len(select_entries(entries, 5, seed=0)) == 5
    assert len(select_entries(entries, None, seed=0)) == 12
    assert select_entries(entries, 3, seed=1) == select_entries(list(reversed(entries)), 3, seed=1)
    with pytest.raises(InputError):
        select_entries(entries, 0, seed=0)


def test_select_per_split_and_class():
    entries = [
        _e(f"{s}{c}{i}", label=c, split=s) for s in ("train", "test") for c in ("esca", "healthy") for i in range(5)
    ]
    picked = select_entries(entries, 2, seed=0)
    assert len(picked) == 8


def test_group_leakage_ignores_fallback_groups():
    recs = [
        {"group_id": "A", "grouping_source": "leaf_map", "split": "train"},
        {"group_id": "A", "grouping_source": "leaf_map", "split": "test"},
        {"group_id": "fallback_x", "grouping_source": "filename_fallback", "split": "train"},
        {"group_id": "fallback_x", "grouping_source": "filename_fallback", "split": "test"},
    ]
    assert group_leakage(recs) == ["A"]


# -- local class directories ------------------------------------------------------
def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_local_split_dirs_unknown_classes_and_groups(tmp_path):
    root = tmp_path / "in"
    _write(root / "train" / "Black_Rot" / "a.png", png_bytes((200, 0, 0)))
    _write(root / "train" / "Grape___healthy" / "b.png", png_bytes((0, 200, 0)))
    _write(root / "train" / "Grape___healthy" / "notes.txt", b"x")
    _write(root / "test" / "esca" / "c.JPG", png_bytes((128, 128, 128)))
    _write(root / "test" / "Apple___scab" / "d.png", png_bytes((1, 2, 3)))
    groups = tmp_path / "groups.csv"
    groups.write_text("path,group_id\ntrain/Black_Rot/a.png,leaf-1\n")
    src = LocalClassDirs(root, groups_file=groups)
    out = tmp_path / "out"
    run_prepare(src, out, max_per_class=10)
    recs = {r["source"]["path"]: r for r in read_manifest(out / "manifest.jsonl")}
    assert recs["train/Black_Rot/a.png"]["label"] == "black_rot"
    assert recs["train/Black_Rot/a.png"]["group_id"] == "leaf-1"
    assert recs["train/Black_Rot/a.png"]["grouping_source"] == "groups_file"
    assert recs["train/Grape___healthy/b.png"]["grouping_source"] == "none"
    assert recs["test/esca/c.JPG"]["split"] == "test"
    assert not any("Apple" in p for p in recs)
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["source"]["skipped_class_dirs"] == ["test/Apple___scab"]
    assert prov["source"]["skipped_files"] == 1
    assert any("grouping" in c for c in prov["source"]["caveats"])
    assert str(tmp_path) not in json.dumps(prov)


def test_local_strict_labels_rejects_unknown(tmp_path):
    _write(tmp_path / "in" / "mildew" / "a.png", png_bytes())
    with pytest.raises(InputError, match="unrecognized"):
        LocalClassDirs(tmp_path / "in", strict=True).load_entries()


def test_local_without_splits_and_hash_assignment(tmp_path):
    root = tmp_path / "in"
    for i in range(20):
        _write(root / "healthy" / f"{i}.png", png_bytes((0, 200, 0)))
    src = LocalClassDirs(root)
    entries = src.load_entries()
    assert {e.split for e in entries} == {"unspecified"}
    assert any("No source split" in c for c in src.caveats)
    hashed = LocalClassDirs(root, assign_split="hash", test_fraction=0.5, seed=1).load_entries()
    assert {e.split for e in hashed} == {"train", "test"}
    assert all(e.source_split == "unspecified" for e in hashed)
    assert hash_split("k", 1, 0.5) == hash_split("k", 1, 0.5)


def test_local_hash_split_refused_with_source_splits(tmp_path):
    _write(tmp_path / "in" / "train" / "esca" / "a.png", png_bytes())
    with pytest.raises(InputError, match="only allowed"):
        LocalClassDirs(tmp_path / "in", assign_split="hash").load_entries()


def test_local_mixed_layout_and_missing_dir(tmp_path):
    _write(tmp_path / "in" / "train" / "esca" / "a.png", png_bytes())
    _write(tmp_path / "in" / "healthy" / "b.png", png_bytes())
    with pytest.raises(InputError, match="mixed"):
        LocalClassDirs(tmp_path / "in").load_entries()
    with pytest.raises(SourceUnavailableError):
        LocalClassDirs(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(SourceUnavailableError, match="no recognizable"):
        LocalClassDirs(tmp_path / "empty").load_entries()


def test_local_groups_file_validation(tmp_path):
    _write(tmp_path / "in" / "esca" / "a.png", png_bytes())
    bad = tmp_path / "g.csv"
    bad.write_text("file,leaf\n")
    with pytest.raises(InputError, match="path,group_id"):
        LocalClassDirs(tmp_path / "in", groups_file=bad).load_entries()
