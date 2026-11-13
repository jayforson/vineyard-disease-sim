"""Tests only create isolated temporary assets; no Gazebo or network needed."""

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "world_generator", Path(__file__).parents[1] / "simulation/generate_world.py"
)
world = importlib.util.module_from_spec(spec)
spec.loader.exec_module(world)


def test_world_is_self_contained(tmp_path):
    path = world.generate(tmp_path, rows=2, vines=2)
    root = ET.parse(path).getroot()
    assert root.attrib["version"] == "1.9"
    assert root.find("world").attrib["name"] == "vineyard"
    assert len(root.findall(".//sensor")) == 2
    for uri in root.findall(".//uri") + root.findall(".//albedo_map"):
        assert (tmp_path / uri.text).is_file()
    for uri in root.findall(".//uri"):
        mesh = tmp_path / uri.text
        assert mesh.suffix == ".obj"
        material = mesh.with_suffix(".mtl").read_text()
        assert f"map_Kd {mesh.stem}.png" in material
        assert mesh.with_suffix(".png").is_file()
    truth = json.loads((tmp_path / "ground_truth.json").read_text())
    assert len(truth["leaves"]) == 32
    assert len({leaf["leaf_id"] for leaf in truth["leaves"]}) == 32
    assert {leaf["label"] for leaf in truth["leaves"]} == set(world.CLASSES)
    assert all(leaf["source"]["clinically_validated"] is False for leaf in truth["leaves"])


def test_seed_reproducible(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    world.generate(first, rows=1, vines=1, seed=7)
    world.generate(second, rows=1, vines=1, seed=7)
    assert (first / "ground_truth.json").read_bytes() == (second / "ground_truth.json").read_bytes()
    for file in (first / "assets").glob("*.png"):
        assert file.read_bytes() == (second / "assets" / file.name).read_bytes()


def test_real_photo_provenance_and_image_conversion(tmp_path):
    source = tmp_path / "sample.jpg"
    Image.new("RGB", (120, 100), "green").save(source)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"path": "sample.jpg", "label": "healthy", "split": "train"}) + "\n")
    output = tmp_path / "scene"
    world.generate(output, rows=1, vines=1, manifest=manifest)
    truth = json.loads((output / "ground_truth.json").read_text())
    real = [leaf for leaf in truth["leaves"] if leaf["source"]["kind"] == "photo_texture"]
    assert real and real[0]["source"]["source_split"] == "train"
    assert len(real[0]["source"]["source_sha256"]) == 64
    with Image.open(output / "assets" / (real[0]["leaf_id"] + ".png")) as image:
        assert image.format == "PNG"


def test_empty_or_broken_manifest_never_silently_fakes_data(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")
    with pytest.raises(ValueError, match="No supported"):
        world.generate(tmp_path / "scene", manifest=manifest)
    manifest.write_text(json.dumps({"path": "missing.png", "label": "healthy", "split": "train"}) + "\n")
    with pytest.raises(FileNotFoundError):
        world.generate(tmp_path / "scene", manifest=manifest)


@pytest.mark.parametrize("rows,vines", [(0, 1), (1, 0), (21, 1), (1, 101)])
def test_bound_scene_size(tmp_path, rows, vines):
    with pytest.raises(ValueError):
        world.generate(tmp_path, rows=rows, vines=vines)


def test_photo_split_and_attribution_retained(tmp_path):
    Image.new("RGB", (120, 100), "green").save(tmp_path / "train.png")
    Image.new("RGB", (120, 100), "yellow").save(tmp_path / "test.png")
    manifest = tmp_path / "manifest.jsonl"
    records = [
        {"image": f"{split}.png", "label": "healthy", "split": split, "group_id": split} for split in ("train", "test")
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (tmp_path / "provenance.json").write_text(
        json.dumps(
            {
                "source": {
                    "dataset": "test_fixture",
                    "dataset_url": "https://example.invalid/dataset",
                    "license": "CC-BY-SA-3.0",
                    "citation": "Fixture attribution",
                }
            }
        )
    )
    output = tmp_path / "scene"
    world.generate(output, rows=1, vines=1, manifest=manifest)
    truth = json.loads((output / "ground_truth.json").read_text())
    real = [leaf["source"] for leaf in truth["leaves"] if leaf["source"]["kind"] == "photo_texture"]
    assert real and {source["source_split"] for source in real} == {"train"}
    assert all(source["license"] == "CC-BY-SA-3.0" for source in real)
    assert all(source["attribution"] == "Fixture attribution" for source in real)
    assert all(source["dataset"] == "test_fixture" for source in real)


def test_regeneration_removes_only_owned_stale_assets(tmp_path):
    world.generate(tmp_path, rows=2, vines=2)
    note = tmp_path / "assets" / "user-note.txt"
    note.write_text("keep")
    world.generate(tmp_path, rows=1, vines=1)
    assert note.read_text() == "keep"
    assert len(list((tmp_path / "assets").glob("leaf_r*_v*_*.obj"))) == 8
    assert not (tmp_path / "assets" / "leaf_r1_v1_0.obj").exists()


def test_closeup_frame_matches_declared_roi_config(tmp_path):
    path = world.generate(tmp_path, rows=1, vines=1, view="leaf")
    root = ET.parse(path).getroot()
    config = json.loads((Path(__file__).parents[1] / "simulation/closeup-roi.json").read_text())
    for camera in root.findall(".//sensor/camera"):
        assert int(camera.find("image/width").text) == config["image_width"]
        assert int(camera.find("image/height").text) == config["image_height"]
