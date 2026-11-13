#!/usr/bin/env python3
"""Generate a self-contained, deterministic Gazebo Harmonic vineyard fixture.

Stylized textures are test fixtures, NOT clinically validated disease examples.
Use --manifest to map real prepared leaf photos to individual visual assets.
No texture, image, dataset or external Gazebo Fuel asset is bundled in Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ("healthy", "black_rot", "esca", "isariopsis_leaf_spot")


def sub(parent, tag, text=None, **attrs):
    node = ET.SubElement(parent, tag, attrs)
    if text is not None:
        node.text = str(text)
    return node


def pose(parent, xyz, rpy=(0, 0, 0)):
    sub(parent, "pose", " ".join(f"{x:.6f}" for x in (*xyz, *rpy)))


def plugin(parent, filename, name):
    return sub(parent, "plugin", filename=filename, name=name)


def solid(world, name, xyz, shape, size, color, rpy=(0, 0, 0), collision=True):
    model = sub(world, "model", name=name)
    sub(model, "static", "true")
    pose(model, xyz, rpy)
    link = sub(model, "link", name="body")
    visual = sub(link, "visual", name="visual")
    geometry = sub(visual, "geometry")
    node = sub(geometry, shape)
    if shape == "box":
        sub(node, "size", " ".join(map(str, size)))
    elif shape == "cylinder":
        sub(node, "radius", size[0])
        sub(node, "length", size[1])
    else:
        raise ValueError(shape)
    material = sub(visual, "material")
    sub(material, "ambient", color)
    sub(material, "diffuse", color)
    if collision:
        coll = sub(link, "collision", name="collision")
        coll.append(ET.fromstring(ET.tostring(geometry)))
    return model


def leaf_obj(path):
    """Per-leaf OBJ/MTL preserves texture bindings in Gazebo's mesh importer."""
    outline = [
        (0, 0.17),
        (0.055, 0.115),
        (0.14, 0.12),
        (0.095, 0.025),
        (0.14, -0.015),
        (0.075, -0.075),
        (0.02, -0.08),
        (0, -0.14),
        (-0.02, -0.08),
        (-0.075, -0.075),
        (-0.14, -0.015),
        (-0.095, 0.025),
        (-0.14, 0.12),
        (-0.055, 0.115),
    ]
    points = [(0, 0)] + outline
    lines = [f"mtllib {path.stem}.mtl", f"o {path.stem}"]
    lines += [f"v {x} {y} 0" for x, y in points]
    lines += [f"vt {(x + 0.15) / 0.3} {(y + 0.15) / 0.33}" for x, y in points]
    lines += ["vn 0 0 1", "vn 0 0 -1", f"usemtl {path.stem}"]
    for index in range(2, len(points) + 1):
        next_index = index + 1 if index < len(points) else 2
        lines.append(f"f 1/1/1 {index}/{index}/1 {next_index}/{next_index}/1")
        lines.append(f"f 1/1/2 {next_index}/{next_index}/2 {index}/{index}/2")
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".mtl").write_text(
        f"newmtl {path.stem}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 1\nd 1\nmap_Kd {path.stem}.png\n"
    )


def fixture_texture(path, label, seed):
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    image = Image.new("RGB", (256, 256), (45, 115, 39))
    draw = ImageDraw.Draw(image)
    draw.line((128, 245, 128, 12), fill=(130, 170, 80), width=3)
    for y in range(45, 230, 32):
        draw.line((128, y, 30, y - 40), fill=(100, 145, 55), width=2)
        draw.line((128, y, 226, y - 40), fill=(100, 145, 55), width=2)
    if label == "black_rot":
        for _ in range(14):
            x, y = rng.randrange(40, 215), rng.randrange(35, 220)
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(152, 106, 57))
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(52, 40, 26))
    elif label == "esca":
        for y in range(35, 225, 28):
            draw.line((24, y - 12, 112, y + 10), fill=(195, 154, 52), width=13)
            draw.line((144, y + 10, 235, y - 12), fill=(195, 154, 52), width=13)
    elif label == "isariopsis_leaf_spot":
        for _ in range(35):
            x, y = rng.randrange(30, 230), rng.randrange(20, 235)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(105, 64, 34))
    image.save(path)


def load_photos(manifest, source_split="train"):
    if manifest is None:
        return {}
    provenance_path = manifest.with_name("provenance.json")
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    dataset_source = provenance.get("source", {})
    photos = {c: [] for c in CLASSES}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        # Prepared pipeline manifests use label/path; aliases ease external imports.
        label = record.get("label", record.get("class_name"))
        if label not in photos:
            continue
        if source_split != "all" and record.get("split") != source_split:
            continue
        record["_dataset_source"] = dataset_source
        original = record.get("image", record.get("path", record.get("image_path")))
        if not original:
            raise ValueError("Manifest row is missing path/image_path")
        path = Path(original)
        if not path.is_absolute():
            path = manifest.parent / path
        if not path.is_file():
            raise FileNotFoundError(path)
        photos[label].append((path, record))
    if not any(photos.values()):
        raise ValueError("No supported grape photos in manifest")
    return photos


def generate(output, rows=3, vines=6, seed=42, manifest=None, view="wide", source_split="train"):
    if rows < 1 or vines < 1 or rows > 20 or vines > 100:
        raise ValueError("Use 1..20 rows and 1..100 vines per row")
    if view not in ("wide", "leaf") or source_split not in ("train", "test", "all"):
        raise ValueError("Invalid view or source split")
    photos = load_photos(manifest, source_split)
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    # Only remove files owned by this generator, never unrelated assets.
    for old_asset in assets.glob("leaf_r*_v*_*"):
        if old_asset.is_file() and old_asset.suffix in (".obj", ".mtl", ".png"):
            old_asset.unlink()
    rng = random.Random(seed)
    root = ET.Element("sdf", version="1.9")
    world = sub(root, "world", name="vineyard")
    sub(world, "gravity", "0 0 -9.81")
    physics = sub(world, "physics", name="physics", type="ignored")
    sub(physics, "max_step_size", 0.01)
    sub(physics, "real_time_factor", 1)
    plugin(world, "gz-sim-physics-system", "gz::sim::systems::Physics")
    plugin(world, "gz-sim-user-commands-system", "gz::sim::systems::UserCommands")
    plugin(world, "gz-sim-scene-broadcaster-system", "gz::sim::systems::SceneBroadcaster")
    sensors = plugin(world, "gz-sim-sensors-system", "gz::sim::systems::Sensors")
    sub(sensors, "render_engine", "ogre2")
    scene = sub(world, "scene")
    sub(scene, "ambient", ".65 .65 .65 1")
    sub(scene, "background", ".62 .78 .92 1")
    sub(scene, "shadows", "true")
    sun = sub(world, "light", name="sun", type="directional")
    pose(sun, (0, 0, 15))
    sub(sun, "cast_shadows", "true")
    sub(sun, "diffuse", ".9 .9 .85 1")
    sub(sun, "specular", ".2 .2 .2 1")
    sub(sun, "direction", "-.4 .2 -1")
    solid(world, "ground", (vines, (rows - 1) * 1.5, -0.05), "box", (vines * 2 + 8, rows * 3 + 8, 0.1), ".35 .27 .15 1")
    truth = []
    for row in range(rows):
        y = row * 3.0
        solid(
            world,
            f"cover_{row}",
            (vines, y + 1.5, 0.005),
            "box",
            (vines * 2 + 3, 1.15, 0.015),
            ".2 .32 .12 1",
            collision=False,
        )
        for x in (-0.7, vines * 2 - 0.7):
            solid(world, f"post_{row}_{x}", (x, y, 1), "box", (0.08, 0.08, 2), ".48 .38 .24 1")
        for height in (0.8, 1.25, 1.7):
            solid(
                world,
                f"wire_{row}_{height}",
                (vines - 1, y, height),
                "cylinder",
                (0.008, vines * 2),
                ".35 .35 .38 1",
                (0, math.pi / 2, 0),
                collision=False,
            )
        solid(
            world,
            f"drip_{row}",
            (vines - 1, y, 0.12),
            "cylinder",
            (0.02, vines * 2),
            ".05 .05 .05 1",
            (0, math.pi / 2, 0),
        )
        for vine in range(vines):
            x = vine * 2.0
            solid(world, f"trunk_{row}_{vine}", (x, y, 0.5), "cylinder", (0.06, 1), ".34 .2 .09 1")
            solid(
                world, f"cordon_{row}_{vine}", (x, y, 1), "cylinder", (0.035, 1.2), ".34 .2 .09 1", (0, math.pi / 2, 0)
            )
            solid(world, f"emitter_{row}_{vine}", (x, y, 0.12), "box", (0.06, 0.055, 0.045), ".04 .04 .04 1")
            for index in range(8):
                label = CLASSES[(row * vines + vine + index) % len(CLASSES)]
                leaf_id = f"leaf_r{row}_v{vine}_{index}"
                texture = assets / f"{leaf_id}.png"
                source = {"kind": "stylized_synthetic", "clinically_validated": False}
                candidates = photos.get(label, [])
                if candidates:
                    photo, record = candidates[(vine + index) % len(candidates)]
                    from PIL import Image

                    with Image.open(photo) as image:
                        image.convert("RGB").save(texture)
                    source = {
                        "kind": "photo_texture",
                        "source_sha256": hashlib.sha256(photo.read_bytes()).hexdigest(),
                        "source_split": record.get("split"),
                        "source_group_id": record.get("group_id"),
                        "dataset": record.get("_dataset_source", {}).get("dataset", "unspecified"),
                        "dataset_url": record.get("_dataset_source", {}).get("dataset_url"),
                        "license": record.get("_dataset_source", {}).get("license", "unspecified"),
                        "attribution": record.get("_dataset_source", {}).get("citation", "not supplied"),
                        "clinically_validated": False,
                    }
                else:
                    fixture_texture(texture, label, seed + row * 10000 + vine * 100 + index)
                xyz = (x + (index % 4 - 1.5) * 0.32, y + rng.uniform(-0.16, 0.16), 1.1 + (index // 4) * 0.45)
                model = sub(world, "model", name=leaf_id)
                sub(model, "static", "true")
                pose(model, xyz)
                link = sub(model, "link", name="leaf")
                visual = sub(link, "visual", name="leaf_visual")
                pose(visual, (0, 0, 0), (math.pi / 2 + rng.uniform(-0.25, 0.25), 0, rng.uniform(-0.35, 0.35)))
                mesh = sub(sub(visual, "geometry"), "mesh")
                leaf_obj(assets / f"{leaf_id}.obj")
                sub(mesh, "uri", f"assets/{leaf_id}.obj")
                label_plugin = plugin(visual, "gz-sim-label-system", "gz::sim::systems::Label")
                sub(label_plugin, "label", CLASSES.index(label) + 1)
                truth.append(
                    {
                        "leaf_id": leaf_id,
                        "vine_id": f"r{row}_v{vine}",
                        "label": label,
                        "position_m": xyz,
                        "source": source,
                    }
                )
    if view not in ("wide", "leaf"):
        raise ValueError("view must be wide or leaf")
    # Static rover/camera fixture: move via /world/vineyard/set_pose for viewpoint tests.
    target = truth[0]["position_m"]
    rig_xyz = (1, -2, 0) if view == "wide" else (target[0], target[1] - 0.55, target[2] - 1.35)
    rig = solid(world, "camera_rig", rig_xyz, "box", (0.7, 0.45, 0.22), ".15 .19 .23 1", collision=False)
    rig.find("link").find("visual").find("geometry").find("box").find("size").text = ".7 .45 .22"
    link = rig.find("link")
    for name, sensor_type in (("rgb", "camera"), ("truth", "segmentation")):
        sensor = sub(link, "sensor", name=name, type=sensor_type)
        pose(sensor, (0, 0, 1.35), (0, 0, math.pi / 2))
        sub(sensor, "topic", f"/vineyard/camera/{name}")
        sub(sensor, "update_rate", 5)
        camera = sub(sensor, "camera")
        if sensor_type == "segmentation":
            sub(camera, "segmentation_type", "instance")
        sub(camera, "horizontal_fov", 1.0)
        image = sub(camera, "image")
        sub(image, "width", 640)
        sub(image, "height", 480)
        clip = sub(camera, "clip")
        sub(clip, "near", 0.05)
        sub(clip, "far", 30)
    ET.indent(root)
    world_path = output / "vineyard.sdf"
    ET.ElementTree(root).write(world_path, encoding="utf-8", xml_declaration=True)
    metadata = {
        "schema_version": 1,
        "seed": seed,
        "rows": rows,
        "vines_per_row": vines,
        "classes": {str(i + 1): c for i, c in enumerate(CLASSES)},
        "purpose": "perception integration fixture; not a disease biology simulator",
        "view": view,
        "photo_source_split": source_split,
        "redistribution_note": "Photo textures retain source provenance/license metadata; they are excluded from Git. Source metadata is not a redistribution authorization.",
        "leaves": truth,
    }
    (output / "ground_truth.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return world_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("simulation/generated"))
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--vines", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Prepared real-image JSONL manifest; leaves may fall back to explicitly synthetic textures",
    )
    parser.add_argument(
        "--view", choices=("wide", "leaf"), default="wide", help="Wide vineyard view or close-up single-leaf fixture"
    )
    parser.add_argument(
        "--source-split",
        choices=("train", "test", "all"),
        default="train",
        help="Keep photo source splits separate; default train",
    )
    args = parser.parse_args()
    print(generate(args.output, args.rows, args.vines, args.seed, args.manifest, args.view, args.source_split))


if __name__ == "__main__":
    main()
