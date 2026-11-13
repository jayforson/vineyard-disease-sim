"""Shared fakes for the pipeline tests (no ML, network or ROS dependencies)."""

from __future__ import annotations

import io
import json
import re
import zipfile

from vineyard_disease.sources.http import HttpClient, Response

# The 38 labels of the default checkpoint (public config.json), in index order.
MODEL_LABELS = [
    "Apple Scab",
    "Apple with Black Rot",
    "Cedar Apple Rust",
    "Healthy Apple",
    "Healthy Blueberry Plant",
    "Cherry with Powdery Mildew",
    "Healthy Cherry Plant",
    "Corn (Maize) with Cercospora and Gray Leaf Spot",
    "Corn (Maize) with Common Rust",
    "Corn (Maize) with Northern Leaf Blight",
    "Healthy Corn (Maize) Plant",
    "Grape with Black Rot",
    "Grape with Esca (Black Measles)",
    "Grape with Isariopsis Leaf Spot",
    "Healthy Grape Plant",
    "Orange with Citrus Greening",
    "Peach with Bacterial Spot",
    "Healthy Peach Plant",
    "Bell Pepper with Bacterial Spot",
    "Healthy Bell Pepper Plant",
    "Potato with Early Blight",
    "Potato with Late Blight",
    "Healthy Potato Plant",
    "Healthy Raspberry Plant",
    "Healthy Soybean Plant",
    "Squash with Powdery Mildew",
    "Strawberry with Leaf Scorch",
    "Healthy Strawberry Plant",
    "Tomato with Bacterial Spot",
    "Tomato with Early Blight",
    "Tomato with Late Blight",
    "Tomato with Leaf Mold",
    "Tomato with Septoria Leaf Spot",
    "Tomato with Spider Mites or Two-spotted Spider Mite",
    "Tomato with Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy Tomato Plant",
]
IDX = {label: i for i, label in enumerate(MODEL_LABELS)}


def dist(peaks: dict[str, float], labels=MODEL_LABELS) -> list[float]:
    """Probability vector with given label masses; the rest spread uniformly."""
    rest = 1.0 - sum(peaks.values())
    others = [lab for lab in labels if lab not in peaks]
    return [peaks.get(lab, rest / len(others)) for lab in labels]


# Mock classifier: decide by the image's dominant colour.
COLOR_RULES = {
    (200, 0, 0): {"Grape with Black Rot": 0.9},
    (0, 200, 0): {"Healthy Grape Plant": 0.8},
    (0, 0, 200): {"Tomato with Septoria Leaf Spot": 0.7, "Grape with Esca (Black Measles)": 0.2},
    (128, 128, 128): {"Grape with Esca (Black Measles)": 0.3, "Grape with Black Rot": 0.25},
    (200, 200, 0): {"Grape with Isariopsis Leaf Spot": 0.6},
}


class MockClassifier:
    def __init__(self, labels=MODEL_LABELS, rules=None):
        self.labels = list(labels)
        self.rules = rules or COLOR_RULES
        self.info = {"model_id": "mock/classifier", "revision": "test", "device": "cpu", "num_labels": len(labels)}
        self.calls: list[int] = []
        self.sizes: list[tuple[int, int]] = []

    def predict_proba(self, images):
        self.calls.append(len(images))
        out = []
        for img in images:
            self.sizes.append(img.size)
            rgb = img.convert("RGB").resize((1, 1)).getpixel((0, 0))
            rule = min(self.rules, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, rgb)))
            out.append(dist(self.rules[rule], self.labels))
        return out


def png_bytes(color=(200, 0, 0), size=(64, 64), *, noise=True, mode="RGB") -> bytes:
    from PIL import Image

    img = Image.new(mode, size, color if mode == "RGB" else 128)
    if noise:  # add texture so the quality checker does not flag low contrast
        px = img.load()
        for x in range(0, size[0], 4):
            for y in range(0, size[1], 4):
                if mode == "RGB":
                    px[x, y] = tuple(min(255, c + 60) for c in color)
                else:
                    px[x, y] = 250
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fake Hugging Face server (resolve URLs, redirect to "CDN", Range support).
REV = "test-revision"
BASE = f"https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{REV}/"
CDN = "https://cdn.example.invalid/data.zip?sig=abc"

GRAPE_FILES = {
    # (split, class): [file names]
    ("train", "Grape___Black_rot"): [
        "a1___FAM_B.Rot 0001.JPG",
        "a2___FAM_B.Rot 0002.JPG",
        "a3___FAM_B.Rot 0003.JPG",
        "a4___FAM_B.Rot 0004.JPG",
    ],
    ("train", "Grape___healthy"): ["h1___Mt.N.V_HL 0001.JPG", "h2___Mt.N.V_HL 0002.JPG"],
    ("train", "Grape___Esca_(Black_Measles)"): ["e1___FAM_B.Msls 0001.JPG"],
    ("train", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"): ["l1___FAM_L.Blight 0001.JPG"],
    ("test", "Grape___Black_rot"): ["b1___FAM_B.Rot 0101.JPG"],
    ("test", "Grape___Esca_(Black_Measles)"): ["e2___FAM_B.Msls 0101.JPG", "e3___FAM_B.Msls 0102.JPG"],
    ("test", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"): ["l2___FAM_L.Blight 0101.JPG"],
    ("test", "Grape___healthy"): ["h3___Mt.N.V_HL 0101.JPG"],
}
LEAF_MAP = {
    "fam_b.rot 0001": ["Grape___Black_rot:::1.0"],
    "fam_b.rot 0002": ["Grape___Black_rot:::1.0"],
    "fam_b.rot 0003": ["Grape___Black_rot:::2.0"],
    "fam_b.rot 0004": ["Apple___Black_rot:::9.0", "Grape___Black_rot:::3.0"],
    "fam_b.rot 0101": ["Grape___Black_rot:::50.0"],
    "fam_b.msls 0001": ["Grape___Esca_(Black_Measles):::1.0"],
    "fam_b.msls 0101": ["Grape___Esca_(Black_Measles):::60.0"],
    "fam_b.msls 0102": ["Grape___Esca_(Black_Measles):::60.0"],
    "fam_l.blight 0001": ["Grape___Leaf_blight_(Isariopsis_Leaf_Spot):::1.0"],
    "fam_l.blight 0101": ["Peach___healthy:::3.0", "Apple___healthy:::4.0"],  # ambiguous -> fallback
}
CLASS_COLORS = {
    "Grape___Black_rot": (200, 0, 0),
    "Grape___healthy": (0, 200, 0),
    "Grape___Esca_(Black_Measles)": (128, 128, 128),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (200, 200, 0),
}


def build_fake_dataset(prefix: str = "", extra_members: dict | None = None):
    split_lines = {"train": [], "test": []}
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_STORED) as zf:
        for (split, cls), names in GRAPE_FILES.items():
            for name in names:
                rel = f"raw/color/{cls}/{name}"
                split_lines[split].append(rel)
                zf.writestr(prefix + rel, png_bytes(CLASS_COLORS[cls]))
        rel = "raw/color/Apple___healthy/x1___RS_HL 0001.JPG"
        split_lines["train"].append(rel)
        zf.writestr(prefix + rel, png_bytes((0, 200, 0)))
        for rel, data in (extra_members or {}).items():
            zf.writestr(prefix + rel, data)
    files = {
        "splits/color_train.txt": ("\n".join(split_lines["train"]) + "\n").encode(),
        "splits/color_test.txt": ("\n".join(split_lines["test"]) + "\n\n").encode(),
        "leaf_grouping/leaf-map.json": json.dumps(LEAF_MAP).encode(),
        "data.zip": zbuf.getvalue(),
    }
    return files, split_lines


class FakeHub:
    """Transport serving ``files`` under BASE; data.zip redirects to CDN."""

    def __init__(
        self,
        files: dict[str, bytes],
        *,
        fail: set[str] | None = None,
        no_range: bool = False,
        expire_once: bool = False,
    ):
        self.files = files
        self.fail = fail or set()
        self.no_range = no_range
        self.expire_once = expire_once
        self.log: list[tuple[str, str | None]] = []
        self.auth_seen: list[str] = []

    def __call__(self, url, headers, timeout):
        rng = headers.get("Range")
        self.log.append((url, rng))
        if "Authorization" in headers:
            self.auth_seen.append(url)
        if url == CDN:
            if self.expire_once:
                self.expire_once = False
                return Response(403, {}, b"expired", url)
            return self._serve(self.files["data.zip"], rng, url)
        if not url.startswith(BASE):
            return Response(404, {}, b"", url)
        rel = url[len(BASE) :]
        if rel in self.fail:
            raise OSError("network unreachable")
        if rel not in self.files:
            return Response(404, {}, b"", url)
        if rel == "data.zip":
            return self._serve(self.files[rel], rng, CDN)
        return Response(200, {}, self.files[rel], url)

    def _serve(self, data, rng, url):
        if rng is None or self.no_range:
            return Response(200, {"content-length": str(len(data))}, data, url)
        m = re.match(r"bytes=(\d+)-(\d+)", rng)
        start, end = int(m.group(1)), min(int(m.group(2)), len(data) - 1)
        return Response(206, {"content-range": f"bytes {start}-{end}/{len(data)}"}, data[start : end + 1], url)


def client_for(hub, **kw) -> HttpClient:
    kw.setdefault("backoff_s", 0.0)
    return HttpClient(hub, **kw)


class FakeStamp:
    def __init__(self, sec=1, nanosec=2):
        self.sec, self.nanosec = sec, nanosec


class FakeHeader:
    def __init__(self, frame_id="camera_optical", sec=1):
        self.stamp = FakeStamp(sec)
        self.frame_id = frame_id


class FakeImageMsg:
    """Duck-typed sensor_msgs/msg/Image."""

    def __init__(self, width, height, encoding, data, step=None, frame_id="camera_optical"):
        self.width, self.height, self.encoding = width, height, encoding
        self.step = (
            step
            if step is not None
            else width * {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(encoding, 3)
        )
        self.data = data
        self.is_bigendian = 0
        self.header = FakeHeader(frame_id)


def rgb_frame_msg(width, height, regions, encoding="rgb8", background=(0, 0, 200)):
    """Build an Image message; ``regions`` is [(x, y, w, h, (r, g, b))]."""
    pixels = [[background for _ in range(width)] for _ in range(height)]
    for x0, y0, w, h, color in regions:
        for y in range(y0, y0 + h):
            for x in range(x0, x0 + w):
                pixels[y][x] = color
    data = bytearray()
    for row in pixels:
        for r, g, b in row:
            data += bytes((b, g, r) if encoding == "bgr8" else (r, g, b))
    return FakeImageMsg(width, height, encoding, bytes(data))
