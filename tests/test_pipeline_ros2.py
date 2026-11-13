"""ROS 2 adapter logic without ROS: image conversion, ROIs, observations."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest

pytest.importorskip("PIL")

from test_pipeline_support import MODEL_LABELS, FakeImageMsg, MockClassifier, rgb_frame_msg

from vineyard_disease.errors import InputError, MissingDependencyError, UnsupportedModelError
from vineyard_disease.gating import GateConfig
from vineyard_disease.roi import FULL_FRAME_MODE, Roi, RoiError, load_roi_config, roi_config_from_dict
from vineyard_disease.ros2_node import NdjsonLog, ObservationPipeline, image_msg_to_pil

RED, GREEN, BLUE = (200, 0, 0), (0, 200, 0), (0, 0, 200)


@pytest.mark.parametrize("encoding", ["rgb8", "bgr8"])
def test_image_msg_conversion_colour_order(encoding):
    msg = rgb_frame_msg(4, 2, [(0, 0, 2, 2, RED)], encoding=encoding, background=BLUE)
    img = image_msg_to_pil(msg)
    assert img.mode == "RGB" and img.size == (4, 2)
    assert img.getpixel((0, 0)) == RED and img.getpixel((3, 1)) == BLUE


def test_image_msg_padding_alpha_and_mono():
    row = bytes((1, 2, 3, 4) * 2) + b"\x00\x00"  # 2 rgba8 pixels + 2 bytes padding
    img = image_msg_to_pil(FakeImageMsg(2, 2, "bgra8", row * 2, step=10))
    assert img.getpixel((1, 1)) == (3, 2, 1)
    mono = image_msg_to_pil(FakeImageMsg(3, 1, "mono8", bytes((10, 20, 30))))
    assert mono.getpixel((2, 0)) == (30, 30, 30)


def test_image_msg_rejects_bad_input():
    with pytest.raises(ValueError, match="unsupported"):
        image_msg_to_pil(FakeImageMsg(1, 1, "16UC1", b"\x00\x00", step=2))
    with pytest.raises(ValueError, match="dimensions"):
        image_msg_to_pil(FakeImageMsg(4, 4, "rgb8", b"\x00" * 10))


def test_roi_config_parsing(tmp_path):
    cfg = roi_config_from_dict(
        {"image_width": 40, "image_height": 20, "rois": [{"id": "a", "x": 0, "y": 0, "width": 10, "height": 10}]}
    )
    assert cfg.rois[0].box(40, 20) == (0, 0, 10, 10)
    cfg.check_frame(40, 20)
    with pytest.raises(RoiError, match="made for"):
        cfg.check_frame(80, 40)
    p = tmp_path / "rois.json"
    p.write_text(json.dumps({"normalized": True, "rois": [{"x": 0.5, "y": 0, "width": 0.5, "height": 1}]}))
    loaded = load_roi_config(p)
    assert loaded.rois[0].id == "roi_0" and loaded.rois[0].box(40, 20) == (20, 0, 40, 20)
    for bad in (
        {},
        {"rois": []},
        {"rois": [{"x": 0}]},
        {
            "rois": [
                {"id": "a", "x": 0, "y": 0, "width": 1, "height": 1},
                {"id": "a", "x": 0, "y": 0, "width": 1, "height": 1},
            ]
        },
        {"image_width": 3, "rois": [{"x": 0, "y": 0, "width": 1, "height": 1}]},
    ):
        with pytest.raises(InputError):
            roi_config_from_dict(bad)
    with pytest.raises(InputError):
        Roi("n", 0.8, 0, 0.5, 0.5, normalized=True)
    with pytest.raises(InputError):
        Roi("z", 0, 0, 0, 5)
    with pytest.raises(InputError):
        load_roi_config(tmp_path / "missing.json")


def make_pipeline(tmp_path, **kw):
    published = []
    log = NdjsonLog(tmp_path / "logs" / "obs.ndjson")
    pipe = ObservationPipeline(
        MockClassifier(), GateConfig(), publish=published.append, log=log, source_topic="/camera/image_raw", **kw
    )
    return pipe, published, tmp_path / "logs" / "obs.ndjson"


def test_roi_mode_publishes_json_and_ndjson(tmp_path):
    cfg = roi_config_from_dict(
        {
            "image_width": 60,
            "image_height": 30,
            "rois": [
                {"id": "leaf_b", "x": 30, "y": 0, "width": 30, "height": 30},
                {"id": "leaf_a", "x": 0, "y": 0, "width": 30, "height": 30},
                {"id": "leaf_out", "x": 50, "y": 0, "width": 30, "height": 30},
            ],
        }
    )
    pipe, published, log_path = make_pipeline(tmp_path, roi_config=cfg)
    msg = rgb_frame_msg(60, 30, [(0, 0, 30, 30, RED), (30, 0, 30, 30, GREEN)])
    doc = pipe.handle_msg(msg)
    assert len(published) == 1 and json.loads(published[0]) == doc
    obs = {o["roi_id"]: o for o in doc["observations"]}
    assert obs["leaf_a"]["predicted_class"] == "black_rot"
    assert obs["leaf_b"]["predicted_class"] == "healthy"
    assert obs["leaf_out"]["status"] == "roi_error"
    assert doc["actuation"] == "none" and doc["ground_truth_used"] is False
    assert doc["crop_mode"] == "roi_config" and doc["stamp"] == {"sec": 1, "nanosec": 2}
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["frame_id"] == "camera_optical"


def test_full_frame_mode_is_explicit(tmp_path):
    pipe, _published, _ = make_pipeline(tmp_path, full_frame_single_leaf=True)
    doc = pipe.handle_msg(rgb_frame_msg(8, 8, [], background=BLUE))
    assert doc["crop_mode"] == FULL_FRAME_MODE
    assert doc["input_assumption"] == "frame is a single leaf crop"
    assert doc["observations"][0]["status"] == "non_grape"
    with pytest.raises(InputError):
        ObservationPipeline(MockClassifier(), GateConfig())  # neither ROI nor full-frame
    with pytest.raises(InputError):
        ObservationPipeline(
            MockClassifier(),
            GateConfig(),
            roi_config=roi_config_from_dict({"rois": [{"x": 0, "y": 0, "width": 1, "height": 1}]}),
            full_frame_single_leaf=True,
        )


def test_frame_size_mismatch_and_bad_encoding_are_reported(tmp_path):
    cfg = roi_config_from_dict(
        {"image_width": 10, "image_height": 10, "rois": [{"x": 0, "y": 0, "width": 5, "height": 5}]}
    )
    pipe, published, _ = make_pipeline(tmp_path, roi_config=cfg)
    doc = pipe.handle_msg(rgb_frame_msg(8, 8, []))
    assert "made for" in doc["error"] and doc["observations"] == []
    doc = pipe.handle_msg(FakeImageMsg(1, 1, "32FC1", b"\x00" * 4, step=4))
    assert "unsupported" in doc["error"] and len(published) == 2


def test_throttling(tmp_path):
    t = [0.0]
    pipe, published, _ = make_pipeline(tmp_path, full_frame_single_leaf=True, min_period_s=1.0, clock=lambda: t[0])
    msg = rgb_frame_msg(4, 4, [], background=GREEN)
    assert pipe.handle_msg(msg) is not None
    t[0] = 0.5
    assert pipe.handle_msg(msg) is None
    t[0] = 1.6
    assert pipe.handle_msg(msg) is not None
    assert (pipe.frames_seen, pipe.frames_processed, len(published)) == (3, 2, 2)


def test_pipeline_requires_supported_model(tmp_path):
    with pytest.raises(UnsupportedModelError):
        ObservationPipeline(MockClassifier(labels=MODEL_LABELS[:12]), GateConfig(), full_frame_single_leaf=True)


def test_run_without_rclpy_raises_missing_dependency(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "rclpy" or name.startswith("rclpy."):
            raise ImportError("no rclpy")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "rclpy", raising=False)
    from vineyard_disease.ros2_node import run_ros2_observe

    with pytest.raises(MissingDependencyError, match="rclpy"):
        run_ros2_observe(None, [], MockClassifier(), GateConfig(), None)


def test_disabled_log_is_noop(tmp_path):
    NdjsonLog(None).write({"a": 1})
    NdjsonLog("").write({"a": 1})
