"""HF classifier wrapper and ROS 2 node wiring against fake torch/transformers/rclpy."""

import json
import os
import sys
import types
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(__file__))

import pytest

pytest.importorskip("PIL")

from test_pipeline_support import MODEL_LABELS, MockClassifier, rgb_frame_msg

from vineyard_disease.errors import InputError, SourceUnavailableError


# -- fake torch / transformers ---------------------------------------------------
class FakeTensor:
    def __init__(self, rows):
        self.rows = rows

    def to(self, device):
        self.device = device
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


def make_fake_ml(monkeypatch, *, cuda=False, fail_load=False, labels=MODEL_LABELS):
    import math

    calls = {"processor": [], "model": [], "threads": None}
    torch = types.ModuleType("torch")
    torch.__version__ = "0.fake"
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    torch.inference_mode = nullcontext
    torch.set_num_threads = lambda n: calls.__setitem__("threads", n)

    def softmax(t, dim):
        out = []
        for row in t.rows:
            m = max(row)
            e = [math.exp(x - m) for x in row]
            out.append([x / sum(e) for x in e])
        return FakeTensor(out)

    torch.softmax = softmax

    class Processor:
        def __call__(self, images, return_tensors):
            assert return_tensors == "pt"
            calls["images"] = [img.mode for img in images]
            return {"pixel_values": FakeTensor([[0.0]] * len(images))}

    class Model:
        config = types.SimpleNamespace(id2label={str(i): lab for i, lab in enumerate(labels)})

        def eval(self):
            calls["eval"] = True

        def to(self, device):
            calls["device"] = device

        def __call__(self, pixel_values):
            n = len(pixel_values.rows)
            logits = [[5.0 if j == 12 else 0.0 for j in range(len(labels))] for _ in range(n)]
            return types.SimpleNamespace(logits=FakeTensor(logits))

    def proc_from_pretrained(model_id, **kw):
        calls["processor"].append((model_id, kw))
        if fail_load:
            raise OSError("offline")
        return Processor()

    def model_from_pretrained(model_id, **kw):
        calls["model"].append((model_id, kw))
        return Model()

    transformers = types.ModuleType("transformers")
    transformers.__version__ = "4.fake"
    transformers.AutoImageProcessor = types.SimpleNamespace(from_pretrained=proc_from_pretrained)
    transformers.AutoModelForImageClassification = types.SimpleNamespace(from_pretrained=model_from_pretrained)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


def test_hf_classifier_loads_safely_on_cpu(monkeypatch):
    from PIL import Image

    from vineyard_disease.classifier import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, HFImageClassifier

    calls = make_fake_ml(monkeypatch)
    clf = HFImageClassifier(num_threads=2)
    for _, kw in calls["processor"] + calls["model"]:
        assert kw["trust_remote_code"] is False
        assert kw["revision"] == DEFAULT_MODEL_REVISION
    assert calls["model"][0][0] == DEFAULT_MODEL_ID
    assert calls["device"] == "cpu" and calls["eval"] and calls["threads"] == 2
    assert clf.labels == MODEL_LABELS and clf.info["num_labels"] == 38
    probs = clf.predict_proba([Image.new("L", (10, 10)), Image.new("RGBA", (5, 5))])
    assert calls["images"] == ["RGB", "RGB"]
    assert len(probs) == 2 and len(probs[0]) == 38 and sum(probs[0]) == pytest.approx(1.0)
    assert max(range(38), key=lambda i: probs[0][i]) == 12
    assert clf.predict_proba([]) == []


def test_hf_classifier_device_and_load_errors(monkeypatch):
    from vineyard_disease.classifier import HFImageClassifier, softmax

    make_fake_ml(monkeypatch, cuda=False)
    with pytest.raises(InputError, match="CUDA"):
        HFImageClassifier(device="cuda")
    with pytest.raises(InputError, match="unsupported device"):
        HFImageClassifier(device="tpu")
    make_fake_ml(monkeypatch, fail_load=True)
    with pytest.raises(SourceUnavailableError, match="local-files-only"):
        HFImageClassifier()
    assert sum(softmax([1.0, 2.0, 3.0])) == pytest.approx(1.0)


def test_cli_predict_with_fake_hf_backend(monkeypatch, capsys, tmp_path):
    from test_pipeline_support import png_bytes

    from vineyard_disease import cli

    calls = make_fake_ml(monkeypatch)
    img = tmp_path / "leaf.png"
    img.write_bytes(png_bytes())
    out = tmp_path / "p.jsonl"
    assert cli.main(["predict", str(img), "-o", str(out), "--local-files-only", "--model-revision", ""]) == 0
    row = json.loads(out.read_text())
    assert row["status"] == "accepted" and row["predicted_class"] == "esca"
    assert calls["model"][0][1]["local_files_only"] is True and calls["model"][0][1]["revision"] is None


# -- fake rclpy ---------------------------------------------------------------------
def make_fake_ros(monkeypatch, frames):
    state = {"published": [], "ok": True, "init_args": None, "subs": []}

    class Logger:
        def info(self, msg):
            state.setdefault("info", []).append(msg)

        def error(self, msg):
            state.setdefault("errors", []).append(msg)

    class Publisher:
        def __init__(self, topic):
            self.topic = topic

        def publish(self, msg):
            state["published"].append((self.topic, msg.data))

    class Node:
        def __init__(self, name):
            state["node_name"] = name

        def create_publisher(self, typ, topic, depth):
            state["pub_topic"] = topic
            return Publisher(topic)

        def create_subscription(self, typ, topic, cb, qos):
            state["subs"].append((topic, qos))
            state["cb"] = cb

        def get_logger(self):
            return Logger()

        def destroy_node(self):
            state["destroyed"] = True

    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda args=None: state.__setitem__("init_args", args)
    rclpy.ok = lambda: state["ok"]

    def spin(node):
        for f in frames:
            state["cb"](f)
        raise KeyboardInterrupt

    rclpy.spin = spin
    rclpy.shutdown = lambda: state.__setitem__("ok", False)
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = Node
    qos_mod = types.ModuleType("rclpy.qos")
    qos_mod.qos_profile_sensor_data = "SENSOR_QOS"
    sensor = types.ModuleType("sensor_msgs.msg")
    sensor.Image = object
    std = types.ModuleType("std_msgs.msg")
    std.String = type("String", (), {"data": ""})
    for name, mod in {
        "rclpy": rclpy,
        "rclpy.node": node_mod,
        "rclpy.qos": qos_mod,
        "sensor_msgs": types.ModuleType("sensor_msgs"),
        "sensor_msgs.msg": sensor,
        "std_msgs": types.ModuleType("std_msgs"),
        "std_msgs.msg": std,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return state


def test_ros2_node_wiring_publishes_observations_only(monkeypatch, tmp_path):
    from vineyard_disease.cli import build_parser
    from vineyard_disease.gating import GateConfig
    from vineyard_disease.roi import roi_config_from_dict
    from vineyard_disease.ros2_node import run_ros2_observe

    class Exploding:
        header = rgb_frame_msg(1, 1, []).header
        encoding = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

    frames = [rgb_frame_msg(640, 480, [(160, 80, 320, 320, (0, 200, 0))]), Exploding()]
    state = make_fake_ros(monkeypatch, frames)
    log = tmp_path / "obs.ndjson"
    args = build_parser().parse_args(
        ["ros2-observe", "--roi-config", "x.json", "--log-ndjson", str(log), "--min-period-s", "0"]
    )
    cfg = roi_config_from_dict(
        {
            "image_width": 640,
            "image_height": 480,
            "rois": [{"id": "closeup_center", "x": 160, "y": 80, "width": 320, "height": 320}],
        }
    )
    assert run_ros2_observe(args, ["--ros-args", "-r", "x:=y"], MockClassifier(), GateConfig(), cfg) == 0
    assert state["init_args"][1:] == ["--ros-args", "-r", "x:=y"]
    assert state["subs"] == [("/camera/image_raw", "SENSOR_QOS")]
    assert [t for t, _ in state["published"]] == ["/vineyard/leaf_observations"]  # nothing else is published
    doc = json.loads(state["published"][0][1])
    assert doc["observations"][0]["predicted_class"] == "healthy"
    assert "boom" in state["errors"][0]
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(lines) == 2 and "boom" in lines[1]["error"]
    assert state["destroyed"] and state["ok"] is False


def test_cli_ros2_observe_with_fake_ros(monkeypatch, tmp_path):
    import vineyard_disease.classifier as clf_mod
    from vineyard_disease import cli

    state = make_fake_ros(monkeypatch, [rgb_frame_msg(4, 4, [], background=(0, 200, 0))])
    monkeypatch.setattr(clf_mod, "load_classifier", lambda args: MockClassifier())
    roi = tmp_path / "roi.json"
    roi.write_text(json.dumps({"rois": [{"id": "a", "x": 0, "y": 0, "width": 4, "height": 4}]}))
    assert cli.main(["ros2-observe", "--roi-config", str(roi), "--log-ndjson", "", "--image-topic", "/cam"]) == 0
    assert state["subs"][0][0] == "/cam" and len(state["published"]) == 1
