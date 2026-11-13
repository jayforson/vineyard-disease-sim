"""ROS 2 image-input adapter (observation only, no actuation).

Subscribes to a ``sensor_msgs/msg/Image`` topic, crops known ROIs (or uses
the full frame when explicitly declared a single-leaf crop), classifies each
crop, and publishes one JSON document per processed frame as
``std_msgs/msg/String`` on an observation topic. Every published document is
also appended to an NDJSON log. The node never publishes commands, never
subscribes to simulation ground truth, and takes no spraying or other action.

``rclpy`` and the ROS message packages are imported only by
:func:`run_ros2_observe`; everything else here is testable without ROS.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from . import DISCLAIMER
from .errors import InputError, require
from .gating import GateConfig, LabelSpace, decide
from .roi import FULL_FRAME_MODE, ROI_MODE, Roi, RoiConfig, RoiError

OBSERVATION_SCHEMA = "vineyard_disease/leaf_observation/v1"
SUPPORTED_ENCODINGS = {
    "rgb8": ("RGB", 3),
    "bgr8": ("RGB", 3),
    "rgba8": ("RGBA", 4),
    "bgra8": ("RGBA", 4),
    "mono8": ("L", 1),
}


def image_msg_to_pil(msg):
    """Convert a sensor_msgs/Image-like object (8-bit encodings) to RGB PIL."""
    image_mod = require("PIL.Image", "ROS image conversion", "Install Pillow.")
    enc = str(msg.encoding).lower()
    if enc not in SUPPORTED_ENCODINGS:
        raise ValueError(f"unsupported image encoding '{msg.encoding}' (supported: {sorted(SUPPORTED_ENCODINGS)})")
    mode, channels = SUPPORTED_ENCODINGS[enc]
    width, height, step = int(msg.width), int(msg.height), int(msg.step)
    row = width * channels
    data = bytes(msg.data)
    if width <= 0 or height <= 0 or step < row or len(data) < step * (height - 1) + row:
        raise ValueError("image message dimensions do not match its data")
    if step != row:
        data = b"".join(data[r * step : r * step + row] for r in range(height))
    else:
        data = data[: row * height]
    img = image_mod.frombytes(mode, (width, height), data)
    if enc in ("bgr8", "bgra8"):
        bands = img.split()
        img = image_mod.merge(mode, (bands[2], bands[1], bands[0], *bands[3:]))
    return img.convert("RGB")


class NdjsonLog:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, obj: dict) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")


class ObservationPipeline:
    """Frame -> JSON observation document. Pure Python apart from Pillow."""

    def __init__(
        self,
        classifier,
        gate: GateConfig,
        *,
        roi_config: RoiConfig | None = None,
        full_frame_single_leaf: bool = False,
        source_topic: str = "",
        publish: Callable[[str], None] | None = None,
        log: NdjsonLog | None = None,
        min_period_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if (roi_config is None) == (not full_frame_single_leaf):
            raise InputError("choose exactly one of an ROI config or full-frame single-leaf-crop mode")
        self.classifier = classifier
        self.space = LabelSpace(classifier.labels)
        self.space.require_supported()
        self.gate = gate
        self.roi_config = roi_config
        self.mode = ROI_MODE if roi_config is not None else FULL_FRAME_MODE
        self.source_topic = source_topic
        self.publish = publish or (lambda _s: None)
        self.log = log or NdjsonLog(None)
        self.min_period_s = min_period_s
        self.clock = clock
        self._last: float | None = None
        self.frames_seen = 0
        self.frames_processed = 0
        self.model = {k: classifier.info.get(k) for k in ("model_id", "revision", "device", "num_labels")}

    def _rois(self, w: int, h: int) -> list[Roi]:
        if self.roi_config is None:
            return [Roi("full_frame", 0, 0, w, h)]
        self.roi_config.check_frame(w, h)
        return list(self.roi_config.rois)

    def process_image(self, image, *, stamp: dict, frame_id: str) -> dict:
        w, h = image.size
        doc = {
            "schema": OBSERVATION_SCHEMA,
            "stamp": stamp,
            "frame_id": frame_id,
            "source_topic": self.source_topic,
            "frame_size": {"width": w, "height": h},
            "crop_mode": self.mode,
            "input_assumption": "frame is a single leaf crop"
            if self.mode == FULL_FRAME_MODE
            else "each configured ROI contains a single leaf",
            "ground_truth_used": False,
            "actuation": "none",
            "model": self.model,
            "disclaimer": DISCLAIMER,
            "observations": [],
        }
        try:
            rois = self._rois(w, h)
        except RoiError as exc:
            doc["error"] = str(exc)
            return doc
        crops, kept = [], []
        for roi in rois:
            try:
                crops.append(image.crop(roi.box(w, h)))
                kept.append(roi)
            except RoiError as exc:
                doc["observations"].append(
                    {
                        "roi_id": roi.id,
                        "roi": roi.as_dict(),
                        "status": "roi_error",
                        "predicted_class": None,
                        "score": None,
                        "error": str(exc),
                    }
                )
        probs = self.classifier.predict_proba(crops) if crops else []
        for roi, p in zip(kept, probs):
            obs = {"roi_id": roi.id, "roi": roi.as_dict()}
            obs.update(decide(p, self.space, self.gate))
            doc["observations"].append(obs)
        doc["observations"].sort(key=lambda o: o["roi_id"])
        return doc

    def handle_msg(self, msg) -> dict | None:
        """Throttle, convert, classify, publish and log one Image message."""
        self.frames_seen += 1
        now = self.clock()
        if self._last is not None and now - self._last < self.min_period_s:
            return None
        self._last = now
        stamp = {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)}
        frame_id = str(msg.header.frame_id)
        try:
            image = image_msg_to_pil(msg)
            doc = self.process_image(image, stamp=stamp, frame_id=frame_id)
        except (ValueError, OSError) as exc:
            doc = {
                "schema": OBSERVATION_SCHEMA,
                "stamp": stamp,
                "frame_id": frame_id,
                "source_topic": self.source_topic,
                "crop_mode": self.mode,
                "error": str(exc),
                "observations": [],
                "actuation": "none",
                "ground_truth_used": False,
            }
        text = json.dumps(doc, sort_keys=True, ensure_ascii=False)
        self.publish(text)
        self.log.write(doc)
        self.frames_processed += 1
        return doc


def run_ros2_observe(args, ros_args: list[str] | None, classifier, gate: GateConfig, roi_config) -> int:
    """Spin the ROS 2 node until shutdown. Requires a sourced ROS 2 environment."""
    hint = "Source a ROS 2 Jazzy environment (e.g. /opt/ros/jazzy/setup.bash) first."
    rclpy = require("rclpy", "the ROS 2 adapter", hint)
    node_mod = require("rclpy.node", "the ROS 2 adapter", hint)
    qos_mod = require("rclpy.qos", "the ROS 2 adapter", hint)
    sensor_msgs = require("sensor_msgs.msg", "the ROS 2 adapter", hint)
    std_msgs = require("std_msgs.msg", "the ROS 2 adapter", hint)

    rclpy.init(args=["vineyard_leaf_observer", *(ros_args or [])])
    node = node_mod.Node("vineyard_leaf_observer")
    publisher = node.create_publisher(std_msgs.String, args.observation_topic, 10)

    def publish(text: str) -> None:
        msg = std_msgs.String()
        msg.data = text
        publisher.publish(msg)

    pipeline = ObservationPipeline(
        classifier,
        gate,
        roi_config=roi_config,
        full_frame_single_leaf=args.full_frame_single_leaf_only,
        source_topic=args.image_topic,
        publish=publish,
        log=NdjsonLog(args.log_ndjson),
        min_period_s=args.min_period_s,
    )

    def callback(msg) -> None:
        try:
            pipeline.handle_msg(msg)
        except Exception as exc:  # noqa: BLE001 - keep the node alive; record the failure
            node.get_logger().error(f"leaf observation failed: {exc}")
            pipeline.log.write({"schema": OBSERVATION_SCHEMA, "error": f"{type(exc).__name__}: {exc}"})

    node.create_subscription(sensor_msgs.Image, args.image_topic, callback, qos_mod.qos_profile_sensor_data)
    node.get_logger().info(f"observing {args.image_topic} -> {args.observation_topic} ({pipeline.mode}); no actuation")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0
