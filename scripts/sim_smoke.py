#!/usr/bin/env python3
"""Headless, event-driven Gazebo camera smoke test; no network or hardware writes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, default=Path("simulation/generated/vineyard.sdf"))
    parser.add_argument("--output", type=Path, default=Path("runs/sim-smoke"))
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument(
        "--observe", action="store_true", help="Also require an actual ROS disease observation from the CPU model"
    )
    parser.add_argument("--roi-config", type=Path, default=Path("simulation/closeup-roi.json"))
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model weights")
    args = parser.parse_args()
    if args.timeout <= 0 or args.frames < 1:
        parser.error("timeout and frames must be positive")
    if not args.world.is_file():
        parser.error("world missing; run simulation/generate_world.py first")
    import numpy as np
    import rclpy
    from PIL import Image as PILImage
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    args.output.mkdir(parents=True, exist_ok=True)
    processes, logs = [], []
    rclpy.init()
    node = Node("vineyard_camera_smoke")
    observed = []
    observations = []
    last_image = None

    def receive_observation(message):
        result = json.loads(message.data)
        if result.get("actuation") != "none" or result.get("ground_truth_used") is not False:
            raise RuntimeError("Observer violated observation-only / no-ground-truth contract")
        if result.get("error") or not result.get("observations"):
            raise RuntimeError(f"Observer returned no valid inference: {result.get('error')}")
        if any(item.get("status") not in ("accepted", "non_grape", "low_score") for item in result["observations"]):
            raise RuntimeError("Observer returned an inference/ROI error")
        observations.append(result)

    def receive(message):
        nonlocal last_image
        if message.encoding not in ("rgb8", "bgr8", "rgba8", "bgra8"):
            return
        channels = 4 if "a8" in message.encoding else 3
        raw = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(message.height, message.step)
        rgb = raw[:, : message.width * channels].reshape(message.height, message.width, channels)[:, :, :3]
        if message.encoding.startswith("bgr"):
            rgb = rgb[:, :, ::-1]
        last_image = PILImage.fromarray(rgb.copy())
        observed.append(
            {
                "width": message.width,
                "height": message.height,
                "encoding": message.encoding,
                "stamp_sec": message.header.stamp.sec,
                "stamp_nanosec": message.header.stamp.nanosec,
            }
        )

    node.create_subscription(Image, "/vineyard/camera/rgb", receive, qos_profile_sensor_data)
    if args.observe:
        node.create_subscription(String, "/vineyard/leaf_observations", receive_observation, 10)
    began = time.monotonic()
    failure = None
    try:
        env = dict(os.environ)
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        env["GZ_SIM_RESOURCE_PATH"] = (
            str(args.world.resolve().parent) + os.pathsep + env.get("GZ_SIM_RESOURCE_PATH", "")
        )
        commands = [
            ("gazebo.log", ["gz", "sim", "-v", "4", "-s", "-r", "--headless-rendering", str(args.world.resolve())]),
            (
                "bridge.log",
                [
                    "ros2",
                    "run",
                    "ros_gz_bridge",
                    "parameter_bridge",
                    "/vineyard/camera/rgb@sensor_msgs/msg/Image[gz.msgs.Image",
                ],
            ),
        ]
        if args.observe:
            if not args.roi_config.is_file():
                raise FileNotFoundError(args.roi_config)
            observer = [
                sys.executable,
                "-m",
                "vineyard_disease",
                "ros2-observe",
                "--image-topic",
                "/vineyard/camera/rgb",
                "--roi-config",
                str(args.roi_config),
                "--log-ndjson",
                str(args.output / "observations.ndjson"),
                "--num-threads",
                "2",
            ]
            if args.local_files_only:
                observer.append("--local-files-only")
            commands.append(("observer.log", observer))
        for filename, command in commands:
            log = (args.output / filename).open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            )
        # Wait on camera events, not a shell sleep/re-read loop.
        while len(observed) < args.frames or (args.observe and not observations):
            remaining = args.timeout - (time.monotonic() - began)
            if remaining <= 0:
                raise TimeoutError(f"No sufficient camera frames within {args.timeout}s; see logs")
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"Simulation/bridge exited early with {process.returncode}; see logs")
            rclpy.spin_once(node, timeout_sec=min(remaining, 0.5))
        last_image.save(args.output / "camera.png")
        if np.asarray(last_image).std() < 2:
            raise RuntimeError("Camera image is effectively uniform; rendered scene not verified")
    except Exception as error:  # noqa: BLE001 - smoke must record runtime failures and clean up owned processes
        failure = str(error)
        try:
            topics = subprocess.run(
                ["gz", "topic", "-l"], check=False, capture_output=True, text=True, timeout=5, env=env
            )
            (args.output / "topics.log").write_text(topics.stdout + topics.stderr)
        except (OSError, subprocess.SubprocessError) as diagnostic_error:
            (args.output / "topics.log").write_text(f"Topic diagnostics unavailable: {diagnostic_error}\n")
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        for log in logs:
            log.close()
        node.destroy_node()
        rclpy.shutdown()
    result = {
        "ok": failure is None,
        "frames": observed,
        "perception_enabled": args.observe,
        "observations": observations,
        "elapsed_seconds": round(time.monotonic() - began, 2),
        "failure": failure,
        "rendering": "software requested; inspect renderer log for actual backend",
        "not_proved": ["field disease accuracy", "trained leaf detection", "navigation", "disease biology"],
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
