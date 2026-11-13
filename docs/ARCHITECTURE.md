# Architecture

```text
Official PlantVillage grape images (pinned source revision)
  -> bounded acquisition + provenance + source splits / leaf groups
  -> manifest JSONL and local images (ignored by Git)
  -> CPU MobileNetV2 inference
  -> predictions JSONL + per-class evaluation

Procedural vineyard + stylized or prepared photo textures
  -> Gazebo Harmonic RGB / instance-segmentation cameras
  -> ROS 2 RGB bridge
  -> single-leaf crop inference / observation logger
  -> capture files + replayable observations (ignored by Git)
```

## Separation of concerns

The scene generator creates original low-poly vineyard geometry, trellis posts/wires, irrigation lines/emitters, trunks, two-sided leaf meshes and a static camera/rover fixture. The seed fixes placements and stylized textures. `--view wide` shows the vineyard; `--view leaf` places the camera at a known single-leaf fixture. This placement is an integration aid, not a learned leaf detector.

Generated SDF references `assets/` relative to its resource root. The smoke runner configures `GZ_SIM_RESOURCE_PATH` from the world directory; do the same for custom Gazebo launch commands. Keep the generated SDF and its assets directory together. Generated asset paths do not assume a specific developer machine.

`ground_truth.json` is for fixture auditing/evaluation. The RGB pipeline does not consume its disease class labels. Gazebo instance/semantic object masks also stay on the evaluator side. A leaf mask cannot be used as a lesion mask without separate lesion annotation or rendering support.

The Python pipeline is independently runnable without Gazebo. Loading model dependencies is lazy: acquisition, manifest checks and unit tests can run without torch/ROS. The ROS observer is optional and accepts image input only; it does not subscribe to simulation disease labels or publish treatment commands.

## Runtime

The recommended reproducible environment is an Ubuntu 24.04 ROS Jazzy container with Gazebo Harmonic. The source image is digest-pinned and ML top-level versions are pinned in the Dockerfile. NumPy is kept on the 1.26 ABI to match Ubuntu's ROS/cv_bridge/scipy environment. No GPU is required for classifier inference. Software rendering is requested for headless Gazebo; rendering performance and graphics-driver availability remain separate from inference requirements.

`scripts/run-container.sh` uses an unprivileged user, a private container network, no host devices, 4 CPUs, 6 GiB memory and a process limit. The mounted checkout is the only writable project volume. It does not alter host ROS, system packages, Docker configuration, or other containers. Builds still consume local image/cache disk space; do not use broad Docker prune commands on a shared host.

The camera smoke runner waits on ROS image events, captures nonuniform frames, records timestamps and saves logs. It terminates only the Gazebo/bridge processes it started and exits nonzero on timeout/early exit. It is not a navigation, disease-fidelity, model-accuracy or hardware test.

## Next modules

- Fine-tune a small leaf detector on field bounding boxes; stop treating a wide frame as a single leaf crop.
- Add expert-labeled local disease classes, hard negatives and independent field evaluation.
- Add lesion segmentation only with validated lesion masks.
- Add camera trajectories / navigation and vine association.
- Add growth-stage observations and GDD as a separate context pipeline.
- Add reviewed inspection/protection recommendations only after diagnosis, operational context and safety review are available.
