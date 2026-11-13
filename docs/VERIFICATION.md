# Verification — initial prototype

Verified on 13 November 2026 UTC with the pinned Docker environment. These are software/integration checks, not independent disease-accuracy evidence.

## Checks completed

| Check | Result |
|---|---|
| Python package installation and `vineyard-disease --version` | Editable install succeeds; version 0.1.0 |
| Unit/regression suite | 133 tests pass, including real local HTTP-server redirect tests, archive/image size guards, source splits, score gates, ROI conversion and world provenance |
| Ruff check and format | Clean |
| Container dependency consistency | `pip check` reports no broken requirements |
| Simulator | Gazebo Sim 8.15.0 (Harmonic), ROS Jazzy |
| ML environment | PyTorch 2.7.1+cpu, Transformers 4.57.6, NumPy 1.26.4, Pillow 11.3.0 |
| Default stylized scene | 3 rows × 6 vines; textured leaf meshes render; 640×480 RGB frames arrive through the ROS bridge |
| Bounded actual PlantVillage ingest | 96 images: 12 per grape class per official train/test split; deterministic manifest; source revision, image hashes, grouping caveats and source license recorded |
| Actual checkpoint inference | CPU inference and per-class evaluation run on 48 source-test images; model/source overlap means this is an in-distribution pipeline check only |
| Live Gazebo → ROS → model path | Photo-textured close-up fixture produces RGB frames and a model observation; `device: cpu`, `ground_truth_used: false`, `actuation: none` |

## Important result: rendered-image domain gap

The first live photo-textured fixture produced `status: non_grape`, rather than an accepted grape-disease category. The pipeline correctly retained the raw 38-class model scores and abstained. This proves the integration path and conservative output handling, **not reliable disease recognition on rendered leaves**. Do not advertise synthetic/field diagnostic accuracy from this result.

The real-image evaluation is also not an independent benchmark: the baseline model was trained on a PlantVillage-derived dataset, and its relation to the source's official test images cannot be assumed disjoint. Its accepted/abstained counts are recorded in local run outputs, not offered as field-performance claims.

The next perception work is to add annotated field leaf detection, improve rendering/camera appearance, and fine-tune/validate with appropriately split real and synthetic images. No new model training has been performed in this prototype.

## Hardening and publication boundary

- Public dataset acquisition sends no ambient `HF_TOKEN`. Cross-origin redirects strip sensitive headers; regression tests fail under urllib's unsafe default redirect behavior.
- Archive/local image entries are limited to 20 MB; Pillow decompression bombs become input errors. Empty preparation cannot silently replace valid previous outputs.
- Downloaded images, weights, generated photo textures, logs and coverage caches are not included in Git. The README screenshot is the original stylized synthetic scene, not PlantVillage imagery.
- Model and source revisions are pinned. Partial range acquisition records that the entire archive hash has not been recomputed; do not claim otherwise.
- The demo has a static camera/rover fixture and explicit known crop. It does not implement a trained leaf detector, navigation, growth stages, severity or treatment control.

## Reproduce

See the README for container build, bounded data preparation and the live `scripts/sim_smoke.py --observe` command. Local evidence goes under ignored `runs/` and `data/`, so the repository contains instructions and this result summary rather than redistributed data or machine-specific logs.
