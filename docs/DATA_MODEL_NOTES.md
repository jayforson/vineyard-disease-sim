# Data, model and simulation boundaries

## Sources

- PlantVillage: https://huggingface.co/datasets/mohanty/PlantVillage and https://github.com/spMohanty/PlantVillage-Dataset . The official Hugging Face card declares `cc-by-sa-3.0`; retain that source/version information in generated manifests. Do not substitute the license claims of a third-party repackaging.
- Baseline model: https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification . It is a 38-class MobileNetV2 checkpoint fine-tuned on a PlantVillage-derived dataset. The card's evaluation is not an independent vineyard field benchmark. Model-weight licensing is separate from this repository's MIT source-code license; do not infer one from the other.
- Gazebo segmentation camera: https://gazebosim.org/api/sensors/8/segmentationcamera_igngazebo.html . Object/instance labels are evaluator ground truth, not automatically disease-lesion masks.

Downloaded images, model weights, caches, manifests, generated photo textures, and run outputs are excluded from Git. This repository publishes original source code and instructions, not a dataset or model redistribution.

For `--manifest` scenes, each photo-textured asset retains its dataset URL, source split/group, image hash, source-license label and citation from the prepared provenance. Default photo selection is the source training split; use `--source-split test` only for a separately held-out scene. These metadata describe the source, not permission to redistribute assets. Keep generated photo-texture directories out of public commits and retain their source attribution when moving them.

## What is implemented vs what is not established

The baseline can screen cropped grape-leaf images for four supported categories: `healthy`, `black_rot`, `esca`, and `isariopsis_leaf_spot`. It does not include grape powdery mildew, downy mildew, every nutrient deficiency, or every pest. Do not treat a supported-class prediction as evidence that those other conditions are absent.

Scores are uncalibrated classifier scores, not diagnostic certainty. Threshold/margin and non-grape gates provide conservative handling of some cases; they cannot reliably discover all unseen diseases. A separate out-of-distribution study and field calibration are still required.

The Gazebo fixture tests whether the scene, RGB camera, ROS interface and inference path operate together. Default symptom patterns are intentionally stylized synthetic fixtures. Photo-textured leaves are still rendered approximations: viewing them does not establish biological realism. Sim labels must never be passed into classifier inputs or presented as independent disease confirmation.

The prototype has a static camera/rover fixture and a close-up single-leaf view. It does not yet include a trained leaf detector, autonomous navigation, leaf-instance tracking, lesion-severity estimation, growth-stage recognition, water-balance modeling or a treatment controller. Those are future modules, not implied by the screenshot or smoke test.

## Evaluation rules

- Preserve official source train/test and leaf groups where supplied. For local folders, retain grouping provenance; a filename or hash group is not proof that distinct photos show distinct biological leaves.
- Keep source textures and leaf families disjoint between synthetic train/test sets. Random camera frames from the same textured leaf are not independent examples.
- Split field evaluation by vine/vineyard and date where feasible. Report false positives, per-class recall, abstention rate and missing-class coverage.
- Do not cite a tiny bounded sample's score as model accuracy, or simulation integration results as field disease accuracy.
- Neither a leaf image nor classifier confidence provides a calibrated lesion-area percentage, vine phenological stage, root-zone moisture, or a treatment prescription.

No field devices, valves or sprayers are controlled by this code.
