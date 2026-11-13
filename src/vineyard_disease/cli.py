"""Command line interface: ``python -m vineyard_disease <command>``.

Commands: ``prepare``, ``predict``, ``evaluate``, ``ros2-observe``, ``labels``.
Heavy dependencies are imported inside the command handlers only.

Exit codes: 0 ok, 1 other error, 2 invalid input/usage, 3 data or model
source unavailable (nothing synthesized), 4 optional dependency missing,
5 model label space does not cover the grape classes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .errors import InputError, PipelineError, require

DEFAULT_HF_OUTPUT = "data/plantvillage_grape_color"
DEFAULT_LOCAL_OUTPUT = "data/local_grape"
DEFAULT_CACHE = "data/cache/huggingface"
DEFAULT_ROS_LOG = "data/logs/leaf_observations.ndjson"


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from exc
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


def _unit_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from exc
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError("must be within [0, 1]")
    return v


def _nonneg_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from exc
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v


def _add_model_args(p: argparse.ArgumentParser) -> None:
    from .classifier import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION

    g = p.add_argument_group("model")
    g.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id (default: %(default)s)")
    g.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
        help="model revision/commit (default: pinned %(default)s; '' = latest)",
    )
    g.add_argument("--device", default="cpu", help="torch device: cpu (default) or cuda[:N]")
    g.add_argument("--num-threads", type=_positive_int, default=None, help="torch CPU threads")
    g.add_argument("--local-files-only", action="store_true", help="do not download; use the local HF cache")
    g.add_argument(
        "--min-score",
        type=_unit_float,
        default=0.5,
        help="minimum top-1 softmax probability to accept a grape class (uncalibrated; default 0.5)",
    )
    g.add_argument(
        "--min-grape-mass",
        type=_unit_float,
        default=0.0,
        help="minimum summed probability of the 4 grape labels (default 0.0)",
    )
    g.add_argument("--top-k", type=_positive_int, default=5, help="labels listed in top_k (default 5)")
    g.add_argument("--batch-size", type=_positive_int, default=8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vineyard-disease",
        description="CPU-first grape-leaf disease screening on single leaf crops. "
        "Scores are uncalibrated and are not a diagnosis.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # prepare -----------------------------------------------------------
    p = sub.add_parser("prepare", help="build a bounded manifest from PlantVillage (HF) or local class dirs")
    p.add_argument(
        "--source",
        choices=("hf", "local"),
        default="hf",
        help="hf: official mohanty/PlantVillage color config (pinned); local: class directories",
    )
    p.add_argument(
        "--output-dir", default=None, help=f"output directory (default {DEFAULT_HF_OUTPUT} or {DEFAULT_LOCAL_OUTPUT})"
    )
    lim = p.add_mutually_exclusive_group()
    lim.add_argument(
        "--max-per-class", type=_positive_int, default=25, help="max images per class per split (default 25)"
    )
    lim.add_argument("--all", action="store_true", help="take every grape image (no bound)")
    p.add_argument("--seed", type=int, default=0, help="selection seed (default 0)")
    p.add_argument("--splits", default=None, help="comma list of splits to keep (hf default: train,test)")
    p.add_argument("--min-side", type=_positive_int, default=32, help="reject images smaller than this (px)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing manifest")
    hf = p.add_argument_group("hf source")
    hf.add_argument("--hf-revision", default=None, help="dataset revision (default: pinned commit)")
    hf.add_argument("--hf-cache-dir", default=DEFAULT_CACHE, help="metadata cache (default %(default)s)")
    hf.add_argument(
        "--hf-meta-dir", default=None, help="offline: directory with splits/color_*.txt and leaf_grouping/leaf-map.json"
    )
    hf.add_argument("--hf-zip", default=None, help="offline: local copy of the dataset's data.zip")
    hf.add_argument("--no-verify-zip", action="store_true", help="skip sha256 of a local --hf-zip")
    hf.add_argument(
        "--max-download-mb", type=_positive_int, default=512, help="abort if more than this is downloaded (default 512)"
    )
    hf.add_argument("--timeout", type=_nonneg_float, default=60.0, help="HTTP timeout seconds")
    lo = p.add_argument_group("local source")
    lo.add_argument("--input-dir", default=None, help="<dir>/<class>/* or <dir>/<split>/<class>/*")
    lo.add_argument("--groups-file", default=None, help="CSV with columns path,group_id")
    lo.add_argument(
        "--assign-split",
        choices=("none", "hash"),
        default="none",
        help="assign train/test by hash when the input has no split dirs (default none)",
    )
    lo.add_argument("--test-fraction", type=float, default=0.2)
    lo.add_argument("--strict-labels", action="store_true", help="fail on unrecognized class directories")

    # predict -----------------------------------------------------------
    p = sub.add_parser("predict", help="classify single-leaf crop images")
    p.add_argument("inputs", nargs="*", help="image files or directories")
    p.add_argument("--manifest", default=None, help="read image paths from a manifest (labels are not used)")
    p.add_argument("--split", default=None, help="manifest split filter")
    p.add_argument("--recursive", action="store_true", help="recurse into input directories")
    p.add_argument("--roi", default=None, help="crop box x,y,width,height (pixels) or ...@norm; default full image")
    p.add_argument("--limit", type=_positive_int, default=None)
    p.add_argument("--output", "-o", default="-", help="predictions JSONL path (default stdout)")
    p.add_argument("--no-all-scores", action="store_true", help="omit the full per-label score map")
    _add_model_args(p)

    # evaluate ----------------------------------------------------------
    p = sub.add_parser("evaluate", help="metrics per grape class against manifest labels")
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", default="test", help="split to evaluate (default test; 'all' = every split)")
    p.add_argument("--predictions", default=None, help="existing predictions JSONL (skip inference)")
    p.add_argument("--output-dir", default=None, help="default <manifest dir>/eval-<split>")
    p.add_argument("--limit", type=_positive_int, default=None)
    _add_model_args(p)

    # ros2-observe ------------------------------------------------------
    p = sub.add_parser("ros2-observe", help="ROS 2 node: Image topic -> JSON leaf observations (no actuation)")
    p.add_argument("--image-topic", default="/camera/image_raw")
    p.add_argument("--observation-topic", default="/vineyard/leaf_observations")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--roi-config", default=None, help="JSON/YAML file with known leaf ROIs")
    mode.add_argument(
        "--full-frame-single-leaf-only",
        action="store_true",
        help="use the whole frame; asserts every frame is a single leaf crop",
    )
    p.add_argument("--log-ndjson", default=DEFAULT_ROS_LOG, help="NDJSON log (default %(default)s; '' disables)")
    p.add_argument("--min-period-s", type=_nonneg_float, default=1.0, help="min seconds between processed frames")
    _add_model_args(p)

    # labels ------------------------------------------------------------
    sub.add_parser("labels", help="print canonical classes and label mappings as JSON")
    return parser


def _gate(args):
    from .gating import GateConfig

    return GateConfig(min_score=args.min_score, min_grape_mass=args.min_grape_mass, top_k=args.top_k)


def _cmd_prepare(args, argv) -> int:
    from .prepare import run_prepare
    from .quality import QualityConfig

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip()) if args.splits else None
    if args.source == "hf":
        if args.input_dir or args.groups_file:
            raise InputError("--input-dir/--groups-file apply to --source local only")
        from .sources.http import ByteBudget, HttpClient
        from .sources.plantvillage_hf import HF_REVISION, SOURCE_SPLITS, PlantVillageHF

        client = HttpClient(timeout=args.timeout, budget=ByteBudget(args.max_download_mb * 1024 * 1024))
        source = PlantVillageHF(
            client,
            cache_dir=args.hf_cache_dir,
            revision=args.hf_revision or HF_REVISION,
            meta_dir=args.hf_meta_dir,
            zip_path=args.hf_zip,
            verify_zip=not args.no_verify_zip,
            splits=splits or SOURCE_SPLITS,
        )
        out = args.output_dir or DEFAULT_HF_OUTPUT
    else:
        if not args.input_dir:
            raise InputError("--source local needs --input-dir")
        from .sources.local_dir import LocalClassDirs

        source = LocalClassDirs(
            args.input_dir,
            groups_file=args.groups_file,
            strict=args.strict_labels,
            assign_split=args.assign_split,
            test_fraction=args.test_fraction,
            seed=args.seed,
            splits=splits,
        )
        out = args.output_dir or DEFAULT_LOCAL_OUTPUT
    summary = run_prepare(
        source,
        out,
        max_per_class=None if args.all else args.max_per_class,
        seed=args.seed,
        quality=QualityConfig(min_side=args.min_side),
        overwrite=args.overwrite,
        command=" ".join(["prepare", *_public_argv(argv)]),
    )
    if args.source == "hf":
        summary["downloaded_bytes"] = client.budget.used
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _public_argv(argv: Sequence[str]) -> list[str]:
    """Drop absolute paths from the recorded command line."""
    return ["<path>" if a.startswith("/") else a for a in argv[1:]]


def _cmd_predict(args) -> int:
    from .classifier import load_classifier
    from .predict import collect_inputs, open_output, predict_iter, write_jsonl
    from .roi import parse_roi_string

    roi = parse_roi_string(args.roi) if args.roi else None
    inputs = collect_inputs(
        args.inputs, manifest=args.manifest, split=args.split, recursive=args.recursive, limit=args.limit
    )
    classifier = load_classifier(args)
    counts: dict[str, int] = {}
    with open_output(args.output) as fh:

        def rows():
            for row in predict_iter(
                inputs, classifier, _gate(args), roi=roi, batch_size=args.batch_size, all_scores=not args.no_all_scores
            ):
                counts[row["status"]] = counts.get(row["status"], 0) + 1
                yield row

        n = write_jsonl(rows(), fh)
    print(json.dumps({"predictions": n, "status_counts": counts, "output": args.output}), file=sys.stderr)
    return 0


def _cmd_evaluate(args) -> int:
    from .evaluate import run_evaluate

    split = None if args.split == "all" else args.split
    out = args.output_dir or str(Path(args.manifest).parent / f"eval-{args.split}")
    classifier = None
    if not args.predictions:
        from .classifier import load_classifier

        classifier = load_classifier(args)
    report = run_evaluate(
        args.manifest,
        out,
        split=split,
        predictions=args.predictions,
        classifier=classifier,
        gate=_gate(args),
        batch_size=args.batch_size,
        limit=args.limit,
    )
    m = report["metrics"]
    summary = {
        "metrics": str(Path(out) / "metrics.json"),
        "predictions": report["predictions"],
        "n": m["n"],
        "coverage": m["coverage"],
        "accuracy": m["accuracy"],
        "selective_accuracy": m["selective_accuracy"],
        "macro_f1": m["macro_f1"],
        "per_class_f1": {c: v["f1"] for c, v in m["per_class"].items()},
        "status_counts": report["status_counts"],
        "claims": report["claims"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_ros2(args, ros_args) -> int:
    from .classifier import load_classifier
    from .roi import load_roi_config
    from .ros2_node import run_ros2_observe

    require("rclpy", "the ROS 2 adapter", "Source a ROS 2 Jazzy environment first.")
    roi_config = load_roi_config(args.roi_config) if args.roi_config else None
    classifier = load_classifier(args)
    return run_ros2_observe(args, ros_args, classifier, _gate(args), roi_config)


def _cmd_labels() -> int:
    from .classifier import DEFAULT_MODEL_ID
    from .labels import CANONICAL_CLASSES, DEFAULT_MODEL_GRAPE_LABELS, PLANTVILLAGE_GRAPE_CLASSES

    print(
        json.dumps(
            {
                "canonical_classes": list(CANONICAL_CLASSES),
                "plantvillage": dict(PLANTVILLAGE_GRAPE_CLASSES),
                "model": {"id": DEFAULT_MODEL_ID, "grape_labels": dict(DEFAULT_MODEL_GRAPE_LABELS)},
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ros_args: list[str] = []
    if "--ros-args" in argv:  # pass ROS remapping/params through to rclpy
        i = argv.index("--ros-args")
        argv, ros_args = argv[:i], argv[i:]
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            return _cmd_prepare(args, ["vineyard-disease", *argv])
        if args.command == "predict":
            return _cmd_predict(args)
        if args.command == "evaluate":
            return _cmd_evaluate(args)
        if args.command == "ros2-observe":
            return _cmd_ros2(args, ros_args)
        if args.command == "labels":
            return _cmd_labels()
        parser.error(f"unknown command {args.command}")  # pragma: no cover
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
