"""CLI contract: argparse wiring, exit codes, lazy imports, offline commands."""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import pytest

from vineyard_disease import cli

SRC = str(Path(__file__).resolve().parents[1] / "src")


def run_cli(argv, capsys):
    code = cli.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_import_and_help_do_not_load_heavy_deps():
    code = (
        "import sys; from vineyard_disease import cli; p = cli.build_parser(); "
        "[p.parse_args(a) for a in (['labels'], ['predict', 'x.png'], ['evaluate', '--manifest', 'm'], "
        "['ros2-observe', '--full-frame-single-leaf-only'], ['prepare'])]; "
        "heavy = [m for m in ('torch', 'transformers', 'PIL', 'rclpy', 'datasets') if m in sys.modules]; "
        "print(heavy)"
    )
    env = dict(os.environ, PYTHONPATH=SRC)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert res.stdout.strip() == "[]"


def test_module_entrypoint_help():
    env = dict(os.environ, PYTHONPATH=SRC)
    res = subprocess.run(
        [sys.executable, "-m", "vineyard_disease", "--help"], capture_output=True, text=True, env=env, check=False
    )
    assert res.returncode == 0
    for cmd in ("prepare", "predict", "evaluate", "ros2-observe", "labels"):
        assert cmd in res.stdout


def test_labels_command(capsys):
    code, out, _ = run_cli(["labels"], capsys)
    data = json.loads(out)
    assert code == 0 and data["canonical_classes"] == ["black_rot", "esca", "isariopsis_leaf_spot", "healthy"]
    assert data["plantvillage"]["Grape___Esca_(Black_Measles)"] == "esca"


@pytest.mark.parametrize(
    "argv",
    [
        ["prepare", "--max-per-class", "0"],
        ["prepare", "--max-per-class", "3", "--all"],
        ["prepare", "--source", "kaggle"],
        ["predict", "x.png", "--min-score", "2"],
        ["ros2-observe"],  # mode is required
        ["ros2-observe", "--roi-config", "r.json", "--full-frame-single-leaf-only"],
        [],
    ],
)
def test_usage_errors_exit_2(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_parser_defaults():
    p = cli.build_parser()
    a = p.parse_args(["predict", "img.png"])
    assert a.device == "cpu" and a.min_score == 0.5 and a.model_revision
    assert a.model_id == "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    a = p.parse_args(["prepare"])
    assert a.source == "hf" and a.max_per_class == 25 and a.seed == 0 and a.max_download_mb == 512
    a = p.parse_args(["ros2-observe", "--roi-config", "r.json"])
    assert a.image_topic == "/camera/image_raw" and a.observation_topic == "/vineyard/leaf_observations"
    assert a.log_ndjson.startswith("data/")


def test_ros_args_are_split_off(monkeypatch):
    seen = {}

    def fake(args, ros_args):
        seen["ros_args"] = ros_args
        return 0

    monkeypatch.setattr(cli, "_cmd_ros2", fake)
    assert cli.main(["ros2-observe", "--full-frame-single-leaf-only", "--ros-args", "-r", "a:=b"]) == 0
    assert seen["ros_args"] == ["--ros-args", "-r", "a:=b"]


def test_ros2_observe_without_rclpy_exits_4(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "rclpy", None)  # import rclpy -> ImportError
    code, _, err = run_cli(["ros2-observe", "--full-frame-single-leaf-only"], capsys)
    assert code == 4 and "rclpy" in err


def test_predict_without_ml_deps_exits_4(monkeypatch, capsys, tmp_path):
    pytest.importorskip("PIL")
    from test_pipeline_support import png_bytes

    img = tmp_path / "a.png"
    img.write_bytes(png_bytes())
    monkeypatch.setitem(sys.modules, "torch", None)
    code, _, err = run_cli(["predict", str(img)], capsys)
    assert code == 4 and "torch" in err


def test_local_prepare_predict_evaluate_roundtrip(monkeypatch, capsys, tmp_path):
    pytest.importorskip("PIL")
    from test_pipeline_support import MockClassifier, png_bytes

    root = tmp_path / "in"
    for split in ("train", "test"):
        for cls, color in (("black_rot", (200, 0, 0)), ("healthy", (0, 200, 0))):
            (root / split / cls).mkdir(parents=True)
            for i in range(3):
                (root / split / cls / f"{i}.png").write_bytes(png_bytes(color))
    out = tmp_path / "data" / "local"
    code, stdout, _ = run_cli(
        ["prepare", "--source", "local", "--input-dir", str(root), "--output-dir", str(out), "--max-per-class", "2"],
        capsys,
    )
    assert code == 0 and json.loads(stdout)["records"] == 8
    prov = json.loads((out / "provenance.json").read_text())
    assert "<path>" in prov["tool"]["command"] and str(tmp_path) not in json.dumps(prov)

    import vineyard_disease.classifier as clf_mod

    monkeypatch.setattr(clf_mod, "load_classifier", lambda args: MockClassifier())
    preds = tmp_path / "preds.jsonl"
    code, _, err = run_cli(
        ["predict", "--manifest", str(out / "manifest.jsonl"), "--split", "test", "-o", str(preds)], capsys
    )
    assert code == 0 and json.loads(err)["predictions"] == 4
    rows = [json.loads(line) for line in preds.read_text().splitlines()]
    assert {r["predicted_class"] for r in rows} == {"black_rot", "healthy"}
    assert all("label" not in r for r in rows)  # ground truth never copied into predictions

    code, stdout, _ = run_cli(
        ["evaluate", "--manifest", str(out / "manifest.jsonl"), "--predictions", str(preds)], capsys
    )
    summary = json.loads(stdout)
    assert code == 0 and summary["n"] == 4 and summary["accuracy"] == 1.0
    assert summary["per_class_f1"]["esca"] is None  # no support -> undefined
    assert (out / "eval-test" / "metrics.json").is_file()

    code, stdout, _ = run_cli(
        ["evaluate", "--manifest", str(out / "manifest.jsonl"), "--split", "all", "--output-dir", str(tmp_path / "e2")],
        capsys,
    )
    assert code == 0 and json.loads(stdout)["n"] == 8


def test_prepare_hf_network_failure_exits_3(monkeypatch, capsys, tmp_path):
    from vineyard_disease.sources import http

    def down(url, headers, timeout):
        raise OSError("network unreachable")

    monkeypatch.setattr(http, "urllib_transport", down)
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    out = tmp_path / "data" / "pv"
    code, _, err = run_cli(["prepare", "--output-dir", str(out), "--hf-cache-dir", str(tmp_path / "c")], capsys)
    assert code == 3 and "No data was synthesized" in err
    assert not (out / "manifest.jsonl").exists()


def test_prepare_argument_conflicts(capsys, tmp_path):
    code, _, err = run_cli(["prepare", "--input-dir", str(tmp_path)], capsys)
    assert code == 2 and "--source local" in err
    code, _, err = run_cli(["prepare", "--source", "local"], capsys)
    assert code == 2 and "--input-dir" in err


def test_evaluate_bad_manifest_exits_2(capsys, tmp_path):
    code, _, err = run_cli(["evaluate", "--manifest", str(tmp_path / "none.jsonl"), "--predictions", "p"], capsys)
    assert code == 2 and "not found" in err
