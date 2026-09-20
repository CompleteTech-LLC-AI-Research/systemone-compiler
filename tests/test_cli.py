import json
import os
from pathlib import Path
import subprocess
import sys
from s1compiler.cli import main
from s1compiler.hardening import propose_cases


def test_cli_demo_complete(tmp_path):
    destination = tmp_path / "demo"
    assert main(["demo", "--out", str(destination)]) == 0
    report = json.loads((destination / "artifacts/report.json").read_text())
    assert report["status"] == "synthetic"
    assert (destination / "artifacts/program.s1.json").exists()
    assert (destination / "sample_prediction.json").exists()
    assert main(["inspect", str(destination / "artifacts/program.s1.json")]) == 0


def test_cli_init_does_not_overwrite(tmp_path):
    destination = tmp_path / "project"
    assert main(["init", str(destination)]) == 0
    assert main(["init", str(destination)]) == 2


def test_cli_draft_run_export_schema(tmp_path):
    project = tmp_path / "project"
    main(["init", str(project)])
    artifact = tmp_path / "program.json"
    assert main(["draft", str(project / "usecase.yaml"), "--out", str(artifact)]) == 0
    assert main(["run", str(artifact), "--state", str(project / "sample_state.json"), "--backend", "mock"]) == 0
    exported = tmp_path / "playground.json"
    assert main(["export-playground", str(artifact), "--state", str(project / "sample_state.json"), "--out", str(exported)]) == 0
    assert set(json.loads(exported.read_text())) == {"state", "model", "questions"}
    assert main(["schema", "--out", str(tmp_path / "schemas")]) == 0


def test_cli_compile(tmp_path):
    project = tmp_path / "project"
    main(["init", str(project)])
    args = ["compile", str(project / "usecase.yaml"), "--out", str(tmp_path / "compiled")]
    for name in ("train", "validation", "calibration", "test"):
        args += [f"--{name}", str(project / f"{name}.jsonl")]
    assert main(args) == 0
    assert (tmp_path / "compiled/report.md").exists()


def test_cli_doctor_no_network():
    assert main(["doctor"]) == 0


def test_hardening_does_not_create_fake_gold(source, splits):
    result = propose_cases(source, splits["train"])
    assert result["benchmark_ready"] is False
    assert all(r["requires_label_review"] and "expected" not in r for r in result["proposals"])


def test_runtime_import_has_no_optimizer_dependencies():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    code = "import s1compiler,sys; assert 'dspy' not in sys.modules; assert 'gepa' not in sys.modules; assert 'typesafe_sdk' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_yaml_parser_error_is_terse(tmp_path, capsys):
    from s1compiler.cli import main
    source = tmp_path / "bad.yaml"
    source.write_text('foo: [private-sensitive-value\n')
    assert main(["draft", str(source), "--out", str(tmp_path / "output.json")]) == 2
    message = capsys.readouterr().err
    assert "private-sensitive-value" not in message
    assert "Invalid or unsupported YAML" in message
