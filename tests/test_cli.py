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


def test_hierarchy_cli_demo_and_format_dispatch(tmp_path, capsys):
    destination = tmp_path / "hierarchy-demo"
    assert main(["demo", "--starter", "hierarchy", "--out", str(destination)]) == 0
    artifact = destination / "artifacts/hierarchy.s1.json"
    state = destination / "project/sample_state.json"
    assert artifact.exists()
    capsys.readouterr()
    assert main(["inspect", str(artifact)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["format"] == "systemone-hierarchy/v1"
    assert inspected["source_to_nodes"]["root:router"] == ["router"]
    assert inspected["final"]["resolution"]["candidates"][0]["distribution_scope"]
    assert main(["run", str(artifact), "--state", str(state)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] in {"completed", "review_required"}
    assert result["synthetic"] is True
    failed = tmp_path / "failed.json"
    assert main(["run", str(artifact), "--state", str(state), "--max-calls", "1",
                 "--out", str(failed)]) == 2
    failure = json.loads(failed.read_text())
    assert failure["status"] == "failed"
    assert failure["stages"]["billing"]["status"] == "failed"
    assert failure["error"]["type"] == "BudgetExceeded"
    assert main(["run", str(artifact), "--state", str(state), "--backend", "typesafe"]) == 2
    assert main(["export-playground", str(artifact), "--state", str(state),
                 "--out", str(tmp_path / "whole.json")]) == 2
    assert main(["export-playground", str(artifact), "--state", str(state),
                 "--stage", "router", "--out", str(tmp_path / "router.json")]) == 0
    assert set(json.loads((tmp_path / "router.json").read_text())) == {"model", "state", "questions"}
    assert main(["export-playground", str(artifact), "--state", str(state),
                 "--stage", "billing", "--out", str(tmp_path / "billing.json")]) == 2
    assert main(["export-playground", str(artifact), "--resolved-state", str(state),
                 "--stage", "billing", "--out", str(tmp_path / "billing.json")]) == 0
    assert json.loads((tmp_path / "billing.json").read_text())["state"] == json.loads(state.read_text())
    assert main(["demo", "--starter", "hierarchy", "--out", str(destination)]) == 2


def test_hierarchy_cli_compile_evaluate_and_draft(tmp_path):
    project = tmp_path / "project"
    assert main(["init", str(project), "--starter", "hierarchy"]) == 0
    draft = tmp_path / "draft.json"
    assert main(["draft", str(project / "source.json"), "--out", str(draft)]) == 0
    assert main(["inspect", str(draft)]) == 0
    assert main(["draft", str(project / "source.json"), "--out", str(draft)]) == 2
    splits = [item for name in ("train", "validation", "calibration", "test")
              for item in (f"--{name}", str(project / f"{name}.jsonl"))]
    output = tmp_path / "compiled"
    assert main(["compile", str(project / "source.json"), *splits, "--out", str(output),
                 "--min-calibration-samples", "1"]) == 0
    artifact = output / "hierarchy.s1.json"
    assert artifact.exists()
    assert main(["compile", str(project / "source.json"), *splits, "--out", str(output)]) == 2
    assert main(["evaluate", str(artifact), *splits,
                 "--out", str(tmp_path / "evaluation.json")]) == 0
    assert (tmp_path / "evaluation.json").exists()
    assert main(["evaluate", str(artifact), "--data", str(project / "test.jsonl"),
                 "--out", str(tmp_path / "invalid.json")]) == 2


def test_mock_hierarchy_runtime_imports_no_optimizer_or_vendor(tmp_path):
    project = tmp_path / "project"
    assert main(["init", str(project), "--starter", "hierarchy"]) == 0
    artifact = tmp_path / "draft.json"
    assert main(["draft", str(project / "source.json"), "--out", str(artifact)]) == 0
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    code = ("import s1compiler,sys; from s1compiler.backends import ManagedBackend,MockBackend; "
            "from s1compiler.hierarchy import HierarchyArtifact; "
            "from s1compiler.hierarchy_runtime import HierarchyRuntime; "
            "a=HierarchyArtifact.load(sys.argv[1]); "
            "r=HierarchyRuntime(a,ManagedBackend(MockBackend(),max_calls=10)).run({'message':'refund sample'}); "
            "assert r['synthetic']; "
            "assert not {'dspy','gepa','typesafe_sdk'}.intersection(sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code, str(artifact)],
                               cwd=tmp_path, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_cli_loads_reworded_frozen_graph(tmp_path, capsys):
    from s1compiler.hierarchy import HierarchyArtifact

    project = tmp_path / "project"
    assert main(["init", str(project), "--starter", "hierarchy"]) == 0
    draft = tmp_path / "draft.json"
    assert main(["draft", str(project / "source.json"), "--out", str(draft)]) == 0
    graph = HierarchyArtifact.load(draft)
    question = next(iter(graph.nodes[0].program.questions.values()))
    question.instructions["question"] += " Use the declared labels."
    reworded = tmp_path / "reworded.json"
    graph.save(reworded)
    capsys.readouterr()
    assert main(["inspect", str(reworded)]) == 0
    assert json.loads(capsys.readouterr().out)["graph_sha256"] == graph.content_hash
    assert main(["run", str(reworded), "--state", str(project / "sample_state.json")]) == 0


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
