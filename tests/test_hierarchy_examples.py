"""Public no-key workflows and checked-in traces for the hierarchy examples."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from s1compiler.backends import ManagedBackend, MockBackend
from s1compiler.cli import main
from s1compiler.hierarchy import HierarchyArtifact, HierarchySource, lower_hierarchy
from s1compiler.hierarchy_data import HierarchySplitGuard, read_hierarchy_jsonl
from s1compiler.hierarchy_runtime import HierarchyRuntime
from s1compiler.io import load_document
from s1compiler.models import Program, UseCase
from s1compiler.runtime import Runtime


ROOT = Path(__file__).resolve().parents[1] / "examples" / "hierarchy"
SPLITS = ("train", "validation", "calibration", "test")


def rows(directory: Path, split: str) -> list[dict]:
    return [json.loads(line) for line in (directory / f"{split}.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("name", ["support", "pr_risk", "nested"])
def test_example_sources_and_split_contracts(name):
    directory = ROOT / name
    source = HierarchySource.load(directory / "source.json")
    assert UseCase.load(directory / "flat_source.json") == source.source
    flat = Program.load(directory / "flat_baseline.s1.json")
    assert flat.decisions == source.source.decisions
    assert flat.provenance["synthetic_fixture"] is True
    assert flat.provenance["deployment_approved"] is False
    flat_result = Runtime(flat, ManagedBackend(MockBackend(), max_calls=10)).run(
        load_document(directory / "sample_state.json"))
    assert set(flat_result["decisions"]) == set(source.source.decisions)
    artifact = lower_hierarchy(source)
    splits = {split: read_hierarchy_jsonl(directory / f"{split}.jsonl", artifact) for split in SPLITS}
    guard = HierarchySplitGuard(artifact, splits)
    assert set(guard.splits) == set(SPLITS)
    assert all(row.id.startswith(f"{name}_{split}_") for split in SPLITS for row in splits[split])
    assert artifact.provenance.status == "draft"
    assert artifact.provenance.deployment_approved is False


@pytest.mark.parametrize("name", ["support", "pr_risk", "nested"])
def test_checked_in_mock_traces_match_public_runtime(name):
    directory = ROOT / name
    artifact = lower_hierarchy(HierarchySource.load(directory / "source.json"))
    expected = load_document(directory / "expected_traces.json")
    assert expected["graph_sha256"] == artifact.content_hash
    assert expected["source_to_nodes"] == artifact.source_to_nodes
    assert expected["provenance"] == {"status": "draft", "synthetic_fixture": True,
                                      "deployment_approved": False}
    backend = ManagedBackend(MockBackend(), max_calls=200)
    sample_rows = {row["id"]: row for row in rows(directory, "train")}
    for trace in expected["traces"]:
        if trace["id"] == "nested_budget_exhaustion":
            limited = ManagedBackend(MockBackend(), max_calls=1)
            result = HierarchyRuntime(artifact, limited).run(sample_rows["nested_train_0"]["state"])
            assert result["status"] == "failed"
            assert result["error"]["type"] == "BudgetExceeded"
            assert result["stages"]["second/check"]["status"] == "failed"
            assert result["synthetic"] == trace["synthetic"]
            assert result["status"] == trace["status"]
            assert result["error"]["type"] == trace["error_type"]
            assert result["path"] == trace["path"]
            assert {key: value["status"] for key, value in result["stages"].items()} == trace["stage_status"]
            continue
        result = HierarchyRuntime(artifact, backend).run(sample_rows[trace["id"]]["state"])
        assert result["synthetic"] is True
        assert result["status"] == trace["status"]
        assert result["path"] == trace["path"]
        assert {key: value["status"] for key, value in result["stages"].items()} == trace["stage_status"]
        assert {key: value["review_required"] for key, value in result["decisions"].items()} == trace["review"]
        for key, value in trace["values"].items():
            actual = result["decisions"][key]["value"]
            if type(value) is float:
                assert actual == pytest.approx(value, abs=1e-6)
            else:
                assert actual == value


def test_examples_cover_branches_join_and_incorrect_router():
    support = ROOT / "support"
    traces = load_document(support / "expected_traces.json")["traces"]
    assert {tuple(trace["path"]) for trace in traces} >= {
        ("router", "billing"), ("router", "technical_triage", "technical_fix"),
        ("router", "technical_triage", "technical_help"), ("router", "sales"), ("router",)}
    wrong = next(trace for trace in traces if trace["case"] == "incorrect_router")
    assert wrong["values"]["resolution"] == "refund"
    assert next(row for row in rows(support, "train") if row["id"] == wrong["id"])["expected"] == {
        "resolution": "bug_fix"}
    source = HierarchySource.load(support / "source.json")
    assert all(candidate.distribution_scope == "branch_conditional"
               for candidate in source.graph.final["resolution"].candidates)

    advisory = lower_hierarchy(HierarchySource.load(ROOT / "pr_risk" / "source.json"))
    combined = next(node for node in advisory.nodes if node.id == "combined")
    assert combined.program.bindings["risk"].kind == "weighted_mean"
    assert all(question.type == "score" for question in combined.program.questions.values())
    assert combined.inputs["breaking"].stage == "breaking"
    assert combined.inputs["severity"].stage == "operational"
    assert all(trace["path"] == ["breaking", "operational", "combined"]
               for trace in load_document(ROOT / "pr_risk" / "expected_traces.json")["traces"])

    nested = load_document(ROOT / "nested" / "expected_traces.json")["traces"]
    nested_source = HierarchySource.load(ROOT / "nested" / "source.json")
    assert nested_source.graph.stages[1].inputs["note"].default == "routine request no risk"
    assert "second_note" not in rows(ROOT / "nested", "train")[3]["state"]
    assert any(trace["case"] == "default_second_note" and "second/check" in trace["path"]
               for trace in nested)
    assert any("first/check" in trace["path"] and "second/check" in trace["path"] for trace in nested)
    assert any(trace["stage_status"]["second/check"] == "skipped" for trace in nested)
    assert any(trace["status"] == "review_required" for trace in nested)
    assert any(trace["id"] == "nested_budget_exhaustion" for trace in nested)


@pytest.mark.parametrize("name", ["support", "pr_risk", "nested"])
def test_authored_cli_compile_and_run_without_keys(name, tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    directory = ROOT / name
    output = tmp_path / f"{name}-run"
    args = ["compile", str(directory / "source.json"), "--out", str(output),
            "--min-calibration-samples", "1"]
    for split in SPLITS:
        args.extend((f"--{split}", str(directory / f"{split}.jsonl")))
    assert main(args) == 0
    artifact = HierarchyArtifact.load(output / "hierarchy.s1.json")
    report = load_document(output / "report.json")
    assert artifact.provenance.status == "synthetic"
    assert artifact.provenance.deployment_approved is False
    assert report["status"] == "synthetic"
    assert report["deployment_approved"] is False
    assert report["accounting"]["owner"]["backend"] == "mock-lexical/v1"
    prediction = tmp_path / f"{name}-prediction.json"
    assert main(["run", str(output / "hierarchy.s1.json"),
                 "--state", str(directory / "sample_state.json"), "--out", str(prediction)]) == 0
    assert load_document(prediction)["synthetic"] is True


def test_regeneration_check_is_part_of_test_suite():
    completed = subprocess.run([sys.executable, str(ROOT / "generate_examples.py"), "--check"],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
