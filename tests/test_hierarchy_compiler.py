"""Authored hierarchy compilation and held-out phase boundaries, all synthetic."""
import copy
from pathlib import Path

import pytest

from s1compiler.backends import ManagedBackend, MockBackend
from s1compiler.data import Example
from s1compiler.errors import ConfigurationError, DataError
from s1compiler.hierarchy import HierarchyArtifact, HierarchySource, lower_hierarchy
from s1compiler.hierarchy_compiler import HierarchyCompileOptions, HierarchyCompiler
from s1compiler.hierarchy_data import HierarchySplitGuard
from s1compiler.hierarchy_metrics import HierarchyEvaluationFailure, evaluate_hierarchy
from s1compiler.hierarchy_runtime import HierarchyRuntime


FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "hierarchy_contract"


def inputs():
    source = HierarchySource.load(FIXTURES / "conditional.json")
    return source, {
        "train": [Example(id="train", state={"message": "refund invoice charge train"},
                          expected={"resolution": "refund"}, group="group_train")],
        "validation": [Example(id="valid", state={"message": "software crash bug validation"},
                               expected={"resolution": "bug_fix"}, group="group_valid")],
        "calibration": [Example(id="calib", state={"message": "refund invoice charge calibration"},
                                expected={"resolution": "refund"}, group="group_calib")],
        "test": [Example(id="holdout", state={"message": "software crash bug test sentinel"},
                         expected={"resolution": "bug_fix"}, group="group_test")],
    }


def test_no_key_compile_freezes_review_gates_and_roundtrips(tmp_path):
    source, splits = inputs()
    backend = ManagedBackend(MockBackend(), max_calls=32)
    compiler = HierarchyCompiler(backend, options=HierarchyCompileOptions(min_calibration_samples=1))
    artifact, report = compiler.compile(source, **splits)
    assert report["status"] == "synthetic"
    assert report["test"]["hierarchy"]["n_root"] == 1
    assert report["selection"]["train"]["n_root"] == 1
    assert report["selection"]["validation"]["n_root"] == 1
    assert report["calibration"]["before_gates"]["n_root"] == 1
    assert report["calibration"]["after_gates"]["n_root"] == 1
    assert report["calibration"]["fit"]["resolution"]["technical"]["status"] == "review_only"
    assert report["test"]["hierarchy"]["coverage"]["review_n"] == 1
    assert report["test"]["paired"]["n_clusters"] == 1
    assert report["accounting"]["native_calls_this_compile"] <= 32
    assert report["accounting"]["native_calls_this_compile"] == backend.budget.used
    assert set(artifact.source.decisions) == set(source.source.decisions)
    assert artifact.source.model_dump(mode="json") == source.source.model_dump(mode="json")
    assert artifact.provenance.deployment_approved is False
    artifact.save(tmp_path / "graph.json")
    loaded = HierarchyArtifact.load(tmp_path / "graph.json")
    assert loaded.content_hash == artifact.content_hash
    assert loaded.provenance_hash == artifact.provenance_hash
    assert HierarchyRuntime(loaded, ManagedBackend(MockBackend())).run(
        splits["test"][0].state)["status"] == "review_required"


def test_test_split_is_never_seen_before_freeze_and_runs_once():
    source, splits = inputs()
    marker = {"frozen": False, "test_calls": 0}

    class Recorder(MockBackend):
        def evaluate(self, program, state):
            if "sentinel" in str(state):
                assert marker["frozen"]
                marker["test_calls"] += 1
            return super().evaluate(program, state)

    class SpyCompiler(HierarchyCompiler):
        def freeze(self, session):
            assert marker["test_calls"] == 0
            artifact = super().freeze(session)
            marker["frozen"] = True
            return artifact

    compiler = SpyCompiler(ManagedBackend(Recorder()),
                          options=HierarchyCompileOptions(min_calibration_samples=1))
    session = compiler.select(source, **splits)
    assert marker["test_calls"] == 0
    compiler.calibrate(session)
    assert marker["test_calls"] == 0
    compiler.freeze(session)
    assert marker["test_calls"] == 0
    compiler.test(session)
    assert marker["test_calls"] == 3  # Flat baseline, router, specialist.
    with pytest.raises(ConfigurationError, match="once"):
        compiler.test(session)


def test_preflight_failure_precedes_calls_and_route_policy_changes_hash():
    source, splits = inputs()
    backend = ManagedBackend(MockBackend())
    bad = copy.deepcopy(splits)
    bad["test"][0].id = bad["train"][0].id
    with pytest.raises(DataError):
        HierarchyCompiler(backend).select(source, **bad)
    assert backend.budget.used == 0

    artifact = lower_hierarchy(source)
    guard = HierarchySplitGuard(artifact, splits)
    report, results = evaluate_hierarchy(artifact, splits["validation"], backend,
                                         guard=guard, split="validation")
    assert report["n_root"] == 1
    changed = artifact.model_copy(deep=True)
    router = next(node for node in changed.nodes if node.id == "router")
    router.program.policies["department"].min_gate = 0.9
    assert changed.content_hash != artifact.content_hash
    with pytest.raises(DataError, match="changed"):
        guard.bind(changed, "validation", splits["validation"][0])
    assert results[0]["graph_sha256"] == artifact.content_hash


def test_owner_budget_failure_is_not_a_zero_quality_candidate():
    source, splits = inputs()
    backend = ManagedBackend(MockBackend(), max_calls=1)
    with pytest.raises(HierarchyEvaluationFailure):
        HierarchyCompiler(backend).select(source, **splits)
    assert backend.budget.used == 1


def test_changed_dataset_cannot_be_frozen_or_sent_to_test():
    source, splits = inputs()
    compiler = HierarchyCompiler(ManagedBackend(MockBackend()),
                                 options=HierarchyCompileOptions(min_calibration_samples=1))
    session = compiler.select(source, **splits)
    compiler.calibrate(session)
    session.splits["test"][0].expected["resolution"] = "general"
    with pytest.raises(DataError, match="changed"):
        compiler.freeze(session)


def test_undersized_gepa_budget_is_rejected_before_any_native_call():
    source, splits = inputs()

    class CountingBackend(MockBackend):
        calls = 0

        def evaluate(self, program, state):
            CountingBackend.calls += 1
            return super().evaluate(program, state)

    backend = ManagedBackend(CountingBackend(), max_calls=32)
    options = HierarchyCompileOptions(optimizer="gepa", max_metric_calls=len(splits["validation"]) + 1,
                                      min_calibration_samples=1)
    compiler = HierarchyCompiler(backend, teacher=object(), options=options)
    with pytest.raises(ConfigurationError, match="exceed initial validation"):
        compiler.select(source, **splits)
    assert CountingBackend.calls == 0
    assert backend.budget.used == 0
    backend.close()
