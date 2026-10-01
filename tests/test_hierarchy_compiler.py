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
    assert any("Flat baseline uses default policies without calibration" in limitation
               and "not a matched-calibration comparison" in limitation
               for limitation in report["limitations"])
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


def test_route_policy_change_requires_fresh_preflight_and_new_execution():
    source, splits = inputs()

    class Recording(MockBackend):
        def __init__(self):
            self.calls = []

        def evaluate(self, program, state):
            self.calls.append(program.name)
            return super().evaluate(program, state)

    recorder = Recording()
    backend = ManagedBackend(recorder)
    authored = lower_hierarchy(source)
    old_guard = HierarchySplitGuard(authored, splits)
    _, old_results = evaluate_hierarchy(authored, splits["validation"], backend,
                                        guard=old_guard, split="validation")
    assert recorder.calls == ["router", "technical"]
    assert old_results[0]["status"] == "completed"
    changed_source = source.model_copy(deep=True)
    changed_source.graph.stages[0].program.policies["department"].force_review = True
    changed = lower_hierarchy(changed_source)
    with pytest.raises(DataError, match="changed"):
        old_guard.bind(changed, "validation", splits["validation"][0])
    assert recorder.calls == ["router", "technical"]
    fresh_guard = HierarchySplitGuard(changed, splits)
    report, fresh_results = evaluate_hierarchy(changed, splits["validation"], backend,
                                              guard=fresh_guard, split="validation")
    assert recorder.calls == ["router", "technical", "router"]
    assert fresh_results[0]["graph_sha256"] == changed.content_hash != authored.content_hash
    assert fresh_results[0]["status"] == "review_required"
    assert fresh_results[0]["stages"]["technical"]["status"] == "review_blocked"
    assert fresh_results[0]["decisions"] == {}
    assert report["quality_by_root_id"] == {"valid": 0.0}
    assert old_results[0]["graph_sha256"] == authored.content_hash
    assert old_results[0]["decisions"]["resolution"]["value"] == "bug_fix"


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


def _calibrated_inputs(per_label=6):
    source, splits = inputs()
    rows = []
    for index in range(per_label):
        rows.append(Example(id=f"cal_refund_{index}", state={"message": f"refund invoice charge calibration {index}"},
                            expected={"resolution": "refund"}, group=f"group_cal_refund_{index}"))
        rows.append(Example(id=f"cal_bug_{index}", state={"message": f"software crash bug calibration {index}"},
                            expected={"resolution": "bug_fix"}, group=f"group_cal_bug_{index}"))
    splits["calibration"] = rows
    return source, splits


def _compile(source, splits, minimum):
    backend = ManagedBackend(MockBackend(), max_calls=400)
    compiler = HierarchyCompiler(backend, options=HierarchyCompileOptions(min_calibration_samples=minimum))
    artifact, report = compiler.compile(source, **splits)
    backend.close()
    return artifact, report


def test_enough_calibration_samples_fit_review_gates_from_calibration_rows_only():
    source, splits = _calibrated_inputs()
    artifact, report = _compile(source, splits, minimum=3)
    fit = report["calibration"]["fit"]["resolution"]
    assert set(fit) == {"billing", "technical"}
    for label, entry in fit.items():
        assert entry["status"] == "fitted", label
        assert entry["observed_n"] == 6 and entry["accepted_n"] == 6
        assert entry["empirical_error"] == 0.0
        assert 0.0 < entry["policy"]["min_gate"] <= 1.0 and entry["policy"]["force_review"] is False
        gate = artifact.final_review_gates["resolution"][label]
        assert (gate.min_gate, gate.force_review) == (entry["policy"]["min_gate"], False)
    assert report["calibration"]["after_gates"]["coverage"]["completed_n"] == 12


def test_too_few_calibration_samples_force_review_for_every_root():
    source, splits = _calibrated_inputs()
    artifact, report = _compile(source, splits, minimum=50)
    for label, entry in report["calibration"]["fit"]["resolution"].items():
        assert entry["status"] == "review_only" and entry["observed_n"] == 6
        assert entry["policy"] == {"min_gate": 0.0, "force_review": True}
        assert artifact.final_review_gates["resolution"][label].force_review is True
    after = report["calibration"]["after_gates"]["coverage"]
    assert after["completed_n"] == 0 and after["review_n"] == 12


def test_fitted_gates_do_not_depend_on_validation_or_test_labels():
    source, splits = _calibrated_inputs()
    first, _ = _compile(source, splits, minimum=3)
    changed_source, changed = _calibrated_inputs()
    changed["validation"][0].expected = {"resolution": "refund"}  # wrong label, held-out rows only
    changed["test"][0].expected = {"resolution": "refund"}
    second, _ = _compile(changed_source, changed, minimum=3)
    assert first.final_review_gates == second.final_review_gates


def test_calibration_preserves_selected_graph_datasets_and_validation_evidence():
    source, splits = _calibrated_inputs()
    original_splits = copy.deepcopy(splits)
    backend = ManagedBackend(MockBackend(), max_calls=400)
    compiler = HierarchyCompiler(backend, options=HierarchyCompileOptions(min_calibration_samples=3))
    session = compiler.select(source, **splits)
    selected_graph = session.candidate.model_dump(mode="json")
    selected_splits = copy.deepcopy(session.splits)
    validation_evidence = copy.deepcopy((session.validation_report, session.baseline_validation,
                                         session.validation_pair, session.proposal_history))
    calls_before = backend.budget.used

    compiler.calibrate(session)

    assert session.phase == "calibrated"
    assert session.candidate.model_dump(mode="json") == selected_graph
    assert session.splits == selected_splits
    assert splits == original_splits
    assert (session.validation_report, session.baseline_validation, session.validation_pair,
            session.proposal_history) == validation_evidence
    assert backend.budget.used - calls_before == 2 * len(splits["calibration"])
    assert all(entry["status"] == "fitted" for entry in session.calibration_fit["resolution"].values())
    frozen = compiler.freeze(session)
    assert frozen.final_review_gates == session.gates
    assert session.candidate.model_dump(mode="json") == selected_graph
    assert session.splits == selected_splits and splits == original_splits
    backend.close()
