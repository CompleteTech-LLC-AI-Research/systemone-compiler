"""Graph evaluation uses root-level quality and honest probability scopes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from typewright.backends import ManagedBackend, MockBackend
from typewright.data import Example
from typewright.errors import DataError
from typewright import hierarchy_metrics
from typewright.hierarchy import HierarchySource, lower_hierarchy
from typewright.hierarchy_data import HierarchySplitGuard
from typewright.hierarchy_metrics import (evaluate_hierarchy, paired_flat_hierarchy,
                                          replay_hierarchy_evaluation, report_from_hierarchy_results,
                                          HierarchyEvaluationFailure)
from typewright.hierarchy_runtime import HierarchyRuntime


FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "hierarchy_contract"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))


def fixture(name):
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / f"{name}.json"))
    state = CASES[f"{name}.json"]["state"]
    result = HierarchyRuntime(artifact, ManagedBackend(MockBackend())).run(state)
    gold = {key: value["value"] for key, value in result["decisions"].items()}
    row = Example(id="root", state=state, expected=gold, group="cluster_a")
    other = {part: [Example(id=part, state={key: f"{value} {part}" for key, value in state.items()},
                            expected=gold, group=f"cluster_{part}")]
             for part in ("train", "calibration", "test")}
    guard = HierarchySplitGuard(artifact, {"validation": [row], **other})
    return artifact, row, guard


@pytest.mark.parametrize("name", ["chain", "conditional", "diamond", "nested"])
def test_graph_evaluation_and_strict_evidence_replay(name, tmp_path):
    artifact, row, guard = fixture(name)
    first, results = evaluate_hierarchy(artifact, [row], ManagedBackend(MockBackend()),
                                        guard=guard, split="validation", evidence_root=tmp_path)
    replayed, replay_results = replay_hierarchy_evaluation(
        artifact, [row], ManagedBackend(MockBackend()), guard=guard,
        split="validation", evidence_root=tmp_path)
    assert first == replayed
    assert results == replay_results
    assert first["synthetic"] is True
    assert first["objective"] == pytest.approx(1)
    assert first["coverage"]["completed_n"] == 1
    assert first["usage"]["native_attempts_total"] == len(results[0]["executed"])
    assert first["stages"]["status_by_cluster"]["cluster_a"]
    if name == "conditional":
        probability = first["outputs"][next(iter(artifact.source.decisions))]["probability"]
        assert probability["eligible_n"] == 0
        assert probability["unavailable_reasons"] == {"branch_conditional_not_global": 1}
    if name == "chain":
        assert first["outputs"]["priority"]["probability"]["brier"] is None


def test_wrong_route_and_review_are_charged_to_all_root_objective(tmp_path):
    artifact, row, guard = fixture("conditional")
    _, results = evaluate_hierarchy(artifact, [row], ManagedBackend(MockBackend()),
                                    guard=guard, split="validation", evidence_root=tmp_path)
    wrong = copy.deepcopy(results[0])
    decision = next(iter(artifact.source.decisions))
    wrong["decisions"][decision]["value"] = next(
        value for value in artifact.source.decisions[decision].criteria
        if value != row.expected[decision])
    wrong_row = Example(id="wrong", state=row.state, expected=row.expected, group=row.group)
    wrong["lineage"]["root_id"] = "wrong"
    report = report_from_hierarchy_results(artifact, [row, wrong_row], [results[0], wrong])
    assert report["objective"] == pytest.approx(0.5)
    assert report["routes"]["final_output_error_n"] == {decision: 1}
    review = copy.deepcopy(wrong)
    review["status"] = "review_required"
    review["decisions"] = {}
    report = report_from_hierarchy_results(artifact, [row, wrong_row], [results[0], review])
    assert report["objective"] == pytest.approx(0.5)
    assert report["coverage"]["review_n"] == 1


def test_perfect_specialist_cannot_hide_wrong_router():
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / "conditional.json"))
    state = {"message": "software crash bug"}
    specialist = next(node for node in artifact.nodes if node.id == "technical")
    assert MockBackend().evaluate(specialist.program, state).answers["resolution"]["choice"] == "bug_fix"

    class WrongRouter(MockBackend):
        def evaluate(self, program, inputs):
            response = super().evaluate(program, inputs)
            if program.name == "router":
                response.answers["department"].update(
                    choice="billing", probabilities={"billing": 0.9, "technical": 0.05, "other": 0.05})
            return response

    row = Example(id="misroute", state=state, expected={"resolution": "bug_fix"})
    others = {part: [Example(id=part, state={"message": f"{part} ticket"},
                             expected={"resolution": "general"})]
              for part in ("train", "calibration", "test")}
    guard = HierarchySplitGuard(artifact, {"validation": [row], **others})
    report, results = evaluate_hierarchy(artifact, [row], ManagedBackend(WrongRouter()),
                                         guard=guard, split="validation")
    assert results[0]["status"] == "completed"
    assert results[0]["executed"] == ["router", "billing"]
    assert report["objective"] == 0
    assert report["routes"]["final_output_error_n"] == {"resolution": 1}


def test_report_rejects_unknown_usage_fabricated_as_known(tmp_path):
    artifact, row, guard = fixture("chain")
    _, results = evaluate_hierarchy(artifact, [row], ManagedBackend(MockBackend()),
                                    guard=guard, split="validation", evidence_root=tmp_path)
    result = copy.deepcopy(results[0])
    result["accounting"]["usage_unknown_calls"] = 0
    result["accounting"]["reported_input_tokens"] = None
    with pytest.raises(DataError, match="Known usage"):
        report_from_hierarchy_results(artifact, [row], [result])


def test_all_cache_has_no_native_latency_and_unknown_usage_is_unavailable(tmp_path):
    artifact, row, guard = fixture("nested")
    _, results = evaluate_hierarchy(artifact, [row], ManagedBackend(MockBackend()),
                                    guard=guard, split="validation", evidence_root=tmp_path)
    result = copy.deepcopy(results[0])
    for stage in result["stages"].values():
        if stage.get("kind") == "leaf":
            stage["cache_hit"] = True
    result["accounting"]["usage_unknown_calls"] = 1
    result["accounting"]["reported_input_tokens"] = None
    result["accounting"]["reported_output_tokens"] = None
    report = report_from_hierarchy_results(artifact, [row], [result])
    assert report["latency_ms"]["uncached_native_mean"] is None
    assert report["latency_ms"]["uncached_native_unavailable_reason"]
    assert report["usage"]["reported_input_tokens"] is None
    assert report["outputs"]["first_review"]["probability"]["eligible_n"] == 1
    assert report["outputs"]["second_review"]["probability"]["eligible_n"] == 1


def test_execution_failure_has_no_selection_score(tmp_path):
    artifact, row, guard = fixture("chain")
    _, results = evaluate_hierarchy(artifact, [row], ManagedBackend(MockBackend()),
                                    guard=guard, split="validation", evidence_root=tmp_path)
    result = copy.deepcopy(results[0])
    result["status"] = "failed"
    report = report_from_hierarchy_results(artifact, [row], [result])
    assert report["status"] == "execution_failed"
    assert report["objective"] is None
    with pytest.raises(HierarchyEvaluationFailure):
        paired_flat_hierarchy(artifact, [row], [[result]], [[{
            "id": row.id, "group": row.group, "gold": row.expected, "result": results[0]}]])


def test_noul_threshold_changes_value_quality_but_not_probability_scope(tmp_path):
    artifact, row, guard = fixture("nested")
    _, results = evaluate_hierarchy(artifact, [row], ManagedBackend(MockBackend()),
                                    guard=guard, split="validation", evidence_root=tmp_path)
    name = "first_review"
    result = copy.deepcopy(results[0])
    result["decisions"][name]["p_true"] = 0.49
    result["decisions"][name]["value"] = False
    low = report_from_hierarchy_results(artifact, [row], [result])
    result["decisions"][name]["p_true"] = 0.51
    result["decisions"][name]["value"] = True
    high = report_from_hierarchy_results(artifact, [row], [result])
    if row.expected[name]:
        assert high["objective"] > low["objective"]
    else:
        assert low["objective"] > high["objective"]
    assert low["outputs"][name]["probability"]["eligible_n"] == 1
    assert high["outputs"][name]["probability"]["eligible_n"] == 1


def test_paired_flat_comparison_requires_identical_roots(tmp_path):
    artifact, row, guard = fixture("chain")
    _, results = evaluate_hierarchy(artifact, [row], ManagedBackend(MockBackend()),
                                    guard=guard, split="validation", evidence_root=tmp_path)
    flat = {"id": row.id, "group": row.group, "gold": row.expected,
            "result": {"model": artifact.source.model, "synthetic": True,
                       "decisions": results[0]["decisions"]}}
    paired = paired_flat_hierarchy(artifact, [row], [results], [[flat]])
    assert paired["mean_delta_hierarchy_minus_flat"] == pytest.approx(0)
    assert paired["n_clusters"] == 1
    with pytest.raises(DataError, match="identical ordered roots"):
        paired_flat_hierarchy(artifact, [row], [results], [[{**flat, "id": "other"}]])


# --- Score and weighted quality, pinned with exact values ----------------------------------------
def _decl(kind, levels=5, tolerance=0.5, weight=1.0):
    criteria = list(range(levels)) if kind == "score" else {"a": "", "b": ""}
    return SimpleNamespace(type=kind, criteria=criteria, score_tolerance=tolerance, weight=weight)


@pytest.mark.parametrize("value,gold,expected", [(2, 2, 1.0), (2, 4, 0.5), (0, 4, 0.0), (4, 0, 0.0),
                                                 (1.0, 1.5, 0.875)])
def test_score_quality_is_one_minus_normalized_absolute_error(value, gold, expected):
    assert hierarchy_metrics._value_quality(_decl("score", levels=5), value, gold) == pytest.approx(expected)


def test_score_quality_scales_with_the_declared_number_of_levels():
    # The same one-level miss costs more on a short scale than on a long one.
    assert hierarchy_metrics._value_quality(_decl("score", levels=3), 1, 2) == pytest.approx(0.5)
    assert hierarchy_metrics._value_quality(_decl("score", levels=11), 1, 2) == pytest.approx(0.9)


def test_choice_and_noul_quality_are_exact_match_only():
    assert hierarchy_metrics._value_quality(_decl("choice"), "a", "a") == 1.0
    assert hierarchy_metrics._value_quality(_decl("choice"), "a", "b") == 0.0
    assert hierarchy_metrics._value_quality(_decl("noul"), True, True) == 1.0
    assert hierarchy_metrics._value_quality(_decl("noul"), True, False) == 0.0


def test_score_tolerance_boundary_is_inclusive_for_correctness_only():
    declaration = _decl("score", levels=5, tolerance=0.5)
    assert hierarchy_metrics._correct(declaration, 2.0, 2.5) is True
    assert hierarchy_metrics._correct(declaration, 2.0, 2.5000001) is False
    assert hierarchy_metrics._value_quality(declaration, 2.0, 2.5) == pytest.approx(0.875)  # still graded, not rounded


def test_complete_quality_weights_each_declared_output():
    source = SimpleNamespace(decisions={"kind": _decl("choice", weight=3.0), "level": _decl("score", levels=5, weight=1.0)})
    row = SimpleNamespace(expected={"kind": "a", "level": 4})
    decisions = {"kind": {"value": "a"}, "level": {"value": 2}}
    # (3 * 1 + 1 * 0.5) / 4
    assert hierarchy_metrics._complete_quality(source, row, decisions) == pytest.approx(0.875)
    decisions["kind"]["value"] = "b"
    assert hierarchy_metrics._complete_quality(source, row, decisions) == pytest.approx(0.125)


def test_graph_objective_on_a_score_output_grades_distance_not_exact_match(tmp_path):
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / "chain.json"))
    state = CASES["chain.json"]["state"]
    predicted = HierarchyRuntime(artifact, ManagedBackend(MockBackend())).run(state)["decisions"]["priority"]["value"]
    levels = len(artifact.source.decisions["priority"].criteria)
    miss = predicted + 1 if predicted + 1 <= levels - 1 else predicted - 1
    row = Example(id="root", state=state, expected={"priority": miss}, group="cluster_a")
    other = {part: [Example(id=part, state={key: f"{value} {part}" for key, value in state.items()},
                            expected={"priority": miss}, group=f"cluster_{part}")]
             for part in ("train", "calibration", "test")}
    guard = HierarchySplitGuard(artifact, {"validation": [row], **other})
    report, _ = evaluate_hierarchy(artifact, [row], ManagedBackend(MockBackend()), guard=guard,
                                   split="validation", evidence_root=tmp_path)
    assert report["objective"] == pytest.approx(1 - 1 / (levels - 1))
    assert report["outputs"]["priority"]["probability"]["unavailable_reason"] == (
        "score_is_numeric_not_a_public_class_probability")


def test_labels_cannot_change_after_split_preflight(tmp_path):
    artifact, row, guard = fixture("chain")
    row.expected = {**row.expected, "priority": 0}
    with pytest.raises(DataError, match="identity, split, group, state, or labels changed"):
        evaluate_hierarchy(artifact, [row], ManagedBackend(MockBackend()), guard=guard,
                           split="validation", evidence_root=tmp_path)
