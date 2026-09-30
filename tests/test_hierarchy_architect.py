"""DSPy graph proposals remain JSON data and use train-only feedback."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1compiler.architect import DSPyTeacher
from s1compiler.backends import ManagedBackend, MockBackend
from s1compiler.data import Example
from s1compiler.errors import BackendError, BudgetExceeded, CandidateError, ConfigurationError
from s1compiler.hierarchy import HierarchySource
from s1compiler.hierarchy_architect import (hierarchy_plan, plan_to_hierarchy,
                                             semantic_review_manifest)
from s1compiler.hierarchy_compiler import HierarchyCompileOptions, HierarchyCompiler
from s1compiler.io import canonical


FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "hierarchy_contract" / "conditional.json"


def sources():
    good = HierarchySource.load(FIXTURE)
    data = good.model_dump(mode="json")
    router = data["graph"]["stages"][0]["program"]["questions"]["department"]["criteria"]
    router["billing"], router["technical"] = router["technical"], router["billing"]
    return HierarchySource.model_validate(data), good


def splits():
    return {
        "train": [Example(id="train", state={"message": "refund invoice charge TRAIN_SECRET"},
                          expected={"resolution": "refund"}, group="group_train")],
        "validation": [Example(id="validation", state={"message": "software crash bug VALIDATION_SECRET"},
                               expected={"resolution": "bug_fix"}, group="group_validation")],
        "calibration": [Example(id="calibration", state={"message": "refund invoice charge CALIB_SECRET"},
                                expected={"resolution": "refund"}, group="group_calibration")],
        "test": [Example(id="test", state={"message": "software crash bug HOLDOUT_SECRET"},
                         expected={"resolution": "bug_fix"}, group="group_test")],
    }


def test_valid_router_specialist_proposal_wins_with_train_only_feedback(tmp_path):
    bad, good = sources()

    class Teacher:
        model = "offline-structure-test-double"

        def __init__(self):
            self.feedback = None

        def propose_hierarchy(self, fixed, current, feedback):
            self.feedback = canonical(feedback)
            assert fixed.source == bad.source
            assert current.graph == bad.graph
            return good

        def accounting(self):
            return {"model": self.model, "signature_calls": 1, "provider_requests_attempted": 0}

    teacher = Teacher()
    compiler = HierarchyCompiler(
        ManagedBackend(MockBackend(), max_calls=40), teacher=teacher,
        options=HierarchyCompileOptions(architect="dspy", structural_rounds=1,
                                        min_calibration_samples=1))
    artifact, report = compiler.compile(bad, **splits())
    assert artifact.content_hash == report["graph_sha256"]
    assert report["selection"]["history"][1]["validation_objective"] > (
        report["selection"]["history"][0]["validation_objective"])
    assert "TRAIN_SECRET" in teacher.feedback
    assert "VALIDATION_SECRET" not in teacher.feedback
    assert "CALIB_SECRET" not in teacher.feedback
    assert "HOLDOUT_SECRET" not in teacher.feedback
    review = report["semantic_review"]
    assert review["requires_human_review"] is True
    assert review["changed_child_prompts"]
    assert review["frozen_graph_sha256"] == artifact.content_hash
    assert artifact.provenance.evidence["semantic_review"] == review
    assert report["accounting"]["teacher"]["provider_requests_attempted"] == 0
    artifact.save(tmp_path / "graph.json")


@pytest.mark.parametrize("mutation", [
    "cycle", "contract", "root_access", "oversize", "probability", "executable", "global_scope"
])
def test_invalid_hierarchy_plans_rejected_atomically(mutation):
    _, source = sources()
    plan = copy.deepcopy(hierarchy_plan(source))
    graph = plan["graph"]
    if mutation == "cycle":
        graph["stages"][1]["after"] = ["technical"]
        graph["stages"][2]["after"] = ["billing"]
    elif mutation == "contract":
        graph["outputs"]["resolution"]["criteria"]["invented"] = "new label"
    elif mutation == "root_access":
        graph["stages"][0]["inputs"]["message"] = {"root": "secret"}
    elif mutation == "oversize":
        graph["stages"] = [copy.deepcopy(graph["stages"][0]) for _ in range(65)]
        for index, stage in enumerate(graph["stages"]):
            stage["id"] = f"router_{index}"
    elif mutation == "probability":
        graph["final"]["resolution"]["candidates"][0]["probability_multiplier"] = 0.8
    elif mutation == "executable":
        graph["stages"][0]["python"] = "import os; os.system('echo bad')"
    else:
        graph["final"]["resolution"]["candidates"][0]["distribution_scope"] = "full_contract"
    with pytest.raises(CandidateError):
        plan_to_hierarchy(source, plan)
    assert hierarchy_plan(source) != plan


def test_nested_reusable_definitions_roundtrip_through_proposal_schema():
    nested = HierarchySource.load(FIXTURE.with_name("nested.json"))
    candidate = plan_to_hierarchy(nested, hierarchy_plan(nested))
    assert candidate == nested
    review = semantic_review_manifest(nested, candidate)
    assert review["changed_composition"] == []
    assert review["selected_plan_sha256"] == review["original_plan_sha256"]


def test_dspy_parse_rejection_is_distinct_from_provider_failures():
    _, source = sources()
    teacher = object.__new__(DSPyTeacher)
    teacher.design_hierarchy = object()
    teacher.rejected = 0
    teacher._predict = lambda *_args, **_kwargs: SimpleNamespace(plan_json='{"graph":{}}')
    with pytest.raises(CandidateError, match="invalid hierarchy"):
        teacher.propose_hierarchy(source, source, {"examples": [], "traces": []})
    assert teacher.rejected == 1
    teacher._predict = lambda *_args, **_kwargs: SimpleNamespace(
        plan_json='{"graph":{},"graph":{},"definitions":{}}')
    with pytest.raises(CandidateError, match="invalid hierarchy"):
        teacher.propose_hierarchy(source, source, {"examples": [], "traces": []})
    assert teacher.rejected == 2
    for failure in (BackendError("provider"), BudgetExceeded("budget"),
                    ConfigurationError("auth")):
        def fail(*_args, **_kwargs):
            raise failure
        teacher._predict = fail
        with pytest.raises(type(failure)):
            teacher.propose_hierarchy(source, source, {"examples": [], "traces": []})
    assert teacher.rejected == 2


def test_rejected_candidate_keeps_prior_valid_graph_and_never_reaches_test():
    bad, _ = sources()

    class Teacher:
        def propose_hierarchy(self, fixed, current, feedback):
            raise CandidateError("invalid graph")

        def accounting(self):
            return {"signature_calls": 1, "provider_requests_attempted": 0}

    compiler = HierarchyCompiler(
        ManagedBackend(MockBackend()), teacher=Teacher(),
        options=HierarchyCompileOptions(architect="dspy", structural_rounds=1))
    session = compiler.select(bad, **splits())
    assert session.candidate.content_hash == session.guard.graph_sha256
    assert session.proposal_history[1]["rejected"] == "invalid_typed_graph"
    assert session.selected_source == bad
    assert session.phase == "selected"
    assert semantic_review_manifest(bad, bad)["changed_composition"] == []
