"""GEPA optimizes qualified graph text while scoring whole synthetic paths."""
import copy
from dataclasses import dataclass
from pathlib import Path

import pytest

from s1compiler.backends import ManagedBackend, MockBackend
from s1compiler.data import Example
from s1compiler.errors import BackendError, CandidateError, DataError
from s1compiler.hierarchy import HierarchyArtifact, HierarchySource, lower_hierarchy
from s1compiler.hierarchy_compiler import HierarchyCompileOptions, HierarchyCompiler
from s1compiler.hierarchy_gepa import (HierarchyGEPAAdapter, components_from_hierarchy,
                                       hierarchy_from_components, hierarchy_text_structure_hash,
                                       optimize_hierarchy_gepa)
from s1compiler.hierarchy_metrics import HierarchyEvaluationFailure
from s1compiler.io import canonical


FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "hierarchy_contract"


@dataclass
class Batch:
    outputs: list
    scores: list
    trajectories: list | None


class FixedTeacher:
    def propose_components(self, candidate, reflective_dataset, components):
        return {key: candidate[key] for key in components}


def dataset():
    return {
        "train": [Example(id="train", state={"message": "refund invoice charge train"},
                          expected={"resolution": "refund"}, group="group_train")],
        "validation": [Example(id="validation", state={"message": "software crash bug valid"},
                               expected={"resolution": "bug_fix"}, group="group_validation")],
        "calibration": [Example(id="calibration", state={"message": "refund invoice charge calib"},
                                expected={"resolution": "refund"}, group="group_calibration")],
        "test": [Example(id="test", state={"message": "software crash bug holdout"},
                         expected={"resolution": "bug_fix"}, group="group_test")],
    }


def test_qualified_components_roundtrip_nested_reuse_without_sibling_mutation(tmp_path):
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / "nested.json"))
    components = components_from_hierarchy(artifact)
    assert hierarchy_from_components(artifact, components) == artifact
    assert len(components) == len(set(components))
    first = next(key for key in components if key.startswith("graph/first/check/question/")
                 and key.endswith("/instructions"))
    second = next(key for key in components if key.startswith("graph/second/check/question/")
                  and key.endswith("/instructions"))
    edited = copy.deepcopy(components)
    edited[first] = canonical("Inspect only the first synthetic note.")
    optimized = hierarchy_from_components(artifact, edited)
    after = components_from_hierarchy(optimized)
    assert after[first] != components[first]
    assert after[second] == components[second]
    assert hierarchy_text_structure_hash(optimized) == hierarchy_text_structure_hash(artifact)
    assert optimized.content_hash != artifact.content_hash
    assert optimized.final == artifact.final
    assert optimized.source == artifact.source
    assert optimized.nodes[0].program.policies == artifact.nodes[0].program.policies
    assert optimized.provenance.status == "draft"
    assert artifact.provenance.status == "draft"
    optimized.save(tmp_path / "optimized.json")
    assert HierarchyArtifact.load(tmp_path / "optimized.json").content_hash == optimized.content_hash


@pytest.mark.parametrize("mutation", ["missing", "extra", "bad_json", "number", "too_long"])
def test_graph_component_mutations_reject_atomically(mutation):
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / "conditional.json"))
    original = components_from_hierarchy(artifact)
    edited = copy.deepcopy(original)
    key = next(iter(edited))
    if mutation == "missing":
        edited.pop(key)
    elif mutation == "extra":
        edited["graph/bad/question/x/instructions"] = '"bad"'
    elif mutation == "bad_json":
        edited[key] = "not JSON"
    elif mutation == "number":
        edited[key] = "42"
    else:
        edited[key] = '"' + "x" * 24000 + '"'
    with pytest.raises(CandidateError):
        hierarchy_from_components(artifact, edited)
    assert components_from_hierarchy(artifact) == original


def test_full_graph_scores_and_unvisited_stage_reflection_are_train_only():
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / "conditional.json"))
    rows = dataset()
    backend = ManagedBackend(MockBackend(), max_calls=20)
    adapter = HierarchyGEPAAdapter(artifact, rows, backend, FixedTeacher(), batch_factory=Batch)
    candidate = components_from_hierarchy(artifact)
    evaluated = adapter.evaluate(rows["train"], candidate, capture_traces=True)
    assert evaluated.scores == [1]
    assert backend.budget.used == 2  # Router and selected specialist.
    technical = next(key for key in candidate if key.startswith("graph/technical/question/")
                     and key.endswith("/instructions"))
    reflection = adapter.make_reflective_dataset(candidate, evaluated, [technical])
    assert reflection[technical][0]["Feedback"]["visited"] is False
    assert reflection[technical][0]["Generated Outputs"]["stage"] is None
    assert reflection[technical][0]["Feedback"]["annotated_stage_errors"] == []
    assert adapter.propose_new_texts(candidate, reflection, [technical]) == {technical: candidate[technical]}
    changed_candidate = dict(candidate, **{technical: canonical("Different prompt")})
    with pytest.raises(DataError, match="registered train"):
        adapter.propose_new_texts(changed_candidate, reflection, [technical])
    with pytest.raises(DataError, match="train"):
        adapter.evaluate(rows["validation"], candidate, capture_traces=True)
    with pytest.raises(DataError, match="registered train"):
        adapter.make_reflective_dataset(candidate, Batch([], [], [{"fake": "test"}]), [technical])


def test_invalid_candidate_is_rejection_but_task_and_budget_failures_propagate():
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / "conditional.json"))
    rows = dataset()
    candidate = components_from_hierarchy(artifact)
    backend = ManagedBackend(MockBackend(), max_calls=20)
    adapter = HierarchyGEPAAdapter(artifact, rows, backend, FixedTeacher(), batch_factory=Batch)
    invalid = adapter.evaluate(rows["train"], {"bad": '"bad"'}, capture_traces=True)
    assert invalid.scores == [0]
    assert backend.budget.used == 0
    with pytest.raises(DataError, match="registered train"):
        adapter.make_reflective_dataset({"bad": '"bad"'}, invalid, [next(iter(candidate))])

    class Failing(MockBackend):
        def evaluate(self, program, state):
            raise BackendError("synthetic provider failure")

    with pytest.raises(HierarchyEvaluationFailure):
        HierarchyGEPAAdapter(artifact, rows, ManagedBackend(Failing()), FixedTeacher(),
                             batch_factory=Batch).evaluate(rows["train"], candidate)
    with pytest.raises(HierarchyEvaluationFailure):
        HierarchyGEPAAdapter(artifact, rows, ManagedBackend(MockBackend(), max_calls=1),
                             FixedTeacher(), batch_factory=Batch).evaluate(rows["train"], candidate)


@pytest.mark.optional
def test_real_gepa_offline_graph_task_and_structure_then_wording():
    pytest.importorskip("gepa")
    good = HierarchySource.load(FIXTURES / "conditional.json")
    data = good.model_dump(mode="json")
    router = data["graph"]["stages"][0]["program"]["questions"]["department"]["criteria"]
    router["billing"], router["technical"] = router["technical"], router["billing"]
    bad = HierarchySource.model_validate(data)
    rows = dataset()

    class Teacher(FixedTeacher):
        def propose_hierarchy(self, fixed, current, feedback):
            assert {item["root_id"] for item in feedback["examples"]} == {"train"}
            return good

        def accounting(self):
            return {"test_double": True, "provider_requests_attempted": 0}

    backend = ManagedBackend(MockBackend(), max_calls=100)
    compiler = HierarchyCompiler(backend, teacher=Teacher(), options=HierarchyCompileOptions(
        architect="dspy", structural_rounds=1, optimizer="gepa", max_metric_calls=16,
        min_calibration_samples=1))
    session = compiler.select(bad, **rows)
    assert [item["phase"] for item in session.proposal_history] == [
        "authored", "dspy_structure", "gepa_wording"]
    assert session.optimization["engine"] == "gepa"
    assert session.optimization["native_stage_calls"] > 0
    assert session.optimization["metric_rows_evaluated"] > 0
    compiler.calibrate(session)
    artifact = compiler.freeze(session)
    report = compiler.test(session)
    assert report["test"]["hierarchy"]["n_root"] == 1
    assert artifact.content_hash == report["graph_sha256"]
    assert report["accounting"]["native_calls_this_compile"] == backend.budget.used


def test_standalone_real_gepa_fake_proposer_executes_graph(capsys):
    pytest.importorskip("gepa")
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / "conditional.json"))
    backend = ManagedBackend(MockBackend(), max_calls=80)
    optimized, report = optimize_hierarchy_gepa(
        artifact, dataset(), backend, FixedTeacher(), max_metric_calls=16)
    assert optimized.content_hash == artifact.content_hash
    assert report["native_stage_calls"] == backend.budget.used
    assert report["native_stage_calls"] > 0
    assert capsys.readouterr().out == ""


def _reflection_setup(teacher):
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / "conditional.json"))
    rows = dataset()
    backend = ManagedBackend(MockBackend(), max_calls=20)
    adapter = HierarchyGEPAAdapter(artifact, rows, backend, teacher, batch_factory=Batch)
    candidate = components_from_hierarchy(artifact)
    evaluated = adapter.evaluate(rows["train"], candidate, capture_traces=True)
    key = next(name for name in candidate if name.startswith("graph/billing/question/")
               and name.endswith("/instructions"))
    reflection = adapter.make_reflective_dataset(candidate, evaluated, [key])
    return adapter, candidate, reflection, key


class _Proposer:
    def __init__(self, response=None, error=None):
        self.response, self.error = response, error

    def propose_components(self, candidate, reflective_dataset, components):
        if self.error is not None:
            raise self.error
        return self.response(candidate, components) if callable(self.response) else self.response


@pytest.mark.parametrize("label,response", [
    ("not_json_text", lambda candidate, keys: {keys[0]: "plain text, not a JSON string"}),
    ("json_number", lambda candidate, keys: {keys[0]: "42"}),
    ("oversized", lambda candidate, keys: {keys[0]: '"' + "x" * 24000 + '"'}),
    ("extra_component", lambda candidate, keys: {keys[0]: candidate[keys[0]],
                                                 "graph/other/question/x/instructions": '"new"'}),
    ("missing_component", lambda candidate, keys: {}),
    ("unknown_component", lambda candidate, keys: {"graph/nowhere/question/x/instructions": '"new"'}),
    ("not_a_mapping", lambda candidate, keys: ["not", "a", "dict"]),
    ("none", lambda candidate, keys: None),
])
def test_a_malformed_proposal_is_rejected_and_the_current_text_is_kept(label, response):
    adapter, candidate, reflection, key = _reflection_setup(_Proposer(response))
    assert adapter.invalid_candidates == 0
    assert adapter.propose_new_texts(candidate, reflection, [key]) == {key: candidate[key]}, label
    assert adapter.invalid_candidates == 1


def test_a_valid_proposal_is_returned_unchanged_and_not_counted_invalid():
    new_text = canonical("A clearer billing instruction.")
    adapter, candidate, reflection, key = _reflection_setup(_Proposer(lambda c, keys: {keys[0]: new_text}))
    assert adapter.propose_new_texts(candidate, reflection, [key]) == {key: new_text}
    assert adapter.invalid_candidates == 0


@pytest.mark.parametrize("error", [BackendError("provider unavailable"), RuntimeError("proposer crashed")])
def test_provider_or_programming_errors_in_the_proposer_propagate(error):
    adapter, candidate, reflection, key = _reflection_setup(_Proposer(error=error))
    with pytest.raises(type(error), match=str(error)):
        adapter.propose_new_texts(candidate, reflection, [key])
    assert adapter.invalid_candidates == 0
