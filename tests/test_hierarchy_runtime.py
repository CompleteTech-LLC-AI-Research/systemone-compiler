"""Native frozen-graph execution with a synthetic recording backend."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from typewright.backends import ManagedBackend, MockBackend
from typewright.errors import ConfigurationError
from typewright.hierarchy import HierarchySource, lower_hierarchy
from typewright.hierarchy_runtime import HierarchyRuntime


FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "hierarchy_contract"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))


class Recording(MockBackend):
    def __init__(self):
        self.calls = []

    def evaluate(self, program, state):
        self.calls.append((program.name, copy.deepcopy(state)))
        return super().evaluate(program, state)


def run_fixture(name, state=None, *, backend=None):
    source = HierarchySource.load(FIXTURES / f"{name}.json")
    recorder = backend or Recording()
    managed = ManagedBackend(recorder)
    result = HierarchyRuntime(lower_hierarchy(source), managed).run(state or CASES[f"{name}.json"]["state"])
    managed.close()
    return result, recorder


@pytest.mark.parametrize("name", ["chain", "conditional", "diamond", "nested"])
def test_contract_graphs_execute_only_expected_native_stages(name):
    result, recorder = run_fixture(name)
    expected = CASES[f"{name}.json"]["expected"]
    assert result["status"] == expected["status"]
    assert result["executed"] == expected["executed"]
    assert result["accounting"]["requests_attempted"] == len(expected["executed"])
    assert len(recorder.calls) == len(expected["executed"])
    assert result["synthetic"] is True
    assert set(result["decisions"])
    if name == "nested":
        assert result["stages"]["first"]["status"] == "completed"
        assert result["stages"]["second"]["status"] == "completed"
        assert len(set(result["executed"])) == len(result["executed"])
    if name == "conditional":
        assert result["stages"]["technical"]["status"] == "skipped"
        chosen = result["decisions"]["resolution"]
        assert chosen["distribution_scope"] == "branch_conditional"
        assert chosen["origin"] == {"stage": "billing", "decision": "resolution"}
        assert set(chosen["probabilities"]) == {"refund", "general"}


def test_join_sees_validated_parent_values_and_no_undeclared_root_state():
    recorder = Recording()
    source = HierarchySource.load(FIXTURES / "diamond.json")
    managed = ManagedBackend(recorder)
    result = HierarchyRuntime(lower_hierarchy(source), managed).run({
        **CASES["diamond.json"]["state"], "secret": "never send"})
    managed.close()
    assert result["status"] == "completed"
    assert [name for name, _ in recorder.calls] == ["breaking", "operational", "combined"]
    assert all("secret" not in state for _, state in recorder.calls)
    joined = recorder.calls[-1][1]
    assert type(joined["breaking"]) is bool
    assert type(joined["severity"]) in (int, float)


def test_nested_caller_state_is_not_mutated_by_child_backend():
    data = json.loads((FIXTURES / "chain.json").read_text(encoding="utf-8"))
    data["source"]["state"]["message"]["type"] = "object"
    data["graph"]["inputs"]["message"]["type"] = "object"
    for stage in data["graph"]["stages"]:
        stage["program"]["state"]["message"]["type"] = "object"

    class Mutating(Recording):
        def evaluate(self, program, state):
            state["message"]["text"] = "mutated by backend"
            return super().evaluate(program, state)

    caller = {"message": {"text": "urgent outage"}}
    recorder = Mutating()
    managed = ManagedBackend(recorder)
    result = HierarchyRuntime(lower_hierarchy(HierarchySource.model_validate(data)), managed).run(caller)
    managed.close()
    assert result["status"] == "completed"
    assert caller == {"message": {"text": "urgent outage"}}


def test_unmatched_branch_reviews_without_inventing_public_decision():
    result, recorder = run_fixture("conditional", {"message": "press partnership"})
    assert result["status"] == "review_required"
    assert result["decisions"] == {}
    assert result["executed"] == ["router"]
    assert len(recorder.calls) == 1


def test_nonidentity_label_map_preserves_conditional_child_distribution():
    data = HierarchySource.load(FIXTURES / "conditional.json").model_dump(mode="json")
    child = next(stage for stage in data["graph"]["stages"] if stage["id"] == "billing")["program"]
    criteria = child["decisions"]["resolution"]["criteria"]
    renamed = {"credit": criteria["refund"], "other": criteria["general"]}
    child["decisions"]["resolution"]["criteria"] = renamed
    child["questions"]["resolution"]["criteria"] = renamed
    data["graph"]["final"]["resolution"]["candidates"][0]["label_map"] = {
        "credit": "refund", "other": "general"}

    class ChildRecording(Recording):
        def evaluate(self, program, state):
            response = super().evaluate(program, state)
            if program.name == "billing":
                self.child_probabilities = copy.deepcopy(response.answers["resolution"]["probabilities"])
            return response

    recorder = ChildRecording()
    artifact = lower_hierarchy(HierarchySource.model_validate(data))
    result = HierarchyRuntime(artifact, ManagedBackend(recorder)).run({"message": "refund invoice charge"})
    assert result["status"] == "completed"
    assert [name for name, _ in recorder.calls] == ["router", "billing"]
    final = result["decisions"]["resolution"]
    assert final["value"] == "refund"
    assert final["origin"] == {"stage": "billing", "decision": "resolution"}
    assert final["distribution_scope"] == "branch_conditional"
    assert set(final["probabilities"]) == {"credit", "other"}
    assert final["probabilities"] == pytest.approx(recorder.child_probabilities)
    assert max(recorder.child_probabilities, key=recorder.child_probabilities.get) == "credit"


@pytest.mark.parametrize("threshold,enabled", [(0.599999, True), (0.6, True), (0.600001, False)])
def test_noul_threshold_routes_on_value_without_changing_p_true(threshold, enabled):
    data = HierarchySource.load(FIXTURES / "chain.json").model_dump(mode="json")
    data["graph"]["stages"][0]["program"]["policies"]["urgent"]["noul_threshold"] = threshold
    data["graph"]["stages"][1]["when"] = {"all": [{
        "ref": {"stage": "signal", "decision": "urgent", "field": "value"},
        "op": "eq", "value": True}]}

    class FixedNoul(Recording):
        def evaluate(self, program, state):
            response = super().evaluate(program, state)
            if program.name == "signal":
                response.answers["urgent"]["noul"] = 0.6
                self.p_true = response.answers["urgent"]["noul"]
            return response

    recorder = FixedNoul()
    result = HierarchyRuntime(lower_hierarchy(HierarchySource.model_validate(data)),
                              ManagedBackend(recorder)).run({"message": "urgent outage"})
    assert recorder.p_true == 0.6
    assert [name for name, _ in recorder.calls] == (["signal", "priority"] if enabled else ["signal"])
    assert result["accounting"]["requests_attempted"] == (2 if enabled else 1)
    assert result["stages"]["priority"]["status"] == ("completed" if enabled else "skipped")
    assert result["status"] == ("completed" if enabled else "review_required")
    if not enabled:
        assert result["decisions"] == {}
    else:
        assert recorder.calls[-1][1]["urgent"] is True


def test_skipped_subgraph_marks_all_nested_stages_with_reason():
    data = json.loads((FIXTURES / "nested.json").read_text(encoding="utf-8"))
    data["graph"]["stages"][0]["when"] = {"all": [{
        "ref": {"root": "first_note"}, "op": "eq", "value": "not present"}]}
    recorder = Recording()
    managed = ManagedBackend(recorder)
    result = HierarchyRuntime(lower_hierarchy(HierarchySource.model_validate(data)), managed).run(
        CASES["nested.json"]["state"])
    managed.close()
    assert result["status"] == "review_required"
    assert result["stages"]["first"] == {"status": "skipped", "reason": "condition_false"}
    assert result["stages"]["first/check"] == {"status": "skipped", "reason": "ancestor_skipped"}
    assert result["executed"] == ["second/check"]


def test_review_blocked_subgraph_marks_unvisited_descendants():
    data = json.loads((FIXTURES / "nested.json").read_text(encoding="utf-8"))
    first = data["graph"]["stages"][0]
    first["after"] = ["second"]
    data["definitions"]["check_note"]["stages"][0]["program"]["policies"]["review"]["force_review"] = True
    recorder = Recording()
    managed = ManagedBackend(recorder)
    result = HierarchyRuntime(lower_hierarchy(HierarchySource.model_validate(data)), managed).run(
        CASES["nested.json"]["state"])
    managed.close()
    assert result["status"] == "review_required"
    assert result["stages"]["first"]["status"] == "review_blocked"
    assert result["stages"]["first/check"]["reason"] == "ancestor_review_blocked"


@pytest.mark.parametrize("continue_marked", [False, True])
def test_parent_review_is_never_erased_by_confident_child(continue_marked):
    data = json.loads((FIXTURES / "conditional.json").read_text(encoding="utf-8"))
    router = data["graph"]["stages"][0]
    router["program"]["policies"]["department"]["force_review"] = True
    router["on_review"] = "continue_marked" if continue_marked else "defer"
    recorder = Recording()
    managed = ManagedBackend(recorder)
    result = HierarchyRuntime(lower_hierarchy(HierarchySource.model_validate(data)), managed).run(
        CASES["conditional.json"]["state"])
    managed.close()
    assert result["status"] == "review_required"
    if continue_marked:
        assert result["executed"] == ["router", "billing"]
        assert result["decisions"]["resolution"]["review_required"] is True
    else:
        assert result["executed"] == ["router"]
        assert result["stages"]["billing"]["status"] == "review_blocked"
        assert result["decisions"] == {}


def test_optional_default_allows_branch_with_skipped_predecessor():
    data = json.loads((FIXTURES / "conditional.json").read_text(encoding="utf-8"))
    technical = data["graph"]["stages"][2]
    technical["program"]["state"]["message"]["required"] = False
    technical["inputs"]["message"] = {
        "stage": "billing", "decision": "resolution", "field": "value", "default": "software crash bug"}
    recorder = Recording()
    managed = ManagedBackend(recorder)
    result = HierarchyRuntime(lower_hierarchy(HierarchySource.model_validate(data)), managed).run(
        {"message": "software crash bug"})
    managed.close()
    assert result["status"] == "completed"
    assert result["executed"] == ["router", "technical"]
    assert recorder.calls[-1][1] == {"message": "software crash bug"}


def test_provider_model_drift_is_failure_without_descendant_dispatch():
    class Drift(Recording):
        def evaluate(self, program, state):
            response = super().evaluate(program, state)
            response.model = "wrong-model"
            return response
    recorder = Drift()
    result, _ = run_fixture("chain", backend=recorder)
    assert result["status"] == "failed"
    assert result["decisions"] == {}
    assert result["stages"]["signal"]["status"] == "failed"
    assert result["stages"]["priority"]["status"] == "pending"
    assert len(recorder.calls) == 1
    assert "pinned target" in result["error"]["message"]


@pytest.mark.parametrize("fault", ["synthetic_drift", "malformed_answer"])
def test_invalid_parent_response_is_not_published_to_child(fault):
    class Invalid(Recording):
        def evaluate(self, program, state):
            response = super().evaluate(program, state)
            if fault == "synthetic_drift":
                response.synthetic = False
            else:
                response.answers["urgent"]["noul"] = -1
            return response
    recorder = Invalid()
    result, _ = run_fixture("chain", backend=recorder)
    assert result["status"] == "failed"
    assert result["decisions"] == {}
    assert result["executed"] == []
    assert len(recorder.calls) == 1


def test_second_leaf_identity_drift_fails_the_graph():
    class DriftSecond(Recording):
        def evaluate(self, program, state):
            response = super().evaluate(program, state)
            if len(self.calls) == 2:
                response.model = "other-version"
            return response
    recorder = DriftSecond()
    result, _ = run_fixture("chain", backend=recorder)
    assert result["status"] == "failed"
    assert result["stages"]["signal"]["status"] == "completed"
    assert result["stages"]["priority"]["status"] == "failed"
    assert result["decisions"] == {}
    assert len(recorder.calls) == 2


def test_nested_leaf_failure_marks_ancestor_and_stops_other_instances():
    class DriftFirst(Recording):
        def evaluate(self, program, state):
            response = super().evaluate(program, state)
            response.model = "wrong-model"
            return response
    recorder = DriftFirst()
    result, _ = run_fixture("nested", backend=recorder)
    assert result["status"] == "failed"
    assert result["stages"]["first/check"]["status"] == "failed"
    assert result["stages"]["first"]["status"] == "failed"
    assert result["stages"]["second"]["status"] == "pending"
    assert len(recorder.calls) == 1


def test_identical_responses_have_identical_route_and_output_scope():
    source = HierarchySource.load(FIXTURES / "conditional.json")
    managed = ManagedBackend(MockBackend())
    runtime = HierarchyRuntime(lower_hierarchy(source), managed)
    first = runtime.run({"message": "software crash bug"})
    second = runtime.run({"message": "software crash bug"})
    managed.close()
    assert first["status"] == second["status"]
    assert first["executed"] == second["executed"]
    assert first["decisions"] == second["decisions"]
    assert first["executed"] == ["router", "technical"]
    assert first["decisions"]["resolution"]["origin"]["stage"] == "technical"


def test_cancelled_before_dispatch_and_release_guard():
    source = HierarchySource.load(FIXTURES / "nested.json")
    recorder = Recording()
    managed = ManagedBackend(recorder)
    artifact = lower_hierarchy(source)
    result = HierarchyRuntime(artifact, managed).run(CASES["nested.json"]["state"],
                                                     cancel_requested=lambda: True)
    assert result["status"] == "cancelled"
    assert not recorder.calls
    managed.synthetic = False
    with pytest.raises(ConfigurationError, match="composition is not measured"):
        HierarchyRuntime(artifact, managed, enforce_release=True)
    managed.close()


def test_frozen_graph_import_and_mock_run_need_no_compile_or_vendor_sdk():
    script = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'dspy', 'gepa', 'typesafe_sdk', 'typesafe'}:
        raise AssertionError('optional dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from typewright import HierarchyRuntime, HierarchySource, lower_hierarchy
from typewright.backends import ManagedBackend, MockBackend
source = HierarchySource.load(sys.argv[1])
backend = ManagedBackend(MockBackend())
result = HierarchyRuntime(lower_hierarchy(source), backend).run({'message': 'refund invoice charge'})
backend.close()
assert result['status'] == 'completed' and result['synthetic']
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(FIXTURES.parents[1] / "src")
    completed = subprocess.run([sys.executable, "-c", script, str(FIXTURES / "conditional.json")],
                               capture_output=True, text=True, env=env)
    assert completed.returncode == 0, completed.stderr
