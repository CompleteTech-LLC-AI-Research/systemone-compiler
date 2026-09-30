"""Shared native attempt accounting, cache isolation, recovery and deadlines."""
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from s1compiler.backends import AnswerCache, AttemptLedger, Budget, ManagedBackend, MockBackend
from s1compiler.errors import BackendError, BudgetExceeded, ConfigurationError
from s1compiler.hierarchy import HierarchySource, lower_hierarchy
from s1compiler.hierarchy_runtime import GraphRetryPolicy, HierarchyRuntime


FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "hierarchy_contract"


def chain():
    return lower_hierarchy(HierarchySource.load(FIXTURES / "chain.json"))


class Recording(MockBackend):
    def __init__(self):
        self.calls = []

    def evaluate(self, program, state):
        self.calls.append(program.name)
        return super().evaluate(program, state)


def test_graph_limit_blocks_child_before_native_dispatch():
    recorder = Recording()
    backend = ManagedBackend(recorder, max_calls=20)
    result = HierarchyRuntime(chain(), backend, max_graph_attempts=1).run({"message": "urgent outage"})
    assert result["status"] == "failed"
    assert result["error"]["type"] == "BudgetExceeded"
    assert recorder.calls == ["signal"]
    assert backend.budget.used == 1
    assert result["accounting"]["predicted_worst_case_leaf_calls"] == 2
    assert result["accounting"]["attempt_ledger"]["used"] == 1
    backend.close()


def test_global_limit_and_graph_limit_reconcile_without_double_charge():
    recorder = Recording()
    backend = ManagedBackend(recorder, max_calls=1)
    result = HierarchyRuntime(chain(), backend).run({"message": "urgent outage"})
    assert result["status"] == "failed"
    assert result["accounting"]["requests_attempted"] == 1
    assert result["accounting"]["attempt_ledger"]["used"] == 1
    assert recorder.calls == ["signal"]
    backend.close()


def test_nested_instances_share_one_graph_limit():
    recorder = Recording()
    backend = ManagedBackend(recorder)
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / "nested.json"))
    result = HierarchyRuntime(artifact, backend, max_graph_attempts=1).run({
        "first_note": "urgent outage", "second_note": "routine request"})
    assert result["status"] == "failed"
    assert result["executed"] == ["first/check"]
    assert recorder.calls == ["check"]
    assert result["accounting"]["attempt_ledger"]["receipts"][0]["node_id"] == "first/check"
    backend.close()


def test_default_has_no_retry_but_explicit_transient_retry_is_charged():
    class Flaky(Recording):
        def evaluate(self, program, state):
            self.calls.append(program.name)
            if len(self.calls) == 1:
                raise TimeoutError("synthetic transport timeout")
            return MockBackend.evaluate(self, program, state)

    recorder = Flaky()
    backend = ManagedBackend(recorder)
    failed = HierarchyRuntime(chain(), backend).run({"message": "urgent outage"})
    assert failed["status"] == "failed"
    assert failed["accounting"]["attempt_ledger"]["receipts"][0]["status"] == "uncertain"
    assert failed["accounting"]["usage_unknown_calls"] == 1
    assert failed["accounting"]["reported_input_tokens"] is None
    assert recorder.calls == ["signal"]
    backend.close()

    recorder = Flaky()
    backend = ManagedBackend(recorder)
    policy = GraphRetryPolicy(max_transient_retries=1)
    completed = HierarchyRuntime(chain(), backend, retry_policy=policy).run({"message": "urgent outage"})
    assert completed["status"] == "completed"
    assert recorder.calls == ["signal", "signal", "priority"]
    receipts = completed["accounting"]["attempt_ledger"]["receipts"]
    assert [item["status"] for item in receipts] == ["uncertain", "validated", "validated"]
    assert completed["accounting"]["requests_attempted"] == 3
    assert completed["accounting"]["reported_output_tokens"] is None
    assert completed["accounting"]["attempt_ledger"]["policy"]["max_transient_retries"] == 1
    backend.close()


def test_tighter_node_limit_blocks_explicit_retry():
    class AlwaysTimeout(Recording):
        def evaluate(self, program, state):
            self.calls.append(program.name)
            raise TimeoutError("synthetic transport timeout")

    recorder = AlwaysTimeout()
    backend = ManagedBackend(recorder)
    runtime = HierarchyRuntime(chain(), backend, retry_policy=GraphRetryPolicy(max_transient_retries=1),
                               node_attempt_limits={"signal": 1})
    result = runtime.run({"message": "urgent outage"})
    assert result["status"] == "failed"
    assert result["error"]["type"] == "BudgetExceeded"
    assert recorder.calls == ["signal"]
    assert result["accounting"]["attempt_ledger"]["used"] == 1
    backend.close()


def test_invalid_response_retry_requires_separate_explicit_policy():
    class MalformedOnce(Recording):
        def evaluate(self, program, state):
            response = super().evaluate(program, state)
            if len(self.calls) == 1:
                response.answers["urgent"]["noul"] = -1
            return response

    recorder = MalformedOnce()
    backend = ManagedBackend(recorder)
    result = HierarchyRuntime(chain(), backend,
                              retry_policy=GraphRetryPolicy(max_invalid_response_retries=1)).run(
                                  {"message": "urgent outage"})
    assert result["status"] == "completed"
    assert recorder.calls == ["signal", "signal", "priority"]
    receipts = result["accounting"]["attempt_ledger"]["receipts"]
    assert [item["status"] for item in receipts] == ["invalid_response", "validated", "validated"]
    backend.close()


def test_invalid_response_is_not_cached_and_identity_failure_is_fatal():
    class MalformedThenValid(Recording):
        def evaluate(self, program, state):
            response = super().evaluate(program, state)
            if len(self.calls) == 1:
                response.answers["urgent"]["noul"] = -1
            return response

    recorder = MalformedThenValid()
    backend = ManagedBackend(recorder, cache=AnswerCache())
    first = HierarchyRuntime(chain(), backend).run({"message": "urgent outage"})
    second = HierarchyRuntime(chain(), backend).run({"message": "urgent outage"})
    assert first["status"] == "failed"
    assert second["status"] == "completed"
    assert recorder.calls == ["signal", "signal", "priority"]
    assert second["accounting"]["cache_hits"] == 0
    backend.close()

    class WrongModel(Recording):
        def evaluate(self, program, state):
            response = super().evaluate(program, state)
            response.model = "unexpected-model"
            return response

    recorder = WrongModel()
    backend = ManagedBackend(recorder)
    result = HierarchyRuntime(chain(), backend,
                              retry_policy=GraphRetryPolicy(max_transient_retries=3,
                                                            max_invalid_response_retries=3)).run(
                                                                {"message": "urgent outage"})
    assert result["status"] == "failed"
    assert result["error"]["type"] == "BackendError"
    assert result["accounting"]["attempt_ledger"]["receipts"][0]["status"] == "identity_rejected"
    assert recorder.calls == ["signal"]
    backend.close()


@pytest.mark.parametrize("delay", [float("nan"), float("inf"), -1, 16])
def test_retry_policy_rejects_non_finite_or_out_of_range_delay(delay):
    with pytest.raises(ConfigurationError):
        GraphRetryPolicy(delay_seconds=delay)


def test_all_cache_run_uses_no_native_reservations_and_reapplies_policy():
    recorder = Recording()
    backend = ManagedBackend(recorder, cache=AnswerCache())
    original = chain()
    first = HierarchyRuntime(original, backend).run({"message": "urgent outage"})
    assert first["status"] == "completed"
    changed = original.model_copy(deep=True)
    changed.nodes[0].program.policies["urgent"].force_review = True
    changed.nodes[0].on_review = "continue_marked"
    second = HierarchyRuntime(changed, backend).run({"message": "urgent outage"})
    assert second["status"] == "review_required"
    assert second["graph_sha256"] != first["graph_sha256"]
    assert second["accounting"]["requests_attempted"] == 0
    assert second["accounting"]["cache_hits"] == 2
    assert second["accounting"]["attempt_ledger"]["used"] == 0
    assert len(recorder.calls) == 2
    backend.close()


def test_changed_child_wire_payload_causes_cache_miss():
    recorder = Recording()
    backend = ManagedBackend(recorder, cache=AnswerCache())
    original = chain()
    HierarchyRuntime(original, backend).run({"message": "urgent outage"})
    changed = original.model_copy(deep=True)
    changed.nodes[0].program.questions["urgent"].instructions = "Changed self-contained prompt"
    result = HierarchyRuntime(changed, backend).run({"message": "urgent outage"})
    assert result["accounting"]["requests_attempted"] == 1
    assert result["accounting"]["cache_hits"] == 1
    assert len(recorder.calls) == 3
    backend.close()


def test_deadline_after_late_completion_preserves_charged_first_leaf():
    class Late(Recording):
        def evaluate(self, program, state):
            time.sleep(0.02)
            return super().evaluate(program, state)

    recorder = Late()
    backend = ManagedBackend(recorder)
    result = HierarchyRuntime(chain(), backend).run({"message": "urgent outage"}, timeout_seconds=0.001)
    assert result["status"] == "cancelled"
    assert result["executed"] == ["signal"]
    assert result["stages"]["signal"]["status"] == "completed"
    assert result["accounting"]["requests_attempted"] == 1
    assert result["accounting"]["attempt_ledger"]["receipts"][0]["status"] == "validated"
    backend.close()


def test_restore_marks_in_flight_uncertain_and_requires_owner_budget_reconciliation():
    artifact = chain()
    nodes = {node.id for node in artifact.nodes}
    owner = Budget(5)
    ledger = AttemptLedger(graph_sha256=artifact.content_hash, maximum=2, nodes=nodes,
                           policy={"max_transient_retries": 0,
                                   "max_invalid_response_retries": 0, "delay_seconds": 0.0})
    ledger.admit(owner, "signal")
    snapshot = ledger.snapshot()
    restored = AttemptLedger.restore(snapshot, graph_sha256=artifact.content_hash,
                                     maximum=2, nodes=nodes,
                                     policy={"max_transient_retries": 0,
                                             "max_invalid_response_retries": 0, "delay_seconds": 0.0})
    assert restored.snapshot()["receipts"][0]["status"] == "uncertain"
    backend = ManagedBackend(Recording(), max_calls=5)
    runtime = HierarchyRuntime(artifact, backend, max_graph_attempts=2)
    with pytest.raises(ConfigurationError, match="owner budget"):
        runtime.run({"message": "urgent outage"}, ledger=restored)
    backend.budget.used = owner.used
    result = runtime.run({"message": "urgent outage"}, ledger=restored)
    assert result["status"] == "failed"  # Next leaf cannot exceed the original two-attempt cap.
    assert result["accounting"]["attempt_ledger"]["used"] == 2
    backend.close()


def test_retry_policy_mismatch_rejected_on_resume():
    artifact = chain()
    nodes = {node.id for node in artifact.nodes}
    policy = {"max_transient_retries": 1, "max_invalid_response_retries": 0, "delay_seconds": 0.0}
    snapshot = AttemptLedger(graph_sha256=artifact.content_hash, maximum=2,
                             nodes=nodes, policy=policy).snapshot()
    with pytest.raises(ConfigurationError, match="identity or limits"):
        AttemptLedger.restore(snapshot, graph_sha256=artifact.content_hash, maximum=2, nodes=nodes,
                              policy={"max_transient_retries": 0})


def test_live_backend_failure_never_falls_back_to_mock(monkeypatch):
    class LiveFailure:
        identity = "fake-live-test/v1"
        synthetic = False
        def __init__(self):
            self.calls = 0
        def evaluate(self, program, state):
            self.calls += 1
            raise BackendError("Test-only live provider unavailable")
        def close(self):
            pass

    def forbid_mock(*args, **kwargs):
        raise AssertionError("Mock fallback is forbidden")

    monkeypatch.setattr(MockBackend, "evaluate", forbid_mock)
    live = LiveFailure()
    backend = ManagedBackend(live)
    result = HierarchyRuntime(chain(), backend).run({"message": "urgent outage"})
    assert result["status"] == "failed"
    assert result["error"]["type"] == "BackendError"
    assert live.calls == 1
    backend.close()


def test_atomic_owner_and_graph_admission_under_concurrent_probes():
    recorder = Recording()
    backend = ManagedBackend(recorder, max_calls=1)
    artifact = chain()
    ledger = AttemptLedger(graph_sha256=artifact.content_hash, maximum=2,
                           nodes={node.id for node in artifact.nodes})
    program = artifact.nodes[0].program
    barrier = threading.Barrier(2)

    def probe():
        barrier.wait(timeout=5)
        try:
            backend.evaluate(program, {"message": "urgent outage"}, ledger=ledger, node_id="signal")
            return "ok"
        except BudgetExceeded:
            return "budget"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: probe(), range(2)))
    assert sorted(outcomes) == ["budget", "ok"]
    assert len(recorder.calls) == 1
    assert backend.budget.used == ledger.snapshot()["used"] == 1
    backend.close()
