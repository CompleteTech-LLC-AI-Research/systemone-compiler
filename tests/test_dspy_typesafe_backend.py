"""Installed DSPy with labeled SDK test doubles; never real Jev results."""
import copy

import pytest

from typewright.backends import AnswerCache, AttemptLedger, ManagedBackend, MockBackend, TypeSafeBackend
from typewright.dspy_typesafe_backend import DSPyTypeSafeBackend
from typewright.errors import BackendError, BudgetExceeded, ConfigurationError
from typewright.runtime import Runtime

STATE = {"message": "refund", "customer_plan": "team"}


@pytest.fixture
def sdk_double(monkeypatch, program):
    import socket
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    def forbid_network(sock, address):
        raise AssertionError("Native SDK doubles must never dispatch sockets")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_network)
    pytest.importorskip("dspy")
    import importlib.metadata
    if importlib.metadata.version("dspy") != "3.4.0":
        pytest.skip("Experimental adapter is qualified only against DSPy 3.4.0")
    sdk = pytest.importorskip("typesafe_sdk")
    monkeypatch.setenv("TYPESAFE_API_KEY", "obviously-fake-offline-fixture")
    envelope = MockBackend().evaluate(program, STATE).to_dict()
    envelope["usage"] = {"input_tokens": 7, "output_tokens": 3}
    calls, constructions = [], []
    control = {"envelope": envelope, "error": None}

    class SDKDouble:
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            self.model = kwargs.get("model")

        def system_one(self, **kwargs):
            calls.append({**kwargs, "model": kwargs.get("model", self.model)})
            if control["error"]:
                raise control["error"]

            class Envelope:
                def model_dump(self, *, mode):
                    assert mode == "json"
                    return copy.deepcopy(control["factory"](kwargs) if "factory" in control else control["envelope"])
            return Envelope()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            pass

    monkeypatch.setattr(sdk, "TypeSafeClient", SDKDouble)
    return calls, constructions, control


def test_consent_and_import_boundary(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="explicit"):
        DSPyTypeSafeBackend()
    with pytest.raises(ConfigurationError, match="TYPESAFE_API_KEY"):
        DSPyTypeSafeBackend(allow_paid=True)


def test_native_equivalence(program, sdk_double):
    calls, constructors, _ = sdk_double
    direct = TypeSafeBackend(allow_paid=True)
    wrapped = DSPyTypeSafeBackend(allow_paid=True)
    a, b = direct.evaluate(program, STATE), wrapped.evaluate(program, STATE)
    assert calls[0] == calls[1]
    assert a.answers == b.answers
    assert a.model == b.model == program.model
    assert a.usage == b.usage == {"input_tokens": 7, "output_tokens": 3}
    assert not a.synthetic and not b.synthetic  # Transport doubles, no live inference claim.
    for config in constructors:
        assert config["retry"].max_retries == 0


def test_managed_cache_budget_and_leaf_receipts(program, sdk_double):
    calls, _, _ = sdk_double
    backend = ManagedBackend(DSPyTypeSafeBackend(allow_paid=True), max_calls=1, cache=AnswerCache())
    ledger = AttemptLedger(graph_sha256="fixture", maximum=1, nodes={"leaf"})
    response = backend.evaluate(program, STATE, ledger=ledger, node_id="leaf")
    assert ledger.snapshot()["receipts"][0]["status"] == "response_received"
    backend.remember_validated(program, STATE, response, ledger=ledger)
    assert ledger.snapshot()["receipts"][0]["status"] == "validated"
    assert backend.evaluate(program, STATE, ledger=ledger, node_id="leaf").cached
    with pytest.raises(BudgetExceeded):
        backend.evaluate(program, {**STATE, "message": "changed"}, ledger=ledger, node_id="leaf")
    assert len(calls) == 1
    assert backend.accounting()["dollar_cost"] is None
    restored = AttemptLedger.restore(ledger.snapshot(), graph_sha256="fixture", maximum=1, nodes={"leaf"})
    assert restored.snapshot() == ledger.snapshot()
    backend.close()


@pytest.mark.parametrize("failure", ["connection", "identity", "malformed"])
def test_failures_charge_and_never_cache(program, sdk_double, failure):
    calls, _, control = sdk_double
    backend = ManagedBackend(DSPyTypeSafeBackend(allow_paid=True), max_calls=1, cache=AnswerCache())
    if failure == "connection":
        control["error"] = TimeoutError("offline transport double")
    elif failure == "identity":
        control["envelope"]["model"] = "different-model"
    else:
        control["envelope"]["answers"] = {}
    with pytest.raises((BackendError, ConfigurationError)):
        Runtime(program, backend).run(STATE)
    assert backend.budget.used == 1
    assert len(calls) == 1
    assert backend.cache.db.execute("SELECT COUNT(*) FROM answers").fetchone()[0] == 0
    backend.close()


def native_graph_fixture(sdk_double, name):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from typewright.hierarchy import HierarchySource, lower_hierarchy
    from typewright.models import Question
    root = Path(__file__).resolve().parents[1] / "examples" / "hierarchy_contract"
    artifact = lower_hierarchy(HierarchySource.load(root / f"{name}.json"))
    state = json.loads((root / "cases.json").read_text())[f"{name}.json"]["state"]
    _, _, control = sdk_double

    def factory(payload):
        leaf = SimpleNamespace(model=artifact.source.model, questions={
            key: Question.model_validate(value) for key, value in payload["questions"].items()})
        return MockBackend().evaluate(leaf, payload["state"]).to_dict()
    control["factory"] = factory
    return artifact, state


@pytest.mark.parametrize("name", ["chain", "conditional", "diamond", "nested"])
def test_native_graph_equivalence_replay_resume(name, sdk_double, tmp_path):
    from typewright.hierarchy_runtime import HierarchyRuntime
    calls, _, _ = sdk_double
    artifact, state = native_graph_fixture(sdk_double, name)
    direct = HierarchyRuntime(artifact, ManagedBackend(TypeSafeBackend(allow_paid=True))).run(state)
    direct_calls = copy.deepcopy(calls)
    calls.clear()
    directory = tmp_path / "evidence"
    compiled = HierarchyRuntime(artifact, ManagedBackend(DSPyTypeSafeBackend(allow_paid=True))).run(
        state, evidence_dir=directory)
    assert calls == direct_calls
    assert compiled["status"] == direct["status"] == "completed"
    for field in ("decisions", "executed", "path"):
        assert compiled[field] == direct[field]
    receipts = compiled["accounting"]["attempt_ledger"]["receipts"]
    assert len(receipts) == len(calls)
    assert all(receipt["status"] == "validated" for receipt in receipts)
    calls.clear()
    for mode in ("replay", "resume"):
        result = HierarchyRuntime(artifact, ManagedBackend(DSPyTypeSafeBackend(allow_paid=True))).run(
            state, evidence_dir=directory, evidence_mode=mode)
        assert result == compiled
        assert calls == []


def test_interrupted_graph_attempt_stays_charged_on_resume(sdk_double, tmp_path, monkeypatch):
    import json
    from typewright.hierarchy_evidence import HierarchyEvidence
    from typewright.hierarchy_runtime import HierarchyRuntime
    calls, _, _ = sdk_double
    artifact, state = native_graph_fixture(sdk_double, "chain")
    directory = tmp_path / "interrupted"
    original = HierarchyEvidence._append

    def interrupt(self, payload):
        if payload["kind"] == "stage":
            raise RuntimeError("offline simulated interruption")
        return original(self, payload)

    with monkeypatch.context() as context:
        context.setattr(HierarchyEvidence, "_append", interrupt)
        with pytest.raises(RuntimeError, match="interruption"):
            HierarchyRuntime(artifact, ManagedBackend(DSPyTypeSafeBackend(allow_paid=True))).run(
                state, evidence_dir=directory)
    assert len(calls) == 1
    assert json.loads((directory / "attempts.json").read_text())["ledger"]["used"] == 1
    calls.clear()
    resumed = HierarchyRuntime(artifact, ManagedBackend(DSPyTypeSafeBackend(allow_paid=True))).run(
        state, evidence_dir=directory, evidence_mode="resume")
    assert resumed["status"] == "completed"
    assert len(calls) == 2
    assert resumed["accounting"]["requests_attempted"] == 3
    assert resumed["accounting"]["usage_unknown_calls"] == 3
    calls.clear()
    replayed = HierarchyRuntime(artifact, ManagedBackend(DSPyTypeSafeBackend(allow_paid=True))).run(
        state, evidence_dir=directory, evidence_mode="replay")
    assert replayed == resumed
    assert calls == []


def test_no_dspy_history_cache_callbacks_or_tracking(program, sdk_double):
    import dspy

    observed = []

    class ForbiddenCallback:
        def on_lm_start(self, *args, **kwargs):
            observed.append("callback")

    class ForbiddenTracker:
        def add_usage(self, *args, **kwargs):
            observed.append("tracker")

    wrapped = DSPyTypeSafeBackend(allow_paid=True)
    clients = []
    original = wrapped._client_type

    def client_factory(**kwargs):
        client = original(**kwargs)
        clients.append(client)
        return client
    wrapped._client_type = client_factory
    with dspy.context(callbacks=[ForbiddenCallback()], usage_tracker=ForbiddenTracker(), disable_history=False):
        wrapped.evaluate(program, STATE)
    assert observed == []
    assert clients[0].cache is False
    assert clients[0].history == []
    assert clients[0].callbacks == []


def test_unknown_usage_is_unknown(program, sdk_double):
    _, _, control = sdk_double
    control["envelope"].pop("usage")
    backend = ManagedBackend(DSPyTypeSafeBackend(allow_paid=True))
    response = backend.evaluate(program, STATE)
    assert response.usage == {}
    assert backend.accounting()["usage_unknown_calls"] == 1
    assert backend.accounting()["dollar_cost"] is None


def test_version_drift_fails_before_dispatch(monkeypatch, sdk_double):
    import importlib.metadata
    calls, _, _ = sdk_double
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "3.4.1" if name == "dspy" else "0.7.0")
    with pytest.raises(ConfigurationError, match="requires"):
        DSPyTypeSafeBackend(allow_paid=True)
    assert calls == []


@pytest.mark.parametrize("model", ["jev", "jev-latest", "jev-preview", "unexpected-alias"])
def test_only_versioned_native_model_dispatches(program, sdk_double, model):
    calls, _, _ = sdk_double
    program = program.model_copy(update={"model": model})
    with pytest.raises(ConfigurationError, match="versioned"):
        DSPyTypeSafeBackend(allow_paid=True).evaluate(program, STATE)
    assert calls == []


def test_import_compile_module_does_not_import_optional_packages():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c",
                    "import sys; import typewright.dspy_typesafe_backend; "
                    "assert not {'dspy', 'gepa', 'typesafe_sdk'} & set(sys.modules)"], check=True)


def test_actual_sdk_constructor_accepts_upstream_wrapper_arguments(monkeypatch):
    import importlib.metadata
    import socket
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    def forbid_network(sock, address):
        raise AssertionError("SDK constructor contract must stay offline")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_network)
    pytest.importorskip("dspy")
    if importlib.metadata.version("dspy") != "3.4.0":
        pytest.skip("Experimental adapter is qualified only against DSPy 3.4.0")
    sdk = pytest.importorskip("typesafe_sdk")
    monkeypatch.setenv("TYPESAFE_API_KEY", "obviously-fake-offline-fixture")
    backend = DSPyTypeSafeBackend(allow_paid=True, timeout=37.0)
    upstream = backend._client_type(model="jev-1.13.0", timeout=37.0, cache=False, callbacks=[])
    kwargs = upstream._sdk_kwargs()
    assert kwargs["retry"].max_retries == 0
    assert kwargs["retry"].timeout == 37.0
    # Instantiate the actual SDK class, rather than the permissive SDKDouble.
    client = sdk.TypeSafeClient(**kwargs)
    try:
        assert client is not None
    finally:
        client.close()
        backend.close()
