import sys
import types
import pytest
from s1compiler.backends import (AnswerCache, Budget, ManagedBackend, MockBackend, Response,
                                  TypeSafeBackend)
from s1compiler.errors import BackendError, BudgetExceeded, ConfigurationError
from s1compiler.models import Program
from s1compiler.runtime import Runtime, apply_policy, normalize_answers


STATE = {"message": "Please refund my duplicate payment.", "customer_plan": "team"}


def test_runtime_mock_is_explicit(program, backend):
    result = Runtime(program, backend).run(STATE)
    assert result["synthetic"] is True
    assert set(result["decisions"]) == set(program.decisions)
    assert result["decisions"]["urgent"]["vendor_confidence"] is None
    assert "confidence" not in result["answers"]["urgent"]


def test_cache_and_accounting(program, backend):
    runtime = Runtime(program, backend)
    first, second = runtime.run(STATE), runtime.run(STATE)
    assert first["answers"] == second["answers"]
    assert second["cache_hit"] and second["latency_ms"] == 0
    assert second["usage"]["input_tokens"] == 0
    assert backend.budget.used == 1
    assert backend.cache_hits == 1


def test_criteria_edit_invalidates_cache(program, backend):
    Runtime(program, backend).run(STATE)
    changed = program.model_copy(deep=True)
    changed.questions["department"].criteria["billing"] = "Receipt only"
    Runtime(changed, backend).run(STATE)
    assert backend.budget.used == 2


def test_policy_edit_reuses_response_but_reapplies_policy(program, backend):
    first = Runtime(program, backend).run(STATE)
    changed = program.model_copy(deep=True)
    changed.policies["department"].force_review = True
    second = Runtime(changed, backend).run(STATE)
    assert second["cache_hit"]
    assert second["decisions"]["department"]["review_required"]
    assert first["answers"] == second["answers"]


def test_extra_state_never_reaches_backend(program):
    class Recording(MockBackend):
        def evaluate(self, program, state):
            assert "secret" not in state
            return super().evaluate(program, state)
    backend = ManagedBackend(Recording())
    Runtime(program, backend).run({**STATE, "secret": "do-not-send"})
    backend.close()


def test_budget_before_network():
    budget = Budget(1)
    budget.reserve()
    with pytest.raises(BudgetExceeded):
        budget.reserve()
    assert budget.used == 1


def test_request_budget_on_uncached_call(program):
    backend = ManagedBackend(MockBackend(), max_calls=1)
    Runtime(program, backend).run(STATE)
    with pytest.raises(BudgetExceeded):
        Runtime(program, backend).run({**STATE, "message": "different"})
    backend.close()


def test_no_auto_live_call(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="explicit"):
        TypeSafeBackend()
    with pytest.raises(ConfigurationError, match="TYPESAFE_API_KEY"):
        TypeSafeBackend(allow_paid=True)


def test_alias_refused(program):
    changed = program.model_copy(deep=True)
    changed.model = "jev-latest"
    backend = ManagedBackend(MockBackend())
    with pytest.raises(ConfigurationError, match="versioned"):
        Runtime(changed, backend).run(STATE)
    assert backend.budget.used == 0
    backend.close()


def test_model_drift_refused(program):
    class Drift(MockBackend):
        def evaluate(self, program, state):
            response = super().evaluate(program, state)
            response.model = "wrong-model"
            return response
    backend = ManagedBackend(Drift())
    with pytest.raises(BackendError, match="pinned"):
        Runtime(program, backend).run(STATE)
    backend.close()


@pytest.mark.parametrize("mutation", ["nan", "negative", "sum", "wrong_label", "missing", "bad_score", "extra"])
def test_malformed_response_rejected(program, mutation):
    response = MockBackend().evaluate(program, STATE)
    if mutation == "nan":
        response.answers["urgent"]["noul"] = float("nan")
    elif mutation == "negative":
        response.answers["urgent"]["noul"] = -0.1
    elif mutation == "sum":
        response.answers["department"]["probabilities"] = {k: .1 for k in program.decisions["department"].criteria}
    elif mutation == "wrong_label":
        response.answers["department"]["choice"] = "invented"
    elif mutation == "missing":
        del response.answers["frustration"]
    elif mutation == "bad_score":
        response.answers["frustration"]["score"] = 88
    else:
        response.answers["extra"] = {"type": "noul", "noul": .8}
    with pytest.raises(BackendError):
        normalize_answers(program, response)


def test_confidence_is_not_max_probability(program):
    response = MockBackend().evaluate(program, STATE)
    response.answers["department"]["confidence"] = 0.123
    answers = normalize_answers(program, response)
    result = apply_policy(program, answers)["department"]
    assert result["vendor_confidence"] == .123
    assert result["gate_score"] == answers["department"]["probabilities"][answers["department"]["choice"]]


def test_binary_gate_uses_chosen_outcome(program):
    response = MockBackend().evaluate(program, STATE)
    response.answers["urgent"]["noul"] = .6
    program.policies["urgent"].noul_threshold = .8
    result = apply_policy(program, normalize_answers(program, response))["urgent"]
    assert result["value"] is False
    assert result["gate_score"] == pytest.approx(.4)


def test_composite_has_no_fake_posterior(program, backend):
    data = program.model_dump()
    data["bindings"]["frustration"] = {"kind": "weighted_mean", "weights": {"frustration": 1, "urgent": 1}}
    composite = Program.model_validate(data)
    result = Runtime(composite, backend).run(STATE)["decisions"]["frustration"]
    assert 0 <= result["value"] <= 2
    assert result["probabilities"] is None
    assert result["vendor_confidence"] is None
    assert result["gate_basis"] == "minimum_component_gate_heuristic"


def test_request_size_limit(program):
    backend = ManagedBackend(MockBackend(), max_request_chars=100)
    with pytest.raises(ConfigurationError, match="character"):
        Runtime(program, backend).run(STATE)
    assert backend.budget.used == 0
    backend.close()


def test_disk_cache_excludes_plaintext_input(program, tmp_path):
    path = tmp_path / "cache.sqlite"
    backend = ManagedBackend(MockBackend(), cache=AnswerCache(path))
    Runtime(program, backend).run({**STATE, "message": "unique-sensitive-marker-x321"})
    backend.close()
    assert b"unique-sensitive-marker-x321" not in path.read_bytes()


def test_cache_expiration(program, tmp_path, monkeypatch):
    cache = AnswerCache(tmp_path / "cache.sqlite", ttl_seconds=1)
    response = MockBackend().evaluate(program, STATE)
    cache.put("key", response)
    assert cache.get("key") is not None
    import time
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now + 2)
    assert cache.get("key") is None
    cache.close()


def test_live_release_guard(program):
    backend = ManagedBackend(MockBackend())
    backend.synthetic = False
    with pytest.raises(ConfigurationError, match="unvalidated"):
        Runtime(program, backend, enforce_release=True)
    backend.close()


def test_typesafe_adapter_sdk_contract_with_test_double(program, monkeypatch):
    calls = []
    class RetryPolicy:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
    class Client:
        def __init__(self, **kwargs):
            assert isinstance(kwargs["retry"], RetryPolicy)
        def system_one(self, *, state, questions, model):
            calls.append((state, questions, model))
            response = MockBackend().evaluate(program, state)
            payload = {"answers": response.answers, "model": model, "usage": {"input_tokens": 45, "output_tokens": 9}}
            return types.SimpleNamespace(model_dump=lambda **kwargs: payload)
        def close(self):
            calls.append("closed")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-real")
    monkeypatch.setitem(sys.modules, "typesafe_sdk", types.SimpleNamespace(TypeSafeClient=Client, RetryPolicy=RetryPolicy))
    backend = ManagedBackend(TypeSafeBackend(allow_paid=True))
    result = Runtime(program, backend).run(STATE)
    assert len(calls) == 1
    assert set(calls[0][1]) == set(program.questions)
    assert calls[0][2] == program.model
    assert result["synthetic"] is False
    assert backend.input_tokens == 45
    backend.close()
    assert calls[-1] == "closed"


def test_infrastructure_failure_is_not_a_low_score(program):
    class Down(MockBackend):
        def evaluate(self, program, state):
            raise BackendError("Provider unavailable")
    backend = ManagedBackend(Down())
    with pytest.raises(BackendError):
        Runtime(program, backend).run(STATE)
    backend.close()


@pytest.mark.parametrize("field,value", [("model", "other-version"), ("synthetic", False)])
def test_corrupt_cache_identity_is_rejected(program, field, value):
    class CorruptCache:
        def get(self, key):
            response = MockBackend().evaluate(program, STATE)
            setattr(response, field, value)
            return response
        def close(self):
            pass
    backend = ManagedBackend(MockBackend(), cache=CorruptCache())
    with pytest.raises(BackendError):
        Runtime(program, backend).run(STATE)
    assert backend.budget.used == 0
    backend.close()


def test_fresh_live_synthetic_mismatch_is_rejected(program):
    class Mislabeled(MockBackend):
        def evaluate(self, program, state):
            response = super().evaluate(program, state)
            response.synthetic = False
            return response
    backend = ManagedBackend(Mislabeled())
    with pytest.raises(BackendError, match="identity"):
        Runtime(program, backend).run(STATE)
    backend.close()


def test_cached_version_is_recorded(program, tmp_path):
    path = tmp_path / "reused.sqlite"
    first = ManagedBackend(MockBackend(), cache=AnswerCache(path))
    Runtime(program, first).run(STATE)
    first.close()
    second = ManagedBackend(MockBackend(), cache=AnswerCache(path))
    result = Runtime(program, second).run(STATE)
    assert result["cache_hit"]
    assert second.accounting()["models_seen"] == [program.model]
    second.close()


def test_sum_tolerance_follows_quantization_not_a_constant():
    """Jev rounds probabilities to 2dp; drift scales with label count.

    Observed live: a 77-label Choice summed to 0.99 (1e-2 off), which the old
    fixed abs_tol=1e-4 rejected. The tolerance must track the returned grid,
    stay tight for full-precision values, and still reject a malformed sum.
    """
    from s1compiler.runtime import MAX_SUM_DRIFT, quantization_step, sum_tolerance

    # Full-precision values get the strict tolerance.
    assert quantization_step([0.14173942690489524, 0.8582605730951048]) == 0.0
    assert sum_tolerance([0.14173942690489524, 0.8582605730951048]) == 1e-4

    # Two-decimal grid, as the provider actually returns.
    assert quantization_step([0.93, 0.02, 0.02, 0.0]) == 0.01
    assert sum_tolerance([0.93, 0.02, 0.02, 0.0]) == pytest.approx(0.02)

    # Large label counts are capped rather than growing without bound.
    assert sum_tolerance([0.01] * 77) == MAX_SUM_DRIFT


def test_high_cardinality_quantized_choice_is_accepted(program):
    """The real 77-label shape that blocked the BANKING77 pilot."""
    from s1compiler.models import Program
    from s1compiler.runtime import Runtime

    labels = [f"label_{i}" for i in range(77)]
    data = program.model_dump(mode="json")
    data["decisions"] = {"intent": {"type": "choice", "goal": "Pick one.",
                                    "criteria": {name: f"Definition {name}." for name in labels}}}
    data["questions"] = {"intent": {"type": "choice", "instructions": {"question": "Pick one."},
                                    "criteria": {name: f"Definition {name}." for name in labels}}}
    data["bindings"] = {"intent": {"kind": "question", "question": "intent"}}
    data["policies"] = {"intent": {}}
    wide = Program.model_validate(data)

    # 0.93 + 0.02 + 0.02 + 74 zeros = 0.97 on a 2dp grid: rounded, not malformed.
    probs = {name: 0.0 for name in labels}
    probs[labels[0]], probs[labels[1]], probs[labels[2]] = 0.93, 0.02, 0.02
    answers = {"intent": {"type": "choice", "choice": labels[0],
                          "probabilities": probs, "confidence": 0.9}}

    class Replay:
        identity, synthetic = "typesafe-sdk/0.7.0", False
        def evaluate(self, program, state):
            return Response(answers=answers, model=program.model, usage={})
        def close(self):
            pass

    result = Runtime(wide, ManagedBackend(Replay(), max_calls=2, cache=None)).run(
        {k: "x" for k in wide.state})
    assert result["decisions"]["intent"]["value"] == labels[0]
    # Renormalized to a true distribution.
    assert sum(result["decisions"]["intent"]["probabilities"].values()) == pytest.approx(1.0)

    # A genuinely malformed sum is still rejected at the same cardinality.
    broken = dict(probs)
    broken[labels[0]] = 0.40
    answers["intent"]["probabilities"] = broken
    with pytest.raises(BackendError, match="sum to one"):
        Runtime(wide, ManagedBackend(Replay(), max_calls=2, cache=None)).run(
            {k: "x" for k in wide.state})
