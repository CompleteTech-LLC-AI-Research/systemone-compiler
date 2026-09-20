"""Optional offline contracts against REAL installed packages, no API credentials.

These are skipped when the dependency is not installed. Skips are not successes.
The core adapter tests elsewhere use explicitly identified test doubles.
"""
import inspect
import pytest
from s1compiler.architect import DSPyTeacher
from s1compiler.gepa_adapter import optimize_gepa


@pytest.mark.optional
def test_real_typesafe_sdk_contract(program):
    sdk = pytest.importorskip("typesafe_sdk")
    params = inspect.signature(sdk.TypeSafeClient.system_one).parameters
    assert {"state", "questions", "model"}.issubset(params)
    for question in program.questions.values():
        cls = {"choice": sdk.Choice, "score": sdk.Score, "noul": sdk.Noul}[question.type]
        native = cls(**question.wire())
        assert native.model_dump(mode="json")["type"] == question.type
    sdk.RetryPolicy(max_retries=0, timeout=60)


@pytest.mark.optional
def test_real_gepa_engine_with_fake_proposer_and_mock_model(program, splits, backend):
    pytest.importorskip("gepa")
    class Teacher:
        def propose_components(self, candidate, feedback, components):
            return {key: candidate[key] for key in components}
    result, report = optimize_gepa(program, splits["train"][:3], splits["validation"][:2],
                                  backend, Teacher(), max_metric_calls=16)
    assert set(result.questions) == set(program.questions)
    assert report["engine"] == "gepa"


@pytest.mark.optional
def test_real_dspy_signature_construction_without_inference():
    pytest.importorskip("dspy")
    teacher = DSPyTeacher("openai/test-placeholder-not-a-real-model", allow_paid=True, share_feedback=True)
    assert "plan_json" in teacher.design.signature.fields
    assert "revised_components_json" in teacher.revise.signature.fields
    assert teacher.budget.used == 0


@pytest.mark.optional
def test_real_typesafe_response_schema_drives_runtime(program):
    """The adapter parses a REAL SystemOneResponse, not a hand-written dict.

    Guards the one live-only surface the other tests miss: if the SDK changes
    answer field names, key types, or usage shape, TypeSafeBackend.evaluate
    would break on the first paid call. No network or credentials are used.
    """
    sdk = pytest.importorskip("typesafe_sdk")
    from s1compiler.backends import ManagedBackend, Response
    from s1compiler.runtime import Runtime

    response = sdk.SystemOneResponse.model_validate({
        "model": program.model,
        "usage": {"input_tokens": 123, "output_tokens": 45},
        "answers": {
            "department": {"type": "choice", "choice": "billing", "confidence": 0.81,
                           "probabilities": {"billing": 0.7, "technical": 0.1, "sales": 0.1, "other": 0.1}},
            "urgent": {"type": "noul", "noul": 0.22},
            "frustration": {"type": "score", "score": 1.0, "confidence": 0.6,
                            "legend": {0: "Calm or neutral.", 1: "Clearly concerned or dissatisfied.",
                                       2: "Strongly angry or distressed."},
                            "probabilities": {0: 0.25, 1: 0.5, 2: 0.25}},
        },
    })
    # Exactly the serialization TypeSafeBackend.evaluate performs.
    data = response.model_dump(mode="json")
    assert {"answers", "model", "usage"}.issubset(data)
    replay = Response(answers=data["answers"], model=data["model"], usage=data.get("usage") or {})

    class ReplayProvider:
        identity, synthetic = "typesafe-sdk/0.7.0", False
        def evaluate(self, program, state):
            return replay
        def close(self):
            pass

    backend = ManagedBackend(ReplayProvider(), max_calls=4, cache=None)
    result = Runtime(program, backend).run({"message": "I was charged twice.", "customer_plan": "team"})
    decisions = result["decisions"]
    assert result["synthetic"] is False and result["model"] == program.model
    # Distinct typed values must stay distinct: Choice probability, Noul P(true), Score expectation.
    assert decisions["department"]["value"] == "billing"
    assert decisions["department"]["vendor_confidence"] == pytest.approx(0.81)
    assert decisions["urgent"]["p_true"] == pytest.approx(0.22)
    assert decisions["urgent"]["vendor_confidence"] is None
    assert decisions["frustration"]["value"] == pytest.approx(1.0)
    # JSON serialization turns the SDK's int-keyed Score maps into string keys.
    assert set(decisions["frustration"]["probabilities"]) == {"0", "1", "2"}


@pytest.mark.optional
def test_real_dspy_teacher_omits_sampling_params_by_default():
    """Current reasoning models accept no temperature but 1.0.

    LiteLLM raises UnsupportedParamsError for any other value (drop_params is
    False), so a hardcoded teacher temperature makes those models unusable. The
    adapter must send no sampling parameter unless the operator picked one.
    """
    pytest.importorskip("dspy")
    from litellm.utils import get_optional_params

    default = DSPyTeacher("anthropic/placeholder-not-real", allow_paid=True, share_feedback=True)
    # DSPy always stores the key; None is what keeps it out of the request body.
    assert default.lm.kwargs.get("temperature") is None
    assert "top_p" not in default.lm.kwargs and "top_k" not in default.lm.kwargs
    # The contract that actually matters: nothing reaches the provider payload.
    for model in ("claude-opus-5", "claude-sonnet-5", "claude-opus-4-6"):
        sent = get_optional_params(model=model, custom_llm_provider="anthropic",
                                   temperature=default.lm.kwargs.get("temperature"))
        assert "temperature" not in sent, f"temperature leaked into {model} request"

    explicit = DSPyTeacher("anthropic/placeholder-not-real", allow_paid=True, share_feedback=True,
                           temperature=0.3)
    assert explicit.lm.kwargs["temperature"] == 0.3


@pytest.mark.optional
def test_real_dspy_teacher_rejects_out_of_range_temperature():
    pytest.importorskip("dspy")
    from s1compiler.errors import ConfigurationError
    with pytest.raises(ConfigurationError):
        DSPyTeacher("anthropic/placeholder-not-real", allow_paid=True, share_feedback=True,
                    temperature=2.5)
