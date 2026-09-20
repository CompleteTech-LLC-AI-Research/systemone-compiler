import types

import pytest

from s1compiler.errors import BackendError, BudgetExceeded
from s1compiler.research_teacher import AuditedDSPyTeacher


@pytest.fixture
def teacher(monkeypatch):
    dspy = pytest.importorskip("dspy")
    from litellm import ModelResponse
    def forward(self, *args, **kwargs):
        return ModelResponse(model="deepseek-flash", usage={"prompt_tokens": 10, "completion_tokens": 5},
            choices=[{"message": {"role": "assistant", "content":
                '[[ ## revised_components_json ## ]]\n{"flag/instructions":"Is the claim supported?"}\n[[ ## completed ## ]]'},
                "finish_reason": "stop"}])
    monkeypatch.setattr(dspy.LM, "forward", forward)
    return AuditedDSPyTeacher("deepseek/deepseek-flash", expected_response_model="deepseek-flash",
        max_provider_calls=2, allow_paid=True, share_feedback=True, max_calls=2)


@pytest.mark.optional
def test_real_dspy_boundary_meters_fake_provider_and_disables_history(teacher):
    result = teacher.propose_components({"flag/instructions": '"Check this"'}, {}, ["flag/instructions"])
    assert result["flag/instructions"] == '"Is the claim supported?"'
    account = teacher.accounting()
    assert account["signature_calls"] == account["provider_requests_attempted"] == 1
    assert account["reported_input_tokens"] == 10 and account["reported_output_tokens"] == 5
    assert account["observed_response_models"] == ["deepseek-flash"]
    assert teacher.lm.history == [] and teacher.revise.history == []
    assert account["dollar_cost"] is None


@pytest.mark.optional
def test_teacher_identity_mismatch_is_failure_not_bad_candidate(teacher):
    with pytest.raises(BackendError, match="preregistered identity"):
        teacher.observe(types.SimpleNamespace(model="different-model", usage={}, _hidden_params={}))


@pytest.mark.optional
def test_teacher_provider_budget_is_separate_from_signature_budget(teacher):
    teacher.provider_budget.maximum = 1
    parts = {"flag/instructions": '"Check this"'}
    teacher.propose_components(parts, {}, list(parts))
    with pytest.raises(BudgetExceeded):
        teacher.propose_components(parts, {}, list(parts))
    assert teacher.accounting()["signature_calls"] == 2
    assert teacher.accounting()["provider_requests_attempted"] == 1
