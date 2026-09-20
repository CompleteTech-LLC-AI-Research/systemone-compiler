"""Research-only teacher metering. Keeps metadata, never raw prompts or responses."""
from __future__ import annotations

import math

from .architect import DSPyTeacher
from .backends import Budget
from .errors import BackendError


class AuditedDSPyTeacher(DSPyTeacher):
    """Instrument the synchronous DSPy LM boundary used by this compiler.

    DSPy may issue multiple LM requests for one signature (e.g. adapter fallback).
    Meter them separately, disable history, and enforce the declared response ID.
    Forward counts are LM invocations; SDK retries remain disabled by the parent.
    """
    def __init__(self, model, *, expected_response_model, max_provider_calls, **kwargs):
        super().__init__(model, **kwargs)
        self.expected_response_model = expected_response_model
        self.provider_budget = Budget(max_provider_calls)
        self.observed_models = set()
        self.input_tokens, self.output_tokens = 0, 0
        self.usage_unknown_calls, self.successful_calls = 0, 0
        self.sdk_estimated_cost, self.cost_unknown_calls = 0., 0
        forward = self.lm.forward

        def metered_forward(*args, **call_kwargs):
            self.provider_budget.reserve()
            response = forward(*args, **call_kwargs)
            self.observe(response)
            return response

        self.lm.forward = metered_forward

    def observe(self, response):
        model = getattr(response, "model", None)
        if isinstance(model, str):
            self.observed_models.add(model)
        self.successful_calls += 1
        usage = dict(getattr(response, "usage", {}) or {})
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if type(prompt) is int and prompt >= 0 and type(completion) is int and completion >= 0:
            self.input_tokens += prompt
            self.output_tokens += completion
        else:
            self.usage_unknown_calls += 1
        estimated = getattr(response, "_hidden_params", {}).get("response_cost")
        if type(estimated) in (int, float) and math.isfinite(estimated) and estimated >= 0:
            self.sdk_estimated_cost += estimated
        else:
            self.cost_unknown_calls += 1
        if model != self.expected_response_model:
            raise BackendError("Teacher response model differs from the preregistered identity.")

    def _predict(self, predictor, **kwargs):
        self.budget.reserve()
        with self.dspy.context(lm=self.lm, disable_history=True):
            return predictor(**kwargs)

    def accounting(self):
        result = super().accounting()
        missing_responses = self.provider_budget.used-self.successful_calls
        result.update(expected_response_model=self.expected_response_model,
            observed_response_models=sorted(self.observed_models),
            provider_requests_attempted=self.provider_budget.used,
            provider_request_limit=self.provider_budget.maximum,
            provider_responses_received=self.successful_calls,
            reported_input_tokens=self.input_tokens, reported_output_tokens=self.output_tokens,
            usage_unknown_calls=self.usage_unknown_calls+missing_responses,
            sdk_estimated_cost=self.sdk_estimated_cost if self.successful_calls and not (self.cost_unknown_calls+missing_responses) else None,
            partial_sdk_estimated_cost=self.sdk_estimated_cost,
            cost_unknown_calls=self.cost_unknown_calls+missing_responses,
            dollar_cost=None,
            budget_note="Separate signature and synchronous LM-forward request ceilings; SDK retries disabled. "
                        "Token counts are provider-reported where available. SDK cost estimates are not bills. "
                        "No hard aggregate token or dollar cap; no raw DSPy history retained.")
        return result
