"""Compile-time DSPy request boundary; importing this module does not import DSPy."""
from __future__ import annotations

import inspect

from .errors import BackendError, BudgetExceeded


class _OwnedMeteringFailure(Exception):
    """Marks only our admission and response-check failures across DSPy's boundary."""


def owned_metering_failure(error: Exception) -> Exception | None:
    """Recover our errors, never a similarly named provider or adapter exception.

    DSPy 3.4 translates engine failures into LMUnexpectedError. Return a fresh
    typed error so the caller can preserve that entire causal chain without
    introducing an exception-cause cycle.
    """
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _OwnedMeteringFailure) and isinstance(
            current.__cause__, (BackendError, BudgetExceeded)
        ):
            original = current.__cause__
            return type(original)(*original.args)
        current = current.__cause__
    return None


def make_metered_lm(dspy, model, *, budget, observe, **kwargs):
    """Use a one-attempt engine on 3.4, with the supported 3.3 legacy branch."""
    if "engine" not in inspect.signature(dspy.LM.__init__).parameters:
        lm = dspy.LM(model, **kwargs)
        forward = lm.forward
        response_failure = None

        def metered_forward(*args, **call_kwargs):
            nonlocal response_failure
            # Legacy adapters retry arbitrary errors. A failed identity check is
            # terminal for this teacher and must never dispatch a fallback.
            if response_failure is not None:
                raise response_failure
            budget.reserve()
            response = forward(*args, **call_kwargs)
            try:
                observe(response)
            except BackendError as exc:
                response_failure = exc
                raise
            return response

        lm.forward = metered_forward
        return lm

    # Optional dependency stays behind teacher construction. Custom engines own
    # connections; DSPy rejects these settings on the outer LM.
    from dspy.clients.engines.litellm_engine import LiteLLMEngine

    client_options = {key: kwargs.pop(key) for key in ("api_base", "api_key", "timeout") if key in kwargs}

    class ObservedLiteLLMEngine(LiteLLMEngine):
        def _response(self, raw, request):
            # Canonical conversion retains model/usage but drops LiteLLM's cost
            # metadata. Inspect the raw response before that conversion; retain
            # only the observer's counters, never the response or its content.
            try:
                observe(raw)
            except BackendError as exc:
                raise _OwnedMeteringFailure("Teacher response check failed.") from exc
            return super()._response(raw, request)

    class MeteredEngine:
        # Do not inherit LiteLLMEngine here: its complete_legacy transition hook
        # bypasses complete() for ordinary Predict calls in DSPy 3.4.
        def __init__(self):
            self.delegate = ObservedLiteLLMEngine(**client_options)

        def complete(self, request):
            try:
                budget.reserve()
            except BudgetExceeded as exc:
                raise _OwnedMeteringFailure("Teacher provider request ceiling exhausted.") from exc
            return self.delegate.complete(request)

        def close(self):
            self.delegate.close()

    kwargs["engine"] = MeteredEngine()
    return dspy.LM(model, **kwargs)
