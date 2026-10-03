"""Optional compile-time native adapter, qualified against DSPy 3.4.0 only.

Frozen programs keep using TypeSafeBackend. Importing this module does not import
DSPy or the SDK. ManagedBackend owns all budget, cache and attempt accounting.
"""
from __future__ import annotations

import importlib.metadata
import os
import re
import time
from typing import Any

from .backends import Response, is_transient_sdk_error
from .errors import BackendError, ConfigurationError
from .models import Program


class DSPyTypeSafeBackend:
    identity = "dspy-typesafe/3.4.0+typesafe-sdk/0.7.0"
    synthetic = False

    def __init__(self, *, allow_paid: bool = False, timeout: float = 60.0):
        if not allow_paid:
            raise ConfigurationError("Live calls require explicit allow_paid=True / --allow-paid.")
        if not os.getenv("TYPESAFE_API_KEY"):
            raise ConfigurationError("Set TYPESAFE_API_KEY in your local environment; never paste it into chat.")
        try:
            versions = {name: importlib.metadata.version(name) for name in ("dspy", "typesafe-sdk")}
        except importlib.metadata.PackageNotFoundError as exc:
            raise ConfigurationError("Install DSPy 3.4.0 and typesafe-sdk 0.7.0 for compile-time evaluation.") from exc
        if versions != {"dspy": "3.4.0", "typesafe-sdk": "0.7.0"}:
            raise ConfigurationError("Experimental TypeSafe adapter requires dspy==3.4.0 and typesafe-sdk==0.7.0.")
        import dspy
        from dspy.experimental import TypeSafe
        from typesafe_sdk import RetryPolicy

        # Upstream private hooks are qualified by the exact version above. Preserve
        # raw native dictionaries and the entire SDK JSON envelope: upstream's
        # normal decoding drops Score legend and its public result drops identity.
        class NativeEnvelope(TypeSafe):
            def _sdk_kwargs(self):
                return {**super()._sdk_kwargs(),
                        "retry": RetryPolicy(max_retries=0, timeout=self.timeout)}

            @staticmethod
            def _response(response):
                data = response.model_dump(mode="json")
                if not isinstance(data, dict) or "answers" not in data or "model" not in data:
                    raise BackendError("TypeSafe response lacks answers or model; refusing it.")
                return data

            def _finish(self, request, response, cache_hit):
                # No DSPy history, usage tracker or shared cache. The surrounding
                # ManagedBackend validates identity and settles native receipts.
                if cache_hit:
                    raise BackendError("Unexpected upstream cache hit; refusing it.")
                return response

        self._dspy = dspy
        self._client_type = NativeEnvelope
        self.timeout = timeout

    def evaluate(self, program: Program, state: dict[str, Any]) -> Response:
        if not re.fullmatch(r"jev-\d+\.\d+\.\d+", program.model):
            raise ConfigurationError("Use a versioned model ID, not an alias, for measured execution.")
        client = self._client_type(model=program.model, timeout=self.timeout, cache=False, callbacks=[])
        start = time.perf_counter()
        try:
            with self._dspy.context(disable_history=True, usage_tracker=None, callbacks=[]):
                data = client(state=state, questions={key: q.wire() for key, q in program.questions.items()})
        except BackendError:
            raise
        except Exception as exc:
            failure = BackendError(
                f"TypeSafe request failed ({type(exc).__name__}); check local credentials, quota, and SDK compatibility."
            )
            failure.transient = is_transient_sdk_error(exc)
            raise failure from exc
        return Response(answers=data["answers"], model=data["model"], usage=data.get("usage") or {},
                        latency_ms=(time.perf_counter() - start) * 1000)

    def close(self):
        # Upstream owns a context-managed SDK client per native attempt.
        pass
