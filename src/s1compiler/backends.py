from __future__ import annotations
import copy
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .errors import BackendError, BudgetExceeded, ConfigurationError
from .io import canonical, fingerprint, json_loads
from .models import Program


@dataclass
class Response:
    answers: dict[str, dict[str, Any]]
    model: str
    usage: dict[str, int | None] = field(default_factory=dict)
    latency_ms: float = 0.0
    cached: bool = False
    synthetic: bool = False

    def to_dict(self):
        return {
            "answers": self.answers, "model": self.model, "usage": self.usage,
            "latency_ms": self.latency_ms, "cached": self.cached, "synthetic": self.synthetic,
        }


class Backend(Protocol):
    identity: str
    synthetic: bool
    def evaluate(self, program: Program, state: dict[str, Any]) -> Response: ...
    def close(self) -> None: ...


@dataclass
class Budget:
    maximum: int
    used: int = 0

    def __post_init__(self):
        if self.maximum < 1:
            raise ConfigurationError("Call budget must be positive.")

    def reserve(self):
        if self.used >= self.maximum:
            raise BudgetExceeded(f"Call budget exhausted ({self.used}/{self.maximum}).")
        self.used += 1


class TypeSafeBackend:
    """SDK 0.7.0. No chat completions emulation; one batch of native questions."""
    identity = "typesafe-sdk/0.7.0"
    synthetic = False

    def __init__(self, *, allow_paid: bool = False, timeout: float = 60.0):
        if not allow_paid:
            raise ConfigurationError("Live calls require explicit allow_paid=True / --allow-paid.")
        if not os.getenv("TYPESAFE_API_KEY"):
            raise ConfigurationError("Set TYPESAFE_API_KEY in your local environment; never paste it into chat.")
        try:
            from typesafe_sdk import TypeSafeClient, RetryPolicy
        except ImportError as exc:
            raise ConfigurationError("Install the live extra: pip install -e '.[live]'.") from exc
        # Disabling SDK retries keeps the runner's request budget transparent.
        self.client = TypeSafeClient(retry=RetryPolicy(max_retries=0, timeout=timeout))

    def evaluate(self, program: Program, state: dict[str, Any]) -> Response:
        start = time.perf_counter()
        try:
            response = self.client.system_one(
                state=state,
                questions={key: q.wire() for key, q in program.questions.items()},
                model=program.model,
            )
        except Exception as exc:
            raise BackendError(
                f"TypeSafe request failed ({type(exc).__name__}); check local credentials, quota, and SDK compatibility."
            ) from exc
        # Use documented Pydantic response serialization. Never capture raw headers.
        data = response.model_dump(mode="json")
        return Response(
            answers=data["answers"], model=data["model"], usage=data.get("usage") or {},
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def close(self):
        self.client.close()


STOP = set("the a an to of is are in and or this that does it with for as be on how what has have from".split())


def words(value: Any) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", canonical(value).lower()) if len(w) > 2 and w not in STOP}


class MockBackend:
    """Deterministic lexical fixture, NOT Jev and NOT a performance simulator.

    This backend never sees gold labels. It scores overlap with criteria so that
    editing criteria genuinely changes execution, but numbers are synthetic.
    """
    identity = "mock-lexical/v1"
    synthetic = True

    def evaluate(self, program: Program, state: dict[str, Any]) -> Response:
        start = time.perf_counter()
        token_set = words(state)
        answers = {}
        for key, q in program.questions.items():
            if q.type == "noul":
                criteria = q.criteria or {}
                yes = words(criteria.get("true") or q.instructions)
                no = words(criteria.get("false") or "")
                margin = len(token_set & yes) - len(token_set & no)
                p = 1 / (1 + math.exp(-max(-20, min(20, margin))))
                answers[key] = {"type": "noul", "noul": p}
            else:
                criteria = q.criteria
                names = list(criteria) if q.type == "choice" else [str(i) for i in range(len(criteria))]
                texts = list(criteria.values()) if q.type == "choice" else criteria
                logits = [1.4 * len(token_set & words(t)) for t in texts]
                maximum = max(logits)
                weights = [math.exp(v - maximum) for v in logits]
                probs = {n: w / sum(weights) for n, w in zip(names, weights)}
                entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
                # This is ONLY the fixture's confidence definition, not TypeSafe's.
                confidence = max(0.0, 1 - entropy / math.log(len(names)))
                if q.type == "choice":
                    answers[key] = {"type": "choice", "choice": max(probs, key=probs.get),
                                    "probabilities": probs, "confidence": confidence}
                else:
                    answers[key] = {"type": "score", "score": sum(int(i) * p for i, p in probs.items()),
                                    "probabilities": probs, "confidence": confidence,
                                    "legend": {str(i): t for i, t in enumerate(texts)}}
        return Response(answers, program.model, {"input_tokens": None, "output_tokens": None},
                        (time.perf_counter() - start) * 1000, synthetic=True)

    def close(self):
        pass


class AnswerCache:
    """Optional local SQLite cache. Stores outputs, not state text or credentials.

    Input fingerprints are not anonymization. Treat cache files as sensitive.
    Disk cache use is explicit; in-memory cache is the default.
    """
    def __init__(self, path: str | Path | None = None, ttl_seconds: float = 86400):
        self.ttl = ttl_seconds
        if self.ttl <= 0:
            raise ConfigurationError("Cache TTL must be positive.")
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path) if path else ":memory:")
        if path:
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        self.db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, ts REAL, value TEXT)")

    def get(self, key: str):
        row = self.db.execute("SELECT ts,value FROM answers WHERE key=?", (key,)).fetchone()
        if row is None or time.time() - row[0] > self.ttl:
            return None
        return Response(**json_loads(row[1]))

    def put(self, key: str, response: Response):
        self.db.execute("INSERT OR REPLACE INTO answers VALUES (?,?,?)",
                        (key, time.time(), canonical(response.to_dict())))
        self.db.commit()

    def close(self):
        self.db.close()


class ManagedBackend:
    """Budget, cache, payload limit, model drift checks, and measured accounting."""
    def __init__(self, backend: Backend, *, max_calls: int = 500,
                 cache: AnswerCache | None = None, max_request_chars: int = 120000):
        self.backend = backend
        self.budget = Budget(max_calls)
        self.cache = cache
        self.identity = backend.identity
        self.synthetic = backend.synthetic
        self.max_request_chars = max_request_chars
        self.cache_hits = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.usage_unknown_calls = 0
        self.models_seen: set[str] = set()

    def evaluate(self, program: Program, state: dict[str, Any]) -> Response:
        payload = {"model": program.model, "state": state,
                   "questions": {key: q.wire() for key, q in program.questions.items()}}
        if len(canonical(payload)) > self.max_request_chars:
            raise ConfigurationError("Request exceeds the configured character limit; project less state.")
        # Pin target versions. An alias can silently invalidate prompt/calibration results.
        if program.model in {"jev", "jev-latest", "jev-preview"}:
            raise ConfigurationError("Use a versioned model ID, not an alias, for measured execution.")
        key = fingerprint({"provider": self.identity, "wire_format": 1, "payload": payload})
        response = self.cache.get(key) if self.cache else None
        if response is not None:
            self._validate_identity(program, response)
            self.models_seen.add(response.model)
            self.cache_hits += 1
            response.cached = True
            response.latency_ms = 0.0
            response.usage = {"input_tokens": 0, "output_tokens": 0}
            return response
        self.budget.reserve()
        response = self.backend.evaluate(program, state)
        self._validate_identity(program, response)
        self.models_seen.add(response.model)
        usage = response.usage
        if usage.get("input_tokens") is None or usage.get("output_tokens") is None:
            self.usage_unknown_calls += 1
        self.input_tokens += usage.get("input_tokens") or 0
        self.output_tokens += usage.get("output_tokens") or 0
        # Runtime validates answers before they are trusted. Cache stores raw data
        # only after Runtime calls remember_validated below.
        return response

    def _validate_identity(self, program: Program, response: Response):
        if response.model != program.model:
            raise BackendError("Returned model differs from the pinned target; refusing a mixed-model run.")
        if response.synthetic != self.synthetic:
            raise BackendError("Response synthetic/live identity differs from the configured backend.")

    def remember_validated(self, program: Program, state: dict[str, Any], response: Response):
        if self.cache and not response.cached:
            payload = {"model": program.model, "state": state,
                       "questions": {key: q.wire() for key, q in program.questions.items()}}
            key = fingerprint({"provider": self.identity, "wire_format": 1, "payload": payload})
            self.cache.put(key, copy.deepcopy(response))

    def accounting(self):
        return {"backend": self.identity, "synthetic": self.synthetic,
                "requests_attempted": self.budget.used, "request_limit": self.budget.maximum,
                "cache_hits": self.cache_hits, "reported_input_tokens": self.input_tokens,
                "reported_output_tokens": self.output_tokens,
                "usage_unknown_calls": self.usage_unknown_calls, "models_seen": sorted(self.models_seen),
                "dollar_cost": None}

    def close(self):
        self.backend.close()
        if self.cache:
            self.cache.close()
