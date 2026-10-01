from __future__ import annotations
import copy
import math
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

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
    attempt_receipt: int | None = None  # Local reservation ID; never serialized into cache values.

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
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self):
        if self.maximum < 1:
            raise ConfigurationError("Call budget must be positive.")

    def reserve(self):
        with self._lock:
            if self.used >= self.maximum:
                raise BudgetExceeded(f"Call budget exhausted ({self.used}/{self.maximum}).")
            self.used += 1


class AttemptLedger:
    """Charged, per-graph native reservations; unsettled calls remain charged."""

    def __init__(self, *, graph_sha256: str, maximum: int, nodes: set[str],
                 node_limits: dict[str, int] | None = None, policy: dict[str, Any] | None = None):
        if type(maximum) is not int or maximum < 1 or not nodes:
            raise ConfigurationError("Graph attempt limit and node set must be positive.")
        self.graph_sha256, self.maximum = graph_sha256, maximum
        self.nodes = frozenset(nodes)
        self.node_limits = dict(node_limits or {})
        self.policy = copy.deepcopy(policy or {})
        if not isinstance(self.policy, dict):
            raise ConfigurationError("Attempt ledger policy must be a JSON object.")
        canonical(self.policy)
        if not set(self.node_limits) <= self.nodes or any(
            type(limit) is not int or not 1 <= limit <= maximum for limit in self.node_limits.values()
        ):
            raise ConfigurationError("Node attempt limits must be positive and within the graph limit.")
        self._lock = threading.Lock()
        self._receipts: list[dict[str, Any]] = []
        self.on_change: Callable[[dict[str, Any]], None] | None = None

    def _snapshot_unlocked(self) -> dict[str, Any]:
        return {"format": "systemone-attempt-ledger/v1", "graph_sha256": self.graph_sha256,
                "maximum": self.maximum, "nodes": sorted(self.nodes),
                "node_limits": dict(self.node_limits), "policy": copy.deepcopy(self.policy),
                "used": len(self._receipts), "receipts": copy.deepcopy(self._receipts)}

    def admit(self, budget: Budget, node_id: str) -> int:
        """Reserve graph and owner budgets atomically before native dispatch."""
        with self._lock:
            if node_id not in self.nodes:
                raise ConfigurationError("Attempt reservation names an unknown graph node.")
            if len(self._receipts) >= self.maximum:
                raise BudgetExceeded(f"Graph attempt limit exhausted ({len(self._receipts)}/{self.maximum}).")
            used_by_node = sum(receipt["node_id"] == node_id for receipt in self._receipts)
            if used_by_node >= self.node_limits.get(node_id, self.maximum):
                raise BudgetExceeded(f"Node attempt limit exhausted: {node_id}.")
            budget.reserve()  # Failure leaves the graph ledger unchanged.
            receipt_id = len(self._receipts) + 1
            self._receipts.append({"id": receipt_id, "node_id": node_id, "status": "in_flight"})
            if self.on_change is not None:
                self.on_change(self._snapshot_unlocked())
            return receipt_id

    def mark(self, receipt_id: int, status: str) -> None:
        if status not in {"response_received", "validated", "invalid_response",
                          "identity_rejected", "uncertain"}:
            raise ConfigurationError("Invalid attempt receipt status.")
        with self._lock:
            if not 1 <= receipt_id <= len(self._receipts):
                raise ConfigurationError("Unknown attempt receipt.")
            receipt = self._receipts[receipt_id - 1]
            allowed = {"in_flight": {"response_received", "uncertain"},
                       "response_received": {"validated", "invalid_response", "identity_rejected", "uncertain"}}
            if status not in allowed.get(receipt["status"], set()):
                raise ConfigurationError("Attempt receipt is already settled.")
            receipt["status"] = status
            if self.on_change is not None:
                self.on_change(self._snapshot_unlocked())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def status(self, receipt_id: int | None) -> str | None:
        if receipt_id is None:
            return None
        with self._lock:
            if not 1 <= receipt_id <= len(self._receipts):
                raise ConfigurationError("Unknown attempt receipt.")
            return self._receipts[receipt_id - 1]["status"]

    @classmethod
    def restore(cls, data: dict[str, Any], *, graph_sha256: str, maximum: int,
                nodes: set[str], node_limits: dict[str, int] | None = None,
                policy: dict[str, Any] | None = None) -> AttemptLedger:
        ledger = cls(graph_sha256=graph_sha256, maximum=maximum, nodes=nodes,
                     node_limits=node_limits, policy=policy)
        if not isinstance(data, dict) or set(data) != {
            "format", "graph_sha256", "maximum", "nodes", "node_limits", "policy", "used", "receipts"
        } or data["format"] != "systemone-attempt-ledger/v1" or data["graph_sha256"] != graph_sha256 or (
            data["maximum"] != maximum or data["nodes"] != sorted(nodes) or
            data["node_limits"] != ledger.node_limits or data["policy"] != ledger.policy
        ):
            raise ConfigurationError("Attempt ledger identity or limits differ from the frozen run.")
        receipts = data["receipts"]
        if not isinstance(receipts, list) or type(data["used"]) is not int or data["used"] != len(receipts) or (
            len(receipts) > maximum
        ):
            raise ConfigurationError("Attempt ledger receipt count is invalid.")
        for index, receipt in enumerate(receipts, 1):
            if not isinstance(receipt, dict) or set(receipt) != {"id", "node_id", "status"} or (
                type(receipt["id"]) is not int or receipt["id"] != index or receipt["node_id"] not in nodes or
                receipt["status"] not in {"in_flight", "response_received", "validated",
                                          "invalid_response", "identity_rejected", "uncertain"}
            ):
                raise ConfigurationError("Attempt ledger contains an invalid receipt.")
        for node_id, limit in ledger.node_limits.items():
            if sum(receipt["node_id"] == node_id for receipt in receipts) > limit:
                raise ConfigurationError("Attempt ledger exceeds a frozen node limit.")
        ledger._receipts = copy.deepcopy(receipts)
        for receipt in ledger._receipts:
            if receipt["status"] in {"in_flight", "response_received"}:
                receipt["status"] = "uncertain"
        return ledger


_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_TRANSIENT_NAMES = ("timeout", "connect", "ratelimit", "unavailable", "overloaded")


def is_transient_sdk_error(error: BaseException) -> bool:
    """True for timeouts, connection loss, rate limits, and 5xx-style SDK failures.

    SDK exceptions rarely subclass the builtins, so also match the status code and class names.
    Authentication, quota, validation, and unknown failures are never transient.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        status = getattr(current, "status_code", None)
        if status is None:
            status = getattr(getattr(current, "response", None), "status_code", None)
        if type(status) is int and status in _TRANSIENT_STATUS:
            return True
        if any(any(token in cls.__name__.lower() for token in _TRANSIENT_NAMES)
               for cls in type(current).__mro__ if cls not in (Exception, BaseException, object)):
            return True
        current = current.__cause__ or current.__context__
    return False


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
            failure = BackendError(
                f"TypeSafe request failed ({type(exc).__name__}); check local credentials, quota, and SDK compatibility."
            )
            failure.transient = is_transient_sdk_error(exc)
            raise failure from exc
        # Use documented Pydantic response serialization. Never capture raw headers.
        data = response.model_dump(mode="json")
        if not isinstance(data, dict) or "answers" not in data or "model" not in data:
            raise BackendError("TypeSafe response lacks answers or model; refusing it.")
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
        self._lock = threading.RLock()
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

    def evaluate(self, program: Program, state: dict[str, Any], *,
                 ledger: AttemptLedger | None = None, node_id: str | None = None,
                 before_admission=None) -> Response:
        payload = {"model": program.model, "state": state,
                   "questions": {key: q.wire() for key, q in program.questions.items()}}
        if len(canonical(payload)) > self.max_request_chars:
            raise ConfigurationError("Request exceeds the configured character limit; project less state.")
        # Pin target versions. An alias can silently invalidate prompt/calibration results.
        if program.model in {"jev", "jev-latest", "jev-preview"}:
            raise ConfigurationError("Use a versioned model ID, not an alias, for measured execution.")
        key = fingerprint({"provider": self.identity, "wire_format": 1, "payload": payload})
        with self._lock:  # Graph scheduling stays serial; this also owns SQLite connection access.
            response = self.cache.get(key) if self.cache else None
            if response is not None:
                self._validate_identity(program, response)
                self.models_seen.add(response.model)
                self.cache_hits += 1
                response.cached = True
                response.latency_ms = 0.0
                response.usage = {"input_tokens": 0, "output_tokens": 0}
                return response
            if before_admission is not None:
                before_admission()
            if ledger is not None and node_id is None:
                raise ConfigurationError("Graph admission requires a node ID.")
            receipt = ledger.admit(self.budget, node_id) if ledger else None
            if ledger is None:
                self.budget.reserve()
            try:
                response = self.backend.evaluate(program, state)
            except Exception:
                if ledger is not None:
                    ledger.mark(receipt, "uncertain")
                self.usage_unknown_calls += 1
                raise
            if ledger is not None:
                ledger.mark(receipt, "response_received")
            try:
                self._validate_identity(program, response)
            except BackendError:
                if ledger is not None:
                    ledger.mark(receipt, "identity_rejected")
                self.usage_unknown_calls += 1
                raise
            self.models_seen.add(response.model)
            usage = response.usage
            if usage.get("input_tokens") is None or usage.get("output_tokens") is None:
                self.usage_unknown_calls += 1
            self.input_tokens += usage.get("input_tokens") or 0
            self.output_tokens += usage.get("output_tokens") or 0
            response.attempt_receipt = receipt
            # Runtime validates answers before they are trusted. Cache stores raw data
            # only after Runtime calls remember_validated below.
            return response

    def _validate_identity(self, program: Program, response: Response):
        if response.model != program.model:
            raise BackendError("Returned model differs from the pinned target; refusing a mixed-model run.")
        if response.synthetic != self.synthetic:
            raise BackendError("Response synthetic/live identity differs from the configured backend.")

    def remember_validated(self, program: Program, state: dict[str, Any], response: Response,
                           *, ledger: AttemptLedger | None = None):
        with self._lock:
            if self.cache and not response.cached:
                payload = {"model": program.model, "state": state,
                           "questions": {key: q.wire() for key, q in program.questions.items()}}
                key = fingerprint({"provider": self.identity, "wire_format": 1, "payload": payload})
                self.cache.put(key, copy.deepcopy(response))
            if ledger is not None and response.attempt_receipt is not None:
                ledger.mark(response.attempt_receipt, "validated")

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
