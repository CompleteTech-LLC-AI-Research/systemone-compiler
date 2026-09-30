"""Deterministic execution of validated, frozen hierarchy artifacts."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import time
from typing import Any, Callable

from .backends import AttemptLedger, ManagedBackend
from .errors import BackendError, BudgetExceeded, ConfigurationError, DataError
from .hierarchy import (Candidate, Condition, HierarchyArtifact, LoweredExport,
                        LoweredNode, Reference, RootRef)
from .hierarchy_validation import validate_hierarchy_artifact
from .models import project_state
from .runtime import Runtime


class _MissingReference(Exception):
    pass


class _Cancelled(Exception):
    pass


class _StageFailure(Exception):
    def __init__(self, error: Exception):
        self.error = error


@dataclass(frozen=True)
class GraphRetryPolicy:
    """Explicit, bounded graph retries; defaults make no repeat requests."""

    max_transient_retries: int = 0
    max_invalid_response_retries: int = 0
    delay_seconds: float = 0.0

    def __post_init__(self):
        if type(self.max_transient_retries) is not int or not 0 <= self.max_transient_retries <= 3 or (
            type(self.max_invalid_response_retries) is not int or not 0 <= self.max_invalid_response_retries <= 3
        ) or not isinstance(self.delay_seconds, (int, float)) or not math.isfinite(self.delay_seconds) or (
            not 0 <= self.delay_seconds <= 15
        ):
            raise ConfigurationError("Graph retry counts must be 0–3 and delay must be 0–15 seconds.")


class _NodeBackend:
    """Bind one leaf to the caller's shared backend and attempt ledger."""

    def __init__(self, backend: ManagedBackend, ledger: AttemptLedger, node_id: str,
                 before_admission: Callable[[], None]):
        self.backend, self.ledger, self.node_id = backend, ledger, node_id
        self.before_admission = before_admission
        self.synthetic = backend.synthetic
        self.last_receipt: int | None = None

    def evaluate(self, program, state):
        response = self.backend.evaluate(program, state, ledger=self.ledger,
                                         node_id=self.node_id, before_admission=self.before_admission)
        self.last_receipt = response.attempt_receipt
        return response

    def remember_validated(self, program, state, response):
        self.backend.remember_validated(program, state, response, ledger=self.ledger)


class HierarchyRuntime:
    """Run one self-contained graph through the caller's existing managed backend."""

    def __init__(self, artifact: HierarchyArtifact, backend: ManagedBackend, *,
                 enforce_release: bool = False, max_graph_attempts: int | None = None,
                 node_attempt_limits: dict[str, int] | None = None,
                 retry_policy: GraphRetryPolicy | None = None):
        # A snapshot prevents caller mutations after validation from changing a run.
        self.artifact = HierarchyArtifact.model_validate(artifact.model_dump(mode="json"))
        validate_hierarchy_artifact(self.artifact)
        self.backend = backend
        self.enforce_release = enforce_release
        self.max_graph_attempts = (self.artifact.limits.max_native_calls_per_example
                                   if max_graph_attempts is None else max_graph_attempts)
        if type(self.max_graph_attempts) is not int or not 1 <= self.max_graph_attempts <= (
            self.artifact.limits.max_native_calls_per_example
        ):
            raise ConfigurationError("Graph attempt limit must be positive and no looser than the artifact limit.")
        self.node_attempt_limits = dict(node_attempt_limits or {})
        self.retry_policy = retry_policy or GraphRetryPolicy()
        self.node_ids = {node.id for node in self.artifact.nodes}
        AttemptLedger(graph_sha256=self.artifact.content_hash, maximum=self.max_graph_attempts,
                      nodes=self.node_ids, node_limits=self.node_attempt_limits,
                      policy=asdict(self.retry_policy))
        if enforce_release and not backend.synthetic and self.artifact.provenance.status != "measured":
            raise ConfigurationError("Hierarchy composition is not measured; use experimental mode.")
        if any(node.program.model != self.artifact.source.model for node in self.artifact.nodes):
            raise ConfigurationError("Hierarchy leaf model differs from the pinned root model.")

    @classmethod
    def load(cls, path: str | Path, backend: ManagedBackend, **kwargs) -> HierarchyRuntime:
        return cls(HierarchyArtifact.load(path), backend, **kwargs)

    def run(self, state: dict[str, Any], *, cancel_requested: Callable[[], bool] | None = None,
            timeout_seconds: float | None = None, ledger: AttemptLedger | None = None) -> dict[str, Any]:
        started = time.monotonic()
        if timeout_seconds is not None and (not isinstance(timeout_seconds, (int, float)) or
                                            not math.isfinite(timeout_seconds) or
                                            not 0 < timeout_seconds <= 86400):
            raise ConfigurationError("Graph timeout must be between zero and one day.")
        deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
        if ledger is None:
            ledger = AttemptLedger(graph_sha256=self.artifact.content_hash, maximum=self.max_graph_attempts,
                                   nodes=self.node_ids, node_limits=self.node_attempt_limits,
                                   policy=asdict(self.retry_policy))
        else:
            expected = (self.artifact.content_hash, self.max_graph_attempts,
                        self.node_ids, self.node_attempt_limits, asdict(self.retry_policy))
            actual = (ledger.graph_sha256, ledger.maximum, set(ledger.nodes), ledger.node_limits, ledger.policy)
            if actual != expected or self.backend.budget.used < ledger.snapshot()["used"]:
                raise ConfigurationError("Resumed graph ledger or owner budget differs from the frozen run.")
        root_state = copy.deepcopy(project_state(self.artifact.source.state, state))
        values: dict[str, dict[str, dict[str, Any]]] = {}
        stages: dict[str, dict[str, Any]] = {}
        executed: list[str] = []
        path: list[str] = []
        before_calls = self.backend.budget.used
        before_receipts = ledger.snapshot()["used"]
        before_hits = self.backend.cache_hits
        before_unknown = self.backend.usage_unknown_calls
        before_input = self.backend.input_tokens
        before_output = self.backend.output_tokens
        failure: Exception | None = None
        status = "review_required"
        decisions: dict[str, dict[str, Any]] = {}
        try:
            decisions, review = self._run_scope("", self.artifact.final, root_state,
                                                values, stages, executed, path, cancel_requested,
                                                deadline, ledger)
            status = "review_required" if review or set(decisions) != set(self.artifact.source.decisions) else "completed"
            if self._cancelled(cancel_requested, deadline):
                status = "cancelled"
        except _Cancelled:
            status = "cancelled"
        except _StageFailure as exc:
            status, failure = "failed", exc.error
        for stage in [*self.artifact.nodes, *self.artifact.exports]:
            stages.setdefault(stage.id, {"status": "cancelled" if status == "cancelled" else "pending",
                                         "reason": "cancelled" if status == "cancelled" else "not_reached"})
        unknown_usage = self.backend.usage_unknown_calls - before_unknown
        ledger_snapshot = ledger.snapshot()
        attempted_nodes = [receipt["node_id"] for receipt in ledger_snapshot["receipts"][before_receipts:]]
        retries = len(attempted_nodes) - len(set(attempted_nodes))
        result = {
            "format": "systemone-hierarchy-result/v1",
            "graph_sha256": self.artifact.content_hash,
            "model": self.artifact.source.model,
            "synthetic": self.backend.synthetic,
            "status": status,
            "decisions": decisions,
            "stages": stages,
            "executed": executed,
            "path": path,
            "accounting": {"requests_attempted": self.backend.budget.used - before_calls,
                           "graph_executions": 1,
                           "leaf_evaluations": len(set(attempted_nodes)) + self.backend.cache_hits - before_hits,
                           "cache_hits": self.backend.cache_hits - before_hits,
                           "retries": retries,
                           "latency_ms": (time.monotonic() - started) * 1000,
                           "usage_unknown_calls": unknown_usage,
                           "reported_input_tokens": None if unknown_usage else self.backend.input_tokens - before_input,
                           "reported_output_tokens": None if unknown_usage else self.backend.output_tokens - before_output,
                           "dollar_cost": None,
                           "predicted_worst_case_leaf_calls": len(self.artifact.nodes),
                           "attempt_ledger": ledger_snapshot},
        }
        if failure is not None:
            result["error"] = {"type": type(failure).__name__, "message": str(failure)}
        return result

    @staticmethod
    def _cancelled(cancel_requested: Callable[[], bool] | None, deadline: float | None) -> bool:
        return bool((cancel_requested and cancel_requested()) or (deadline is not None and time.monotonic() >= deadline))

    def _check_cancel(self, cancel_requested: Callable[[], bool] | None, deadline: float | None) -> None:
        if self._cancelled(cancel_requested, deadline):
            raise _Cancelled

    @staticmethod
    def _transient(error: Exception) -> bool:
        current: BaseException | None = error
        while current is not None:
            if isinstance(current, (TimeoutError, ConnectionError)):
                return True
            current = current.__cause__
        return False

    def _run_leaf(self, node: LoweredNode, state: dict[str, Any], ledger: AttemptLedger,
                  cancel_requested: Callable[[], bool] | None, deadline: float | None) -> dict[str, Any]:
        transient_retries = invalid_retries = 0
        while True:
            self._check_cancel(cancel_requested, deadline)
            view = _NodeBackend(self.backend, ledger, node.id,
                                lambda: self._check_cancel(cancel_requested, deadline))
            try:
                return Runtime(node.program, view, enforce_release=self.enforce_release).run(state)
            except (BackendError, TimeoutError, ConnectionError) as error:
                invalid = isinstance(error, BackendError) and ledger.status(view.last_receipt) == "response_received"
                transient = self._transient(error)
                if invalid:
                    ledger.mark(view.last_receipt, "invalid_response")
                if invalid and invalid_retries < self.retry_policy.max_invalid_response_retries:
                    invalid_retries += 1
                elif transient and transient_retries < self.retry_policy.max_transient_retries:
                    transient_retries += 1
                else:
                    if isinstance(error, BackendError):
                        raise
                    raise BackendError(f"Transient backend failure ({type(error).__name__}); retry limit reached.") from error
                if self.retry_policy.delay_seconds:
                    time.sleep(self.retry_policy.delay_seconds)

    def _read(self, ref: Reference, root: dict[str, Any],
              values: dict[str, dict[str, dict[str, Any]]]) -> Any:
        if isinstance(ref, RootRef):
            value = root.get(ref.root)
            present = ref.root in root
        else:
            item = values.get(ref.stage, {}).get(ref.decision, {})
            value = item.get(ref.field)
            present = ref.field in item and value is not None
        if not present or value is None:
            if ref.default is not None:
                return copy.deepcopy(ref.default)
            raise _MissingReference
        return copy.deepcopy(value)

    def _condition(self, condition: Condition | None, root: dict[str, Any],
                   values: dict[str, dict[str, dict[str, Any]]]) -> bool | None:
        if condition is None:
            return True
        missing = False
        for predicate in condition.all:
            try:
                actual = self._read(predicate.ref, root, values)
            except _MissingReference:
                missing = True
                continue
            expected = predicate.value
            accepted = {
                "eq": lambda: actual == expected,
                "in": lambda: actual in expected,
                "lt": lambda: actual < expected,
                "lte": lambda: actual <= expected,
                "gt": lambda: actual > expected,
                "gte": lambda: actual >= expected,
            }[predicate.op]()
            if not accepted:
                return False
        return None if missing else True

    @staticmethod
    def _scope_of(stage_id: str) -> str:
        return stage_id.rpartition("/")[0]

    def _local_stages(self, prefix: str) -> dict[str, LoweredNode | LoweredExport]:
        parent = prefix.removesuffix("/")
        return {stage.id: stage for stage in [*self.artifact.nodes, *self.artifact.exports]
                if self._scope_of(stage.id) == parent}

    def _local_order(self, stages: dict[str, LoweredNode | LoweredExport]) -> list[str]:
        names = set(stages)
        dependencies: dict[str, set[str]] = {}
        for name, stage in stages.items():
            refs = list(stage.inputs.values())
            if stage.when:
                refs.extend(item.ref for item in stage.when.all)
            dependencies[name] = {ref.stage for ref in refs if not isinstance(ref, RootRef) and ref.stage in names}
            dependencies[name].update(item for item in stage.after if item in names)
        order: list[str] = []
        pending = set(names)
        while pending:
            ready = sorted(name for name in pending if dependencies[name] <= set(order))
            if not ready:
                raise DataError("Frozen graph has a local dependency cycle.")
            order.append(ready[0])
            pending.remove(ready[0])
        return order

    def _mark_descendants(self, export_id: str, stages: dict[str, dict[str, Any]],
                          status: str, reason: str) -> None:
        prefix = export_id + "/"
        for descendant in [*self.artifact.nodes, *self.artifact.exports]:
            if descendant.id.startswith(prefix):
                stages.setdefault(descendant.id, {"status": status, "reason": reason})

    def _select_final(self, final: dict[str, Any], values: dict[str, dict[str, dict[str, Any]]],
                      stages: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], bool]:
        selected: dict[str, dict[str, Any]] = {}
        review = False
        for name, mapping in final.items():
            available = [candidate for candidate in mapping.candidates
                         if stages.get(candidate.stage, {}).get("status") == "completed"
                         and candidate.decision in values.get(candidate.stage, {})]
            if len(available) > 1:
                raise _StageFailure(DataError(f"Ambiguous final output: {name}"))
            if not available:
                review = True
                continue
            candidate: Candidate = available[0]
            answer = dict(values[candidate.stage][candidate.decision])
            if candidate.distribution_scope == "branch_conditional":
                answer["value"] = candidate.label_map[answer["value"]]
            answer["distribution_scope"] = candidate.distribution_scope
            answer["origin"] = {"stage": candidate.stage, "decision": candidate.decision}
            answer["review_required"] = bool(answer["review_required"] or
                                             stages[candidate.stage].get("review_required", False))
            review = review or answer["review_required"]
            selected[name] = answer
        return selected, review

    def _run_scope(self, prefix: str, final: dict[str, Any], root: dict[str, Any],
                   values: dict[str, dict[str, dict[str, Any]]], stages: dict[str, dict[str, Any]],
                   executed: list[str], path: list[str], cancel_requested: Callable[[], bool] | None,
                   deadline: float | None, ledger: AttemptLedger,
                   ) -> tuple[dict[str, dict[str, Any]], bool]:
        local = self._local_stages(prefix)
        for name in self._local_order(local):
            stage = local[name]
            if self._cancelled(cancel_requested, deadline):
                stages[name] = {"status": "cancelled", "reason": "caller_cancelled"}
                raise _Cancelled
            condition = self._condition(stage.when, root, values)
            if condition is False:
                stages[name] = {"status": "skipped", "reason": "condition_false"}
                if isinstance(stage, LoweredExport):
                    self._mark_descendants(name, stages, "skipped", "ancestor_skipped")
                continue
            if condition is None:
                stages[name] = {"status": "review_blocked", "reason": "condition_input_unavailable"}
                if isinstance(stage, LoweredExport):
                    self._mark_descendants(name, stages, "review_blocked", "ancestor_review_blocked")
                continue
            ports = stage.program.state if isinstance(stage, LoweredNode) else stage.input_contracts
            refs = list(stage.inputs.values())
            condition_refs = [item.ref for item in stage.when.all] if stage.when else []
            lineage_deps = {ref.stage for ref in [*refs, *condition_refs]
                            if not isinstance(ref, RootRef)} | set(stage.after)
            required_deps = {ref.stage for ref in condition_refs if not isinstance(ref, RootRef)} | set(stage.after)
            required_deps.update(ref.stage for port, ref in stage.inputs.items()
                                 if not isinstance(ref, RootRef) and ports[port].required)
            if any(stages.get(dep, {}).get("status") != "completed" or (
                stages[dep].get("review_required") and self._stage(dep).on_review == "defer"
            ) for dep in required_deps):
                stages[name] = {"status": "review_blocked", "reason": "predecessor_unavailable"}
                if isinstance(stage, LoweredExport):
                    self._mark_descendants(name, stages, "review_blocked", "ancestor_review_blocked")
                continue
            mapped: dict[str, Any] = {}
            blocked = False
            for port, ref in stage.inputs.items():
                try:
                    if not isinstance(ref, RootRef) and (
                        stages.get(ref.stage, {}).get("status") != "completed" or (
                            stages[ref.stage].get("review_required") and self._stage(ref.stage).on_review == "defer"
                        )
                    ):
                        raise _MissingReference
                    mapped[port] = self._read(ref, root, values)
                except _MissingReference:
                    if ports[port].required:
                        blocked = True
                    elif ref.default is not None:
                        mapped[port] = ref.default
            if blocked:
                stages[name] = {"status": "review_blocked", "reason": "required_input_unavailable"}
                if isinstance(stage, LoweredExport):
                    self._mark_descendants(name, stages, "review_blocked", "ancestor_review_blocked")
                continue
            try:
                projected = project_state(ports, mapped)
                stages[name] = {"status": "running"}
                if isinstance(stage, LoweredNode):
                    response = self._run_leaf(stage, projected, ledger, cancel_requested, deadline)
                    values[name] = response["decisions"]
                    executed.append(name)
                    review = any(item["review_required"] for item in response["decisions"].values())
                else:
                    outputs, review = self._run_scope(name + "/", stage.outputs, root, values, stages,
                                                      executed, path, cancel_requested, deadline, ledger)
                    if set(outputs) != set(stage.output_contracts):
                        stages[name] = {"status": "review_blocked", "reason": "subgraph_incomplete"}
                        continue
                    values[name] = outputs
                review = review or any(stages.get(dep, {}).get("review_required", False)
                                       for dep in lineage_deps)
                stages[name] = {"status": "completed", "review_required": review,
                                "kind": "leaf" if isinstance(stage, LoweredNode) else "subgraph"}
                path.append(name)
            except (BackendError, BudgetExceeded, ConfigurationError, DataError) as exc:
                stages[name] = {"status": "failed", "reason": type(exc).__name__}
                raise _StageFailure(exc) from exc
            except _StageFailure:
                stages[name] = {"status": "failed", "reason": "descendant_failed"}
                raise
            except _Cancelled:
                stages[name] = {"status": "cancelled", "reason": "caller_cancelled"}
                raise
        return self._select_final(final, values, stages)

    def _stage(self, name: str) -> LoweredNode | LoweredExport:
        for stage in [*self.artifact.nodes, *self.artifact.exports]:
            if stage.id == name:
                return stage
        raise DataError(f"Unknown frozen stage: {name}")
