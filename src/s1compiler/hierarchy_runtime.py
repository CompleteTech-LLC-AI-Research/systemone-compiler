"""Deterministic execution of validated, frozen hierarchy artifacts."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

from .backends import ManagedBackend
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


class HierarchyRuntime:
    """Run one self-contained graph through the caller's existing managed backend."""

    def __init__(self, artifact: HierarchyArtifact, backend: ManagedBackend, *,
                 enforce_release: bool = False):
        # A snapshot prevents caller mutations after validation from changing a run.
        self.artifact = HierarchyArtifact.model_validate(artifact.model_dump(mode="json"))
        validate_hierarchy_artifact(self.artifact)
        self.backend = backend
        self.enforce_release = enforce_release
        if enforce_release and not backend.synthetic and self.artifact.provenance.status != "measured":
            raise ConfigurationError("Hierarchy composition is not measured; use experimental mode.")
        if any(node.program.model != self.artifact.source.model for node in self.artifact.nodes):
            raise ConfigurationError("Hierarchy leaf model differs from the pinned root model.")

    @classmethod
    def load(cls, path: str | Path, backend: ManagedBackend, **kwargs) -> HierarchyRuntime:
        return cls(HierarchyArtifact.load(path), backend, **kwargs)

    def run(self, state: dict[str, Any], *, cancel_requested: Callable[[], bool] | None = None) -> dict[str, Any]:
        root_state = copy.deepcopy(project_state(self.artifact.source.state, state))
        values: dict[str, dict[str, dict[str, Any]]] = {}
        stages: dict[str, dict[str, Any]] = {}
        executed: list[str] = []
        path: list[str] = []
        before_calls = self.backend.budget.used
        before_hits = self.backend.cache_hits
        failure: Exception | None = None
        status = "review_required"
        decisions: dict[str, dict[str, Any]] = {}
        try:
            decisions, review = self._run_scope("", self.artifact.final, root_state,
                                                values, stages, executed, path, cancel_requested)
            status = "review_required" if review or set(decisions) != set(self.artifact.source.decisions) else "completed"
        except _Cancelled:
            status = "cancelled"
        except _StageFailure as exc:
            status, failure = "failed", exc.error
        for stage in [*self.artifact.nodes, *self.artifact.exports]:
            stages.setdefault(stage.id, {"status": "cancelled" if status == "cancelled" else "pending",
                                         "reason": "cancelled" if status == "cancelled" else "not_reached"})
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
                           "cache_hits": self.backend.cache_hits - before_hits},
        }
        if failure is not None:
            result["error"] = {"type": type(failure).__name__, "message": str(failure)}
        return result

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
                   ) -> tuple[dict[str, dict[str, Any]], bool]:
        local = self._local_stages(prefix)
        for name in self._local_order(local):
            stage = local[name]
            if cancel_requested and cancel_requested():
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
                    response = Runtime(stage.program, self.backend,
                                       enforce_release=self.enforce_release).run(projected)
                    values[name] = response["decisions"]
                    executed.append(name)
                    review = any(item["review_required"] for item in response["decisions"].values())
                else:
                    outputs, review = self._run_scope(name + "/", stage.outputs, root, values, stages,
                                                      executed, path, cancel_requested)
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
