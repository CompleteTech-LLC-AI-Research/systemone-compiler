"""Authored hierarchy compilation with explicit selection, calibration and test gates."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any

from .architect import template_program
from .backends import ManagedBackend
from .compiler import versions
from .data import Example, dataset_hash
from .errors import ConfigurationError, DataError
from .hierarchy import (FinalReviewGate, HierarchyArtifact, HierarchyProvenance,
                        HierarchySource, lower_hierarchy)
from .hierarchy_data import HierarchySplitGuard
from .hierarchy_metrics import evaluate_hierarchy, paired_flat_hierarchy
from .hierarchy_validation import validate_hierarchy_artifact, validate_hierarchy_compile_inputs
from .io import fingerprint
from .metrics import evaluate


@dataclass(frozen=True)
class HierarchyCompileOptions:
    """H09 supports authored graphs without teacher proposals or text optimization."""

    architect: str = "template"
    optimizer: str = "none"
    max_calibration_error: float = 0.05
    min_calibration_samples: int = 10

    def __post_init__(self):
        if self.architect != "template" or self.optimizer != "none":
            raise ConfigurationError("Authored hierarchy compile currently supports template/none only.")
        if (type(self.max_calibration_error) not in (int, float) or
                not math.isfinite(self.max_calibration_error) or
                not 0 <= self.max_calibration_error <= 1 or
                type(self.min_calibration_samples) is not int or self.min_calibration_samples < 1):
            raise ConfigurationError("Invalid hierarchy calibration options.")


@dataclass
class HierarchyCompileSession:
    """One candidate and one held-out phase; do not reuse after testing."""

    source: HierarchySource
    splits: dict[str, tuple[Example, ...]]
    candidate: HierarchyArtifact
    guard: HierarchySplitGuard
    flat_baseline: Any
    owner_start_calls: int
    owner_request_limit: int
    train_report: dict[str, Any]
    validation_report: dict[str, Any]
    baseline_validation: dict[str, Any]
    validation_pair: dict[str, Any]
    observed_nodes: set[str]
    phase: str = "selected"
    calibration_report: dict[str, Any] | None = None
    calibration_fit: dict[str, Any] | None = None
    gates: dict[str, dict[str, FinalReviewGate]] = field(default_factory=dict)
    frozen: HierarchyArtifact | None = None
    frozen_guard: HierarchySplitGuard | None = None
    frozen_calibration_report: dict[str, Any] | None = None


def _paired(artifact: HierarchyArtifact, rows: list[Example], graph: list[dict[str, Any]],
            flat: list[dict[str, Any]]) -> dict[str, Any]:
    records = [{"id": row.id, "group": row.group, "gold": copy.deepcopy(row.expected),
                "result": result} for row, result in zip(rows, flat)]
    return paired_flat_hierarchy(artifact, rows, [graph], [records])


def _correct(declaration, value: Any, gold: Any) -> bool:
    if declaration.type == "score":
        return abs(value - gold) <= declaration.score_tolerance
    return value == gold


def _fit_final_gates(artifact: HierarchyArtifact, rows: list[Example], results: list[dict[str, Any]],
                     options: HierarchyCompileOptions) -> tuple[dict[str, dict[str, FinalReviewGate]], dict[str, Any]]:
    """Fit only post-route gates; no intermediate label is inferred from final gold."""
    gates: dict[str, dict[str, FinalReviewGate]] = {}
    report: dict[str, Any] = {}
    for name, mapping in artifact.final.items():
        declaration = artifact.source.decisions[name]
        gates[name], report[name] = {}, {}
        for stage in sorted({candidate.stage for candidate in mapping.candidates}):
            observed = []
            for row, result in zip(rows, results):
                if result["status"] != "completed":
                    continue
                prediction = result["decisions"][name]
                if prediction["origin"]["stage"] != stage:
                    continue
                score = prediction.get("gate_score")
                if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise DataError("Calibration requires a finite final gate score.")
                observed.append((score, _correct(declaration, prediction["value"], row.expected[name])))
            candidates = []
            for threshold in sorted({0.0, 1.0, *(score for score, _ in observed)}):
                accepted = [correct for score, correct in observed if score >= threshold]
                errors = sum(not correct for correct in accepted)
                if (len(accepted) >= options.min_calibration_samples and
                        errors / len(accepted) <= options.max_calibration_error):
                    candidates.append((len(accepted), -errors, threshold, errors))
            if candidates:
                count, _, threshold, errors = max(candidates)
                gate = FinalReviewGate(min_gate=threshold)
                report[name][stage] = {"status": "fitted", "observed_n": len(observed),
                                       "accepted_n": count, "empirical_error": errors / count,
                                       "policy": gate.model_dump(mode="json")}
            else:
                gate = FinalReviewGate(force_review=True)
                report[name][stage] = {"status": "review_only", "observed_n": len(observed),
                                       "reason": "unreachable_or_insufficient_accepted_calibration_rows",
                                       "policy": gate.model_dump(mode="json")}
            gates[name][stage] = gate
    return gates, report


class HierarchyCompiler:
    """Compile an authored graph; H10/H11 can reuse phase boundaries for search."""

    def __init__(self, backend: ManagedBackend, *, options: HierarchyCompileOptions | None = None):
        self.backend = backend
        self.options = options or HierarchyCompileOptions()

    def select(self, source: HierarchySource, *, train: list[Example], validation: list[Example],
               calibration: list[Example], test: list[Example]) -> HierarchyCompileSession:
        splits = {"train": train, "validation": validation, "calibration": calibration, "test": test}
        preflight = validate_hierarchy_compile_inputs(source, splits)
        candidate = lower_hierarchy(source)
        guard = preflight.split_guard
        if guard.graph_sha256 != candidate.content_hash:
            raise DataError("Lowered graph changed after split preflight.")
        stable = {name: tuple(copy.deepcopy(guard.splits[name])) for name in splits}
        flat = template_program(source.source)
        start = self.backend.budget.used
        train_report, train_results = evaluate_hierarchy(candidate, list(stable["train"]), self.backend,
                                                         guard=guard, split="train")
        validation_report, graph_results = evaluate_hierarchy(
            candidate, list(stable["validation"]), self.backend, guard=guard, split="validation")
        baseline_report, flat_results = evaluate(flat, list(stable["validation"]), self.backend)
        pair = _paired(candidate, list(stable["validation"]), graph_results, flat_results)
        observed = {stage for result in [*train_results, *graph_results] for stage in result["executed"]}
        return HierarchyCompileSession(
            source=source.model_copy(deep=True), splits=stable, candidate=candidate, guard=guard,
            flat_baseline=flat, owner_start_calls=start, owner_request_limit=self.backend.budget.maximum,
            train_report=train_report, validation_report=validation_report,
            baseline_validation=baseline_report, validation_pair=pair, observed_nodes=observed)

    def calibrate(self, session: HierarchyCompileSession) -> HierarchyCompileSession:
        if session.phase != "selected":
            raise ConfigurationError("Calibration follows selection exactly once.")
        rows = list(session.splits["calibration"])
        report, results = evaluate_hierarchy(session.candidate, rows, self.backend,
                                             guard=session.guard, split="calibration")
        session.gates, session.calibration_fit = _fit_final_gates(
            session.candidate, rows, results, self.options)
        session.observed_nodes.update(stage for result in results for stage in result["executed"])
        session.calibration_report = report
        session.phase = "calibrated"
        return session

    def freeze(self, session: HierarchyCompileSession) -> HierarchyArtifact:
        if session.phase != "calibrated":
            raise ConfigurationError("Freeze follows calibration exactly once.")
        if (session.candidate.content_hash != session.guard.graph_sha256 or
                lower_hierarchy(session.source).content_hash != session.candidate.content_hash or any(
            dataset_hash(list(session.splits[name])) != dataset_hash(list(session.guard.splits[name]))
            for name in session.splits
        )):
            raise DataError("Graph or datasets changed after candidate selection.")
        frozen = session.candidate.model_copy(deep=True)
        frozen.final_review_gates = copy.deepcopy(session.gates)
        unvisited = sorted({node.id for node in frozen.nodes} - session.observed_nodes)
        fully_measured = not self.backend.synthetic and not unvisited
        if fully_measured:
            for node in frozen.nodes:
                node.program.provenance = {**node.program.provenance, "status": "measured",
                                           "deployment_approved": False}
        frozen.provenance = HierarchyProvenance(
            status="synthetic" if self.backend.synthetic else "measured" if fully_measured else "draft",
            composition_measured=fully_measured, deployment_approved=False,
            evidence={"composition_report_sha256": fingerprint(session.calibration_report),
                      "source_sha256": fingerprint(session.source.model_dump(mode="json")),
                      "splits": {name: {"sha256": dataset_hash(list(rows)), "n": len(rows)}
                                 for name, rows in session.splits.items()},
                      "child_hashes": {node.id: node.program.content_hash for node in frozen.nodes},
                      "backend": self.backend.identity, "versions": versions(),
                      "execution_kind": "synthetic" if self.backend.synthetic else "real_provider",
                      "unvisited_nodes_before_test": unvisited,
                      "compiled_at": datetime.now(timezone.utc).isoformat(),
                      "owner_request_limit": session.owner_request_limit,
                      "route_policy": "frozen_before_calibration_final_gates",
                      "review_policy": "post_route_only"})
        frozen = HierarchyArtifact.model_validate(frozen.model_dump(mode="json"))
        validate_hierarchy_artifact(frozen)
        frozen_guard = HierarchySplitGuard(frozen, {name: list(rows) for name, rows in session.splits.items()})
        # Recompute the whole calibration path with the final gates. This is bounded
        # and never treated as another validation or candidate-selection pass.
        final_calibration, _ = evaluate_hierarchy(frozen, list(session.splits["calibration"]),
                                                  self.backend, guard=frozen_guard, split="calibration")
        frozen.provenance.evidence["calibration_after_gates_sha256"] = fingerprint(final_calibration)
        frozen.provenance.evidence["owner_calls_before_test"] = (
            self.backend.budget.used - session.owner_start_calls)
        frozen = HierarchyArtifact.model_validate(frozen.model_dump(mode="json"))
        session.frozen = frozen
        session.frozen_guard = frozen_guard
        session.frozen_calibration_report = final_calibration
        session.phase = "frozen"
        return frozen

    def test(self, session: HierarchyCompileSession) -> dict[str, Any]:
        if session.phase != "frozen" or session.frozen is None or session.frozen_guard is None:
            raise ConfigurationError("Held-out test runs once after complete freeze.")
        if session.frozen.content_hash != session.frozen_guard.graph_sha256 or any(
            dataset_hash(list(session.splits[name])) != dataset_hash(list(session.frozen_guard.splits[name]))
            for name in session.splits
        ):
            raise DataError("Frozen graph or datasets changed before held-out test.")
        session.phase = "testing"  # A failed call does not authorize a second held-out phase.
        rows = list(session.splits["test"])
        baseline, flat_results = evaluate(session.flat_baseline, rows, self.backend)
        compiled, graph_results = evaluate_hierarchy(session.frozen, rows, self.backend,
                                                     guard=session.frozen_guard, split="test")
        pair = _paired(session.frozen, rows, graph_results, flat_results)
        session.phase = "tested"
        return {"format": "systemone-hierarchy-compile-report/v1", "graph_sha256": session.frozen.content_hash,
                "provenance_sha256": session.frozen.provenance_hash,
                "status": session.frozen.provenance.status, "deployment_approved": False,
                "selection": {"candidate": "authored", "train": session.train_report,
                              "validation": session.validation_report,
                              "baseline_validation": session.baseline_validation,
                              "paired_validation": session.validation_pair},
                "calibration": {"before_gates": session.calibration_report,
                                "fit": session.calibration_fit,
                                "after_gates": session.frozen_calibration_report},
                "test": {"hierarchy": compiled, "flat": baseline, "paired": pair},
                "accounting": {"owner_request_limit": session.owner_request_limit,
                               "native_calls_this_compile": self.backend.budget.used - session.owner_start_calls,
                               "owner": self.backend.accounting()},
                "limitations": ["Synthetic execution is not Jev quality evidence.",
                                "Post-route gates do not tune routing; low-sample branches review.",
                                "Held-out scores are descriptive, not deployment approval."]}

    def compile(self, source: HierarchySource, *, train: list[Example], validation: list[Example],
                calibration: list[Example], test: list[Example]
                ) -> tuple[HierarchyArtifact, dict[str, Any]]:
        session = self.select(source, train=train, validation=validation,
                              calibration=calibration, test=test)
        self.calibrate(session)
        artifact = self.freeze(session)
        return artifact, self.test(session)
