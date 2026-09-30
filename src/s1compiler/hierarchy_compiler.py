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
from .errors import CandidateError, ConfigurationError, DataError
from .hierarchy import (FinalReviewGate, HierarchyArtifact, HierarchyProvenance,
                        HierarchySource, lower_hierarchy)
from .hierarchy_architect import semantic_review_manifest
from .hierarchy_data import HierarchySplitGuard, HierarchyTeacherInputs
from .hierarchy_gepa import (components_from_hierarchy, hierarchy_text_structure_hash,
                             optimize_hierarchy_gepa)
from .hierarchy_metrics import evaluate_hierarchy, paired_flat_hierarchy
from .hierarchy_validation import validate_hierarchy_artifact, validate_hierarchy_compile_inputs
from .io import fingerprint
from .metrics import evaluate


@dataclass(frozen=True)
class HierarchyCompileOptions:
    """Bounded structure search precedes calibration and the one-shot test."""

    architect: str = "template"
    optimizer: str = "none"
    structural_rounds: int = 0
    max_metric_calls: int = 128
    seed: int = 7
    max_calibration_error: float = 0.05
    min_calibration_samples: int = 10

    def __post_init__(self):
        if self.architect not in {"template", "dspy"} or self.optimizer not in {"none", "gepa"}:
            raise ConfigurationError("Unknown hierarchy architect or optimizer.")
        if (type(self.structural_rounds) is not int or not 0 <= self.structural_rounds <= 10 or
                (self.architect == "template" and self.structural_rounds != 0) or
                (self.architect == "dspy" and self.structural_rounds == 0)):
            raise ConfigurationError("DSPy structure search needs 1..10 rounds; template needs zero.")
        if (type(self.max_metric_calls) is not int or self.max_metric_calls < 1 or
                type(self.seed) is not int):
            raise ConfigurationError("Invalid GEPA metric budget or deterministic seed.")
        if (type(self.max_calibration_error) not in (int, float) or
                not math.isfinite(self.max_calibration_error) or
                not 0 <= self.max_calibration_error <= 1 or
                type(self.min_calibration_samples) is not int or self.min_calibration_samples < 1):
            raise ConfigurationError("Invalid hierarchy calibration options.")


@dataclass
class HierarchyCompileSession:
    """One candidate and one held-out phase; do not reuse after testing."""

    source: HierarchySource
    selected_source: HierarchySource
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
    proposal_history: list[dict[str, Any]]
    semantic_review: dict[str, Any]
    optimization: dict[str, Any]
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
    """Compile an authored graph, optionally selecting bounded DSPy structures."""

    def __init__(self, backend: ManagedBackend, *, teacher=None,
                 options: HierarchyCompileOptions | None = None):
        self.backend, self.teacher = backend, teacher
        self.options = options or HierarchyCompileOptions()

    def select(self, source: HierarchySource, *, train: list[Example], validation: list[Example],
               calibration: list[Example], test: list[Example]) -> HierarchyCompileSession:
        splits = {"train": train, "validation": validation, "calibration": calibration, "test": test}
        preflight = validate_hierarchy_compile_inputs(source, splits)
        fixed_source = source.model_copy(deep=True)
        if (self.options.architect == "dspy" or self.options.optimizer == "gepa") and self.teacher is None:
            raise ConfigurationError("Hierarchy structure or wording search requires an explicit teacher.")
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
        history = [{"phase": "authored", "graph_sha256": candidate.content_hash,
                    "validation_objective": validation_report["objective"]}]
        selected_source = fixed_source.model_copy(deep=True)
        for round_index in range(self.options.structural_rounds):
            teacher_inputs = HierarchyTeacherInputs(guard)
            feedback = {"examples": teacher_inputs.examples(list(stable["train"])),
                        "traces": teacher_inputs.traces(train_results),
                        "root_quality": train_report["quality_by_root_id"]}
            try:
                proposed = self.teacher.propose_hierarchy(fixed_source.model_copy(deep=True),
                                                          selected_source.model_copy(deep=True), feedback)
                if not isinstance(proposed, HierarchySource):
                    raise CandidateError("Teacher did not return a typed hierarchy source.")
                proposed = HierarchySource.model_validate(proposed.model_dump(mode="json"))
                if (proposed.source != fixed_source.source or proposed.limits != fixed_source.limits or
                        proposed.format != fixed_source.format):
                    raise CandidateError("Teacher proposal changed the fixed source contract or limits.")
                proposed_artifact = lower_hierarchy(proposed)
                proposed_guard = HierarchySplitGuard(
                    proposed_artifact, {name: list(rows) for name, rows in stable.items()})
            except (CandidateError, DataError, ValueError) as exc:
                history.append({"phase": "dspy_structure", "round": round_index,
                                "rejected": "invalid_typed_graph", "error_type": type(exc).__name__})
                continue
            measured, proposed_results = evaluate_hierarchy(
                proposed_artifact, list(stable["validation"]), self.backend,
                guard=proposed_guard, split="validation")
            score = measured["objective"]
            history.append({"phase": "dspy_structure", "round": round_index,
                            "graph_sha256": proposed_artifact.content_hash,
                            "validation_objective": score})
            if score > validation_report["objective"]:
                proposed_train, proposed_train_results = evaluate_hierarchy(
                    proposed_artifact, list(stable["train"]), self.backend,
                    guard=proposed_guard, split="train")
                selected_source, candidate, guard = proposed, proposed_artifact, proposed_guard
                train_report, train_results = proposed_train, proposed_train_results
                validation_report, graph_results = measured, proposed_results
                pair = _paired(candidate, list(stable["validation"]), graph_results, flat_results)
                observed = {stage for result in [*train_results, *graph_results]
                            for stage in result["executed"]}
        optimization = {"engine": "none"}
        text_changes = []
        if self.options.optimizer == "gepa":
            optimized, optimization = optimize_hierarchy_gepa(
                candidate, {name: list(rows) for name, rows in stable.items()},
                self.backend, self.teacher, max_metric_calls=self.options.max_metric_calls,
                seed=self.options.seed)
            optimized_guard = HierarchySplitGuard(
                optimized, {name: list(rows) for name, rows in stable.items()})
            measured, optimized_results = evaluate_hierarchy(
                optimized, list(stable["validation"]), self.backend,
                guard=optimized_guard, split="validation")
            history.append({"phase": "gepa_wording", "graph_sha256": optimized.content_hash,
                            "validation_objective": measured["objective"],
                            "native_stage_calls": optimization["native_stage_calls"]})
            optimization["accepted_by_compiler"] = measured["objective"] > validation_report["objective"]
            if optimization["accepted_by_compiler"]:
                optimized_train, optimized_train_results = evaluate_hierarchy(
                    optimized, list(stable["train"]), self.backend,
                    guard=optimized_guard, split="train")
                before_text, after_text = components_from_hierarchy(candidate), components_from_hierarchy(optimized)
                text_changes = [{"path": key, "before_sha256": fingerprint(before_text[key]),
                                 "after_sha256": fingerprint(after_text[key])}
                                for key in before_text if before_text[key] != after_text[key]]
                candidate, guard = optimized, optimized_guard
                train_report, train_results = optimized_train, optimized_train_results
                validation_report, graph_results = measured, optimized_results
                pair = _paired(candidate, list(stable["validation"]), graph_results, flat_results)
                observed = {stage for result in [*train_results, *graph_results]
                            for stage in result["executed"]}
        review = semantic_review_manifest(fixed_source, selected_source)
        review["changed_child_prompts"].extend(text_changes)
        return HierarchyCompileSession(
            source=fixed_source, selected_source=selected_source, splits=stable,
            candidate=candidate, guard=guard,
            flat_baseline=flat, owner_start_calls=start, owner_request_limit=self.backend.budget.maximum,
            train_report=train_report, validation_report=validation_report,
            baseline_validation=baseline_report, validation_pair=pair, observed_nodes=observed,
            proposal_history=history, semantic_review=review, optimization=optimization)

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
        selected_structure = lower_hierarchy(session.selected_source)
        same_structure = (selected_structure.content_hash == session.candidate.content_hash or
                          (session.optimization["engine"] == "gepa" and
                           hierarchy_text_structure_hash(selected_structure) ==
                           hierarchy_text_structure_hash(session.candidate)))
        if (session.candidate.content_hash != session.guard.graph_sha256 or not same_structure or any(
            dataset_hash(list(session.splits[name])) != dataset_hash(list(session.guard.splits[name]))
            for name in session.splits
        )):
            raise DataError("Graph or datasets changed after candidate selection.")
        frozen = session.candidate.model_copy(deep=True)
        frozen.final_review_gates = copy.deepcopy(session.gates)
        unvisited = sorted({node.id for node in frozen.nodes} - session.observed_nodes)
        semantic_review_pending = any(session.semantic_review[key] for key in (
            "changed_routing_goals_or_conditions", "changed_child_prompts", "changed_composition"))
        fully_measured = not self.backend.synthetic and not unvisited and not semantic_review_pending
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
                      "semantic_review_pending": semantic_review_pending,
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
        session.semantic_review["frozen_graph_sha256"] = frozen.content_hash
        frozen.provenance.evidence["semantic_review"] = copy.deepcopy(session.semantic_review)
        frozen.provenance.evidence["proposal_history"] = copy.deepcopy(session.proposal_history)
        frozen.provenance.evidence["teacher"] = (self.teacher.accounting() if self.teacher else None)
        frozen.provenance.evidence["optimization"] = copy.deepcopy(session.optimization)
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
                "selection": {"candidate": ("dspy_structure" if session.selected_source != session.source
                                             else "authored"),
                              "history": session.proposal_history, "train": session.train_report,
                              "validation": session.validation_report,
                              "baseline_validation": session.baseline_validation,
                              "paired_validation": session.validation_pair},
                "semantic_review": session.semantic_review,
                "optimization": session.optimization,
                "calibration": {"before_gates": session.calibration_report,
                                "fit": session.calibration_fit,
                                "after_gates": session.frozen_calibration_report},
                "test": {"hierarchy": compiled, "flat": baseline, "paired": pair},
                "accounting": {"owner_request_limit": session.owner_request_limit,
                               "native_calls_this_compile": self.backend.budget.used - session.owner_start_calls,
                               "owner": self.backend.accounting(),
                               "teacher": self.teacher.accounting() if self.teacher else None},
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
