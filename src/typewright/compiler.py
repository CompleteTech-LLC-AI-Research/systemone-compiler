from __future__ import annotations
import importlib.metadata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from .architect import template_program
from .data import Example, assert_disjoint, dataset_hash
from .errors import CandidateError, ConfigurationError
from .io import fingerprint
from .metrics import evaluate, feedback, report_from_results
from .models import Program, UseCase, assert_contract, project_state
from .policy import fit_policies
from .runtime import apply_policy


@dataclass
class CompileOptions:
    architect: Literal["template", "dspy"] = "template"
    optimizer: Literal["none", "gepa"] = "none"
    structural_rounds: int = 0
    max_metric_calls: int = 128
    seed: int = 7
    max_calibration_error: float = 0.05
    min_calibration_samples: int = 10
    fit_decision_knobs: bool = False

    def __post_init__(self):
        if type(self.fit_decision_knobs) is not bool:
            raise ConfigurationError("fit_decision_knobs must be a boolean.")
        if self.architect not in {"template", "dspy"} or self.optimizer not in {"none", "gepa"}:
            raise ConfigurationError("Unknown architect or optimizer.")
        if not 0 <= self.structural_rounds <= 10:
            raise ConfigurationError("Structural rounds must be in 0..10.")
        if self.min_calibration_samples < 1 or not 0 <= self.max_calibration_error <= 1:
            raise ConfigurationError("Invalid calibration parameters.")


def versions():
    result = {"systemone-compiler": "0.1.0"}
    for name in ("pydantic", "PyYAML", "typesafe-sdk", "dspy", "gepa"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


class Compiler:
    def __init__(self, backend, *, teacher=None, options: CompileOptions | None = None):
        self.backend, self.teacher = backend, teacher
        self.options = options or CompileOptions()

    def compile(self, source: UseCase, *, train: list[Example], validation: list[Example],
                calibration: list[Example], test: list[Example]) -> tuple[Program, dict[str, Any]]:
        options = self.options
        splits = {"train": train, "validation": validation, "calibration": calibration, "test": test}
        assert_disjoint(splits, source)  # Happens before any model call.
        if (options.architect == "dspy" or options.optimizer == "gepa" or options.structural_rounds) and not self.teacher:
            raise ConfigurationError("DSPy architecture / GEPA / structural search require a configured teacher.")
        if options.optimizer == "gepa":
            try:
                import gepa  # noqa: F401
            except ImportError as exc:
                raise ConfigurationError("Install the optimize extra before compiling with GEPA.") from exc
        baseline = template_program(source)
        baseline_validation, _ = evaluate(baseline, validation, self.backend)
        best, best_score = baseline, baseline_validation["objective"]
        candidates = [{"phase": "template", "validation_objective": best_score}]
        if options.architect == "dspy":
            training = [{"input": project_state(source.state, r.state), "expected": r.expected} for r in train[:8]]
            try:
                proposed = self.teacher.propose_plan(source, baseline, training)
                assert_contract(proposed, source)
                measured, _ = evaluate(proposed, validation, self.backend)
                score = measured["objective"]
                candidates.append({"phase": "dspy_architect", "validation_objective": score})
                if score > best_score:
                    best, best_score = proposed, score
            except CandidateError:
                candidates.append({"phase": "dspy_architect", "rejected": "invalid_typed_plan"})
        # Bounded outer structural search. Only training errors enter the teacher.
        for round_index in range(options.structural_rounds):
            _, training_predictions = evaluate(best, train, self.backend)
            traces = [feedback(best, row, pred, include_state=True)
                      for row, pred in zip(train, training_predictions)]
            traces.sort(key=lambda trace: trace["quality"])
            try:
                proposed = self.teacher.propose_plan(source, best, traces[:8])
                assert_contract(proposed, source)
                measured, _ = evaluate(proposed, validation, self.backend)
                score = measured["objective"]
                candidates.append({"phase": "structure", "round": round_index, "validation_objective": score})
                if score > best_score:
                    best, best_score = proposed, score
            except CandidateError:
                candidates.append({"phase": "structure", "round": round_index, "rejected": "invalid_typed_plan"})
        optimization = {"engine": "none"}
        if options.optimizer == "gepa":
            from .gepa_adapter import optimize_gepa
            proposed, optimization = optimize_gepa(
                best, train, validation, self.backend, self.teacher,
                max_metric_calls=options.max_metric_calls, seed=options.seed)
            assert_contract(proposed, source)
            measured, _ = evaluate(proposed, validation, self.backend)
            candidates.append({"phase": "gepa", "validation_objective": measured["objective"]})
            if measured["objective"] > best_score:
                best, best_score = proposed, measured["objective"]
        _, calibration_predictions = evaluate(best, calibration, self.backend)
        best, calibration_fit = fit_policies(
            best, calibration, calibration_predictions,
            max_error=options.max_calibration_error, min_samples=options.min_calibration_samples,
            fit_decision_knobs=options.fit_decision_knobs)
        fitted_predictions = [dict(r, decisions=apply_policy(best, r["answers"])) for r in calibration_predictions]
        calibration_report = report_from_results(best, calibration, fitted_predictions)
        best.provenance = {
            "status": "synthetic" if self.backend.synthetic else "measured",
            "deployment_approved": False,
            "compiled_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": fingerprint(source.model_dump(mode="json")),
            "splits": {name: {"sha256": dataset_hash(rows), "n": len(rows)} for name, rows in splits.items()},
            "architect": options.architect, "optimizer": options.optimizer,
            "seed": options.seed, "versions": versions(),
            "calibration": "empirical threshold fitting, not posterior calibration",
            "teacher_model": getattr(self.teacher, "model", None),
        }
        # The test split is first executed here, AFTER every prompt and policy is frozen.
        # Neither test predictions nor labels are provided to the teacher or optimizer.
        baseline_test, _ = evaluate(baseline, test, self.backend)
        compiled_test, _ = evaluate(best, test, self.backend)
        report = {
            "format": "systemone-report/v1", "program_sha256": best.content_hash,
            "status": best.provenance["status"], "deployment_approved": False,
            "selection_history": candidates, "optimization": optimization,
            "baseline_validation": baseline_validation, "selected_validation_objective": best_score,
            "calibration_fit": calibration_fit, "calibration_evaluation": calibration_report,
            "baseline_test": baseline_test, "compiled_test": compiled_test,
            "test_objective_delta": compiled_test["objective"] - baseline_test["objective"],
            "accounting": self.backend.accounting(),
            "teacher": self.teacher.accounting() if self.teacher else None,
            "provenance": best.provenance,
            "limitations": [
                "Synthetic fixture metrics never establish Jev quality or optimization gains.",
                "A held-out test score is descriptive; choose deployment acceptance criteria independently.",
                "Changing labels, Score meaning, or target model requires a new dataset version and evaluation.",
                "Selection is sequential structure-then-wording; not a global joint-search optimizer.",
                "No autonomous actions, arbitrary code execution, or trained model weights are included.",
            ],
        }
        return best, report
