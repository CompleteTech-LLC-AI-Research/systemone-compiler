from __future__ import annotations
from typing import Any

from .data import Example
from .metrics import is_correct, wilson_upper
from .models import Policy, Program
from .runtime import apply_policy


def fit_policies(program: Program, rows: list[Example], results: list[dict[str, Any]], *,
                 max_error: float = 0.05, min_samples: int = 10) -> tuple[Program, dict[str, Any]]:
    """Fit thresholds on the calibration split only. No new model calls.

    Select the largest observed accepted set within an empirical error target.
    This is an operational heuristic; it is NOT a distribution-free guarantee.
    """
    if not 0 <= max_error <= 1 or min_samples < 1 or not rows or len(rows) != len(results):
        raise ValueError("Invalid calibration data or policy configuration.")
    fitted = program.model_copy(deep=True)
    report = {}
    for name, decision in fitted.decisions.items():
        initial_policy = fitted.policies[name]
        threshold = initial_policy.noul_threshold
        if decision.type == "noul":
            p_values = [r["decisions"][name]["p_true"] for r in results]
            # Includes just-above-observation thresholds to represent all partitions.
            grid = {0.5, 0.001, 0.999}
            grid.update(p for p in p_values if 0 < p < 1)
            grid.update(min(.999999, p + 1e-9) for p in p_values if p < 1)
            threshold = min(grid, key=lambda t: (
                sum((p >= t) != row.expected[name] for p, row in zip(p_values, rows)), abs(t-.5)))
        fitted.policies[name] = Policy(noul_threshold=threshold, min_gate=0, force_review=False)
        predictions = [apply_policy(fitted, r["answers"])[name] for r in results]
        correct = [is_correct(fitted, name, pred["value"], row.expected[name])
                   for pred, row in zip(predictions, rows)]
        gates = sorted({0.0, 1.0, *[pred["gate_score"] for pred in predictions]})
        options = []
        for gate in gates:
            accepted = [i for i, pred in enumerate(predictions) if pred["gate_score"] >= gate]
            errors = sum(not correct[i] for i in accepted)
            if len(accepted) >= min_samples and errors/len(accepted) <= max_error:
                options.append((len(accepted), -errors, gate, errors))
        if options:
            count, _, gate, errors = max(options)
            fitted.policies[name] = Policy(noul_threshold=threshold, min_gate=gate)
            report[name] = {"status": "fitted", "accepted_n": count, "empirical_error": errors/count,
                            "wilson_upper_95_descriptive": wilson_upper(errors, count)}
        else:
            fitted.policies[name] = Policy(noul_threshold=threshold, min_gate=1.0, force_review=True)
            report[name] = {"status": "review_only", "reason": "No threshold met sample/error requirements."}
        report[name].update(policy=fitted.policies[name].model_dump(), min_samples=min_samples,
                            empirical_error_target=max_error)
    return fitted, report
