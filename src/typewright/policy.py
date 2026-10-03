from __future__ import annotations
from typing import Any

from .data import Example
from .metrics import is_correct, wilson_upper
from .io import fingerprint
from .models import Policy, Program
from .runtime import apply_policy


def _loss(program, name, rows, results):
    declaration = program.decisions[name]
    values = [apply_policy(program, r["answers"])[name]["value"] for r in results]
    if declaration.type == "choice":
        return sum(value != row.expected[name] for value, row in zip(values, rows)) / len(rows)
    return sum(abs(value - row.expected[name]) for value, row in zip(values, rows)) / (
        len(rows) * (len(declaration.criteria) - 1))


def _search(program, name, rows, results):
    """Bounded coordinate search; only strictly better candidates replace baseline."""
    declaration = program.decisions[name]
    current = program.model_copy(deep=True)
    best_loss = _loss(current, name, rows, results)
    base = current.policies[name].model_dump()
    if declaration.type == "choice":
        knobs = dict(base.get("choice_weights", dict.fromkeys(declaration.criteria, 1.0)))
        coordinates = list(declaration.criteria)
        grid = [.25, .5, 1.0, 2.0, 4.0]
        field = "choice_weights"
    else:
        knobs = list(base.get("score_cuts", [i + .5 for i in range(len(declaration.criteria) - 1)]))
        coordinates = list(range(len(knobs)))
        raw_program = program.model_copy(deep=True)
        raw_policy = base.copy()
        raw_policy.pop("score_cuts", None)
        raw_policy.pop("fitting_version", None)
        raw_program.policies[name] = Policy(**raw_policy)
        values = sorted({apply_policy(raw_program, r["answers"])[name]["value"] for r in results})
        midpoints = [(a + b) / 2 for a, b in zip(values, values[1:])]
        # At most 33 observed boundaries plus endpoints and conventional cuts.
        grid = sorted({0.0, float(len(declaration.criteria) - 1), *knobs,
                       *[midpoints[i] for i in range(0, len(midpoints), max(1, len(midpoints)//32 + 1))]})
        field = "score_cuts"
    for _ in range(2):
        for coordinate in coordinates:
            for value in grid:
                proposed = knobs.copy()
                proposed[coordinate] = value
                if field == "score_cuts" and any(a >= b for a, b in zip(proposed, proposed[1:])):
                    continue
                candidate = current.model_copy(deep=True)
                candidate.policies[name] = Policy(**{**base, "fitting_version": "native-policy/v1", field: proposed})
                loss = _loss(candidate, name, rows, results)
                if loss < best_loss - 1e-12:
                    current, knobs, best_loss = candidate, proposed, loss
    return current.policies[name]


def _fit_knobs(program, name, rows, results, min_samples):
    detail = {"status": "unchanged", "calibration_n": len(rows), "folds": [],
              "synthetic": any(r.get("synthetic", False) for r in results),
              "objective": "misclassification_rate" if program.decisions[name].type == "choice"
                           else "normalized_mae",
              "method": "native-policy/v1; bounded coordinate search; five grouped held-out folds",
              "min_fold_samples": max(2, min_samples)}
    # Group boundaries are retained inside calibration, too. IDs determine stable
    # order without using outcomes or input text to construct the folds.
    units = {}
    for i, row in enumerate(rows):
        key = ("group", row.group) if row.group else ("id", row.id)
        units.setdefault(key, []).append(i)
    folds = [[] for _ in range(5)]
    for j, key in enumerate(sorted(units, key=fingerprint)):
        folds[j % 5].extend(units[key])
    if any(len(fold) < max(2, min_samples) for fold in folds):
        detail["reason"] = "Insufficient independent held-out calibration samples."
        return program.policies[name], detail
    for fold in folds:
        held_out = set(fold)
        train = [i for i in range(len(rows)) if i not in held_out]
        candidate = program.model_copy(deep=True)
        candidate.policies[name] = _search(program, name, [rows[i] for i in train], [results[i] for i in train])
        gold = [rows[i] for i in fold]
        cached = [results[i] for i in fold]
        detail["folds"].append({"train_n": len(train), "held_out_n": len(fold),
            "default_loss": _loss(program, name, gold, cached),
            "candidate_loss": _loss(candidate, name, gold, cached),
            "policy": candidate.policies[name].model_dump()})
    checks = detail["folds"]
    if (any(f["candidate_loss"] > f["default_loss"] + 1e-12 for f in checks)
            or sum(f["held_out_n"] * (f["default_loss"] - f["candidate_loss"]) for f in checks) <= 1e-12):
        detail["reason"] = "No conservative held-out calibration improvement; retain default."
        return program.policies[name], detail
    fitted = _search(program, name, rows, results)
    if fitted == program.policies[name]:
        detail["reason"] = "Full calibration refit did not beat default."
        return fitted, detail
    detail["status"] = "fitted"
    detail["policy"] = fitted.model_dump()
    return fitted, detail


def fit_policies(program: Program, rows: list[Example], results: list[dict[str, Any]], *,
                 max_error: float = 0.05, min_samples: int = 10,
                 fit_decision_knobs: bool = False) -> tuple[Program, dict[str, Any]]:
    """Fit decision knobs and gates on cached calibration predictions only.

    Select the largest observed accepted set within an empirical error target.
    This is an operational heuristic; it is NOT a distribution-free guarantee.
    """
    if not 0 <= max_error <= 1 or min_samples < 1 or not rows or len(rows) != len(results):
        raise ValueError("Invalid calibration data or policy configuration.")
    if type(fit_decision_knobs) is not bool:
        raise ValueError("fit_decision_knobs must be a boolean.")
    fitted = program.model_copy(deep=True)
    report = {}
    for name, decision in fitted.decisions.items():
        initial_policy = fitted.policies[name]
        knob_report = None
        if fit_decision_knobs and decision.type in {"choice", "score"}:
            initial_policy, knob_report = _fit_knobs(program, name, rows, results, min_samples)
        elif decision.type in {"choice", "score"}:
            knob_report = {"status": "disabled", "reason": "Decision fitting requires explicit opt-in.",
                           "calibration_n": len(rows), "folds": []}
        threshold = initial_policy.noul_threshold
        if decision.type == "noul":
            p_values = [r["decisions"][name]["p_true"] for r in results]
            # Includes just-above-observation thresholds to represent all partitions.
            grid = {0.5, 0.001, 0.999}
            grid.update(p for p in p_values if 0 < p < 1)
            grid.update(min(.999999, p + 1e-9) for p in p_values if p < 1)
            threshold = min(grid, key=lambda t: (
                sum((p >= t) != row.expected[name] for p, row in zip(p_values, rows)), abs(t-.5)))
        fitted.policies[name] = Policy(**{**initial_policy.model_dump(), "noul_threshold": threshold,
                                         "min_gate": 0, "force_review": False})
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
            fitted.policies[name] = Policy(**{**initial_policy.model_dump(), "noul_threshold": threshold,
                                             "min_gate": gate, "force_review": False})
            report[name] = {"status": "fitted", "accepted_n": count, "empirical_error": errors/count,
                            "wilson_upper_95_descriptive": wilson_upper(errors, count)}
        else:
            fitted.policies[name] = Policy(**{**initial_policy.model_dump(), "noul_threshold": threshold,
                                             "min_gate": 1.0, "force_review": True})
            report[name] = {"status": "review_only", "reason": "No threshold met sample/error requirements."}
        if knob_report is not None:
            report[name]["decision_fit"] = knob_report
        report[name].update(calibration_n=len(rows), synthetic=any(r.get("synthetic", False) for r in results),
                            policy=fitted.policies[name].model_dump(), min_samples=min_samples,
                            empirical_error_target=max_error)
    return fitted, report
