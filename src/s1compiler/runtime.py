from __future__ import annotations
import math
from pathlib import Path
from typing import Any

from .backends import ManagedBackend, Response
from .errors import BackendError, ConfigurationError
from .models import Program, project_state


def probability(value: Any, name: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise BackendError(f"Invalid probability/confidence field: {name}.")
    return float(value)


# Hard ceiling on accepted drift, whatever the label count implies. A distribution
# further off than this is treated as malformed rather than rounded.
MAX_SUM_DRIFT = 0.05


def quantization_step(values) -> float:
    """Coarsest decimal grid every value sits on exactly, or 0.0 if none does.

    The provider rounds Choice/Score probabilities before returning them, so the
    sum of a large distribution is off by the accumulated rounding, not by error.
    Detecting the grid lets the tolerance follow the data instead of being a
    constant that silently fits one label count.
    """
    for places in (1, 2, 3, 4):
        scale = 10**places
        if all(abs(v * scale - round(v * scale)) < 1e-9 for v in values):
            return 1.0 / scale
    return 0.0


def sum_tolerance(values) -> float:
    """Worst-case rounding drift for this many values on this grid, capped."""
    step = quantization_step(values)
    return min(MAX_SUM_DRIFT, max(1e-4, len(values) * step / 2))


def normalize_answers(program: Program, response: Response) -> dict[str, dict[str, Any]]:
    if set(response.answers) != set(program.questions):
        raise BackendError("Response question IDs differ from the request; refusing partial/unknown answers.")
    answers = {}
    for key, q in program.questions.items():
        answer = dict(response.answers[key])
        if answer.get("type") != q.type:
            raise BackendError(f"Answer type mismatch: {key}.")
        if q.type == "noul":
            p = probability(answer.get("noul"), key)
            # Do not manufacture a vendor confidence field for Noul.
            answers[key] = {"type": "noul", "noul": p}
            continue
        confidence = probability(answer.get("confidence"), key)
        raw = answer.get("probabilities")
        if not isinstance(raw, dict):
            raise BackendError(f"Missing probability distribution: {key}.")
        probs = {str(k): probability(v, key) for k, v in raw.items()}
        expected = set(q.criteria) if q.type == "choice" else {str(i) for i in range(len(q.criteria))}
        if set(probs) != expected:
            raise BackendError(f"Probability keys disagree with criteria: {key}.")
        total = sum(probs.values())
        if not math.isclose(total, 1.0, abs_tol=sum_tolerance(list(probs.values()))):
            raise BackendError(f"Probability distribution does not sum to one: {key}.")
        probs = {k: v / total for k, v in probs.items()}
        if q.type == "choice":
            choice = answer.get("choice")
            if choice not in probs or probs[choice] + 1e-6 < max(probs.values()):
                raise BackendError(f"Choice is inconsistent with probabilities: {key}.")
            answers[key] = {"type": "choice", "choice": choice,
                            "probabilities": probs, "confidence": confidence}
        else:
            score = answer.get("score")
            if type(score) not in (int, float) or not math.isfinite(score):
                raise BackendError(f"Invalid score: {key}.")
            mean = sum(int(i) * p for i, p in probs.items())
            if not math.isclose(score, mean, abs_tol=0.02) or not 0 <= score <= len(q.criteria) - 1:
                raise BackendError(f"Score is not the probability-weighted level mean: {key}.")
            legend = answer.get("legend")
            if not isinstance(legend, dict) or set(map(str, legend)) != expected:
                raise BackendError(f"Invalid Score legend: {key}.")
            answers[key] = {"type": "score", "score": float(score),
                            "probabilities": probs, "confidence": confidence, "legend": legend}
    return answers


def gate_for_answer(answer: dict[str, Any]) -> tuple[float, str]:
    if answer["type"] == "choice":
        return answer["probabilities"][answer["choice"]], "selected_option_probability"
    if answer["type"] == "noul":
        p = answer["noul"]
        return max(p, 1 - p), "binary_max_probability"
    return answer["confidence"], "vendor_score_confidence"


def apply_policy(program: Program, answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    decisions = {}
    for name, binding in program.bindings.items():
        declaration = program.decisions[name]
        policy = program.policies[name]
        distributions = None
        vendor_confidence = None
        p_true = None
        if binding.kind == "question":
            answer = answers[binding.question]
            gate, basis = gate_for_answer(answer)
            distributions = answer.get("probabilities")
            vendor_confidence = answer.get("confidence")
            if declaration.type == "choice":
                value = answer["choice"]
            elif declaration.type == "noul":
                p_true = answer["noul"]
                value = p_true >= policy.noul_threshold
                distributions = {"false": 1 - p_true, "true": p_true}
                # Gate the probability of the actual selected decision, not always max(p,1-p).
                gate = p_true if value else 1 - p_true
                basis = "selected_binary_outcome_probability"
            else:
                value = answer["score"]
        else:
            values, gates = [], []
            for qid, weight in binding.weights.items():
                answer = answers[qid]
                normalized = (answer["noul"] if answer["type"] == "noul" else
                              answer["score"] / (len(program.questions[qid].criteria) - 1))
                values.append(weight * normalized)
                gates.append(gate_for_answer(answer)[0])
            value = sum(values) / sum(binding.weights.values()) * (len(declaration.criteria) - 1)
            gate = min(gates)
            basis = "minimum_component_gate_heuristic"
            # No invented distribution or composite posterior confidence.
        decisions[name] = {
            "type": declaration.type, "value": value, "probabilities": distributions,
            "p_true": p_true, "vendor_confidence": vendor_confidence,
            "gate_score": gate, "gate_basis": basis,
            "review_required": policy.force_review or gate < policy.min_gate,
        }
    return decisions


class Runtime:
    def __init__(self, program: Program, backend: ManagedBackend, *, enforce_release: bool = False):
        self.program, self.backend = program, backend
        if enforce_release and not backend.synthetic:
            status = program.provenance.get("status", "draft")
            if status != "measured":
                raise ConfigurationError("Artifact is unvalidated or synthetic; use explicit experimental mode.")

    @classmethod
    def load(cls, path: str | Path, backend: ManagedBackend, **kwargs):
        return cls(Program.load(path), backend, **kwargs)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        projected = project_state(self.program.state, state)
        response = self.backend.evaluate(self.program, projected)
        answers = normalize_answers(self.program, response)
        self.backend.remember_validated(self.program, projected, response)
        return {
            "program_sha256": self.program.content_hash,
            "model": response.model, "synthetic": response.synthetic,
            "decisions": apply_policy(self.program, answers), "answers": answers,
            "usage": response.usage, "latency_ms": response.latency_ms, "cache_hit": response.cached,
        }
