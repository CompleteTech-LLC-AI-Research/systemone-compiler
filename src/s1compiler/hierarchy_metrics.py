"""End-to-end hierarchy metrics over fixed public outputs and root examples."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any

from .backends import ManagedBackend
from .data import Example, validate_example
from .errors import BackendError, DataError
from .hierarchy import HierarchyArtifact
from .hierarchy_data import HierarchySplitGuard
from .hierarchy_runtime import HierarchyRuntime
from .metrics import calibration_error, mean


class HierarchyEvaluationFailure(BackendError):
    """Provider or execution failure; never a zero-quality candidate."""

    def __init__(self, report: dict[str, Any], results: list[dict[str, Any]]):
        self.report, self.results = report, results
        super().__init__("Hierarchy evaluation failed; inspect partial results before retrying.")


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _correct(declaration, value: Any, gold: Any) -> bool:
    if declaration.type == "score":
        return abs(value - gold) <= declaration.score_tolerance
    return value == gold


def _value_quality(declaration, value: Any, gold: Any) -> float:
    if declaration.type == "score":
        return max(0.0, 1 - abs(value - gold) / (len(declaration.criteria) - 1))
    return float(value == gold)


def _complete_quality(source, row: Example, decisions: dict[str, Any]) -> float:
    total = sum(decision.weight for decision in source.decisions.values())
    return sum(declaration.weight * _value_quality(declaration, decisions[name]["value"], row.expected[name])
               for name, declaration in source.decisions.items()) / total


def _validate_final_prediction(declaration, prediction: dict[str, Any]) -> None:
    if not isinstance(prediction, dict) or prediction.get("distribution_scope") not in {
        "full_contract", "branch_conditional"
    }:
        raise DataError("Final hierarchy output lacks explicit probability scope.")
    value = prediction.get("value")
    if declaration.type == "choice":
        valid = isinstance(value, str) and value in declaration.criteria
    elif declaration.type == "noul":
        valid = type(value) is bool
    else:
        valid = type(value) in (int, float) and math.isfinite(value) and (
            0 <= value <= len(declaration.criteria) - 1
        )
    if not valid or type(prediction.get("review_required")) is not bool:
        raise DataError("Final hierarchy output violates the source decision contract.")
    if not isinstance(prediction.get("origin"), dict) or set(prediction["origin"]) != {"stage", "decision"}:
        raise DataError("Final hierarchy output lacks stage origin provenance.")
    if declaration.type == "choice" and prediction["distribution_scope"] == "full_contract":
        probabilities = prediction.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(declaration.criteria):
            raise DataError("Full-contract Choice probabilities must cover every public label.")
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
               for p in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-4):
            raise DataError("Full-contract Choice probabilities are invalid.")
    if declaration.type == "noul" and prediction["distribution_scope"] == "full_contract":
        p_true = prediction.get("p_true")
        if type(p_true) not in (int, float) or not math.isfinite(p_true) or not 0 <= p_true <= 1:
            raise DataError("Full-contract Noul output lacks P(true).")


def _validate_result(artifact: HierarchyArtifact, row: Example, result: dict[str, Any]) -> None:
    validate_example(row, artifact.source)
    if not isinstance(result, dict) or result.get("format") != "systemone-hierarchy-result/v1" or (
        result.get("graph_sha256") != artifact.content_hash or result.get("model") != artifact.source.model
    ):
        raise DataError("Hierarchy result differs from the frozen graph or model.")
    lineage = result.get("lineage")
    if not isinstance(lineage, dict) or lineage.get("root_id") != row.id or (
        lineage.get("group") != row.group or lineage.get("graph_sha256") != artifact.content_hash or
        lineage.get("split") not in {"train", "validation", "calibration", "test"}
    ):
        raise DataError("Hierarchy result is not bound to the expected root example.")
    status = result.get("status")
    if status not in {"completed", "review_required", "failed", "cancelled"}:
        raise DataError("Unknown hierarchy result status.")
    decisions = result.get("decisions")
    if not isinstance(decisions, dict) or not set(decisions) <= set(artifact.source.decisions):
        raise DataError("Hierarchy final decision names differ from the source contract.")
    for name, prediction in decisions.items():
        _validate_final_prediction(artifact.source.decisions[name], prediction)
    if status == "completed" and (set(decisions) != set(artifact.source.decisions) or any(
        item["review_required"] for item in decisions.values()
    )):
        raise DataError("Completed hierarchy result is missing or reviewing public outputs.")
    if status == "review_required" and set(decisions) == set(artifact.source.decisions) and not any(
        item["review_required"] for item in decisions.values()
    ):
        raise DataError("Review-required hierarchy result has no review or missing output.")
    if not isinstance(result.get("stages"), dict) or not isinstance(result.get("executed"), list) or (
        not isinstance(result.get("accounting"), dict)
    ):
        raise DataError("Hierarchy result lacks stage or accounting evidence.")
    stages = result["stages"]
    known = {stage.id for stage in [*artifact.nodes, *artifact.exports]}
    if set(stages) != known or any(not isinstance(state, dict) or state.get("status") not in {
        "completed", "skipped", "review_blocked", "pending", "failed", "cancelled"
    } for state in stages.values()):
        raise DataError("Hierarchy stage evidence does not cover the frozen graph.")
    executed, path = result["executed"], result.get("path")
    leaf_ids = {node.id for node in artifact.nodes}
    if (len(executed) != len(set(executed)) or any(stage not in leaf_ids or
            stages[stage]["status"] != "completed" for stage in executed) or
            not isinstance(path, list) or len(path) != len(set(path)) or
            any(stage not in known or stages[stage]["status"] != "completed" for stage in path)):
        raise DataError("Hierarchy path or native execution evidence is invalid.")
    for prediction in decisions.values():
        origin = prediction["origin"]
        if origin["stage"] not in known or stages[origin["stage"]]["status"] != "completed":
            raise DataError("Final output origin is not a completed graph stage.")
    accounting = result["accounting"]
    for key in ("requests_attempted", "cache_hits", "retries", "usage_unknown_calls"):
        if type(accounting.get(key)) is not int or accounting[key] < 0:
            raise DataError("Hierarchy accounting contains an invalid count.")
    if (accounting["retries"] > accounting["requests_attempted"] or
            type(accounting.get("latency_ms")) not in (int, float) or
            not math.isfinite(accounting["latency_ms"]) or accounting["latency_ms"] < 0):
        raise DataError("Hierarchy accounting contains invalid latency or retries.")
    for key in ("reported_input_tokens", "reported_output_tokens"):
        value = accounting.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise DataError("Hierarchy token usage is invalid.")
        if not accounting["usage_unknown_calls"] and value is None:
            raise DataError("Known usage must include token counts.")
    if not isinstance(result.get("synthetic"), bool) or not isinstance(result.get("backend"), str):
        raise DataError("Hierarchy result lacks backend provenance.")


def _probability_diagnostics(name: str, declaration,
                             items: list[tuple[Example, dict[str, Any] | None, str]]) -> dict[str, Any]:
    if declaration.type == "score":
        return {"eligible_n": 0, "brier": None, "log_loss": None, "calibration_ece_10": None,
                "unavailable_reason": "score_is_numeric_not_a_public_class_probability"}
    briers, losses, calibration = [], [], []
    reasons = Counter()
    for row, prediction, status in items:
        if status != "completed" or prediction is None:
            reasons["review_or_incomplete"] += 1
            continue
        if prediction["distribution_scope"] != "full_contract":
            reasons["branch_conditional_not_global"] += 1
            continue
        label = row.expected[name]
        if declaration.type == "choice":
            probabilities = prediction["probabilities"]
            briers.append(sum((p - float(key == label)) ** 2 for key, p in probabilities.items()))
            p_gold = probabilities[label]
            selected = prediction["value"]
            confidence = probabilities[selected]
        else:
            p_true = prediction["p_true"]
            briers.append((p_true - float(label)) ** 2)
            p_gold = p_true if label else 1 - p_true
            confidence = p_true if prediction["value"] else 1 - p_true
        losses.append(-math.log(max(1e-12, p_gold)))
        calibration.append((confidence, prediction["value"] == label))
    return {"eligible_n": len(briers), "brier": mean(briers), "log_loss": mean(losses),
            "calibration_ece_10": calibration_error(calibration),
            "excluded_n": sum(reasons.values()), "unavailable_reasons": dict(reasons),
            "scope": "eligible_full_contract_completed_outputs_only"}


def report_from_hierarchy_results(artifact: HierarchyArtifact, rows: list[Example],
                                  results: list[dict[str, Any]], *,
                                  oracle_paths: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Pure report; failed execution has no candidate-selection objective."""
    if not rows or len(rows) != len(results) or len({row.id for row in rows}) != len(rows):
        raise DataError("Hierarchy reporting requires aligned, unique root examples and results.")
    for row, result in zip(rows, results):
        _validate_result(artifact, row, result)
    if len({result["synthetic"] for result in results}) != 1 or (
        len({result["backend"] for result in results}) != 1
    ):
        raise DataError("Cannot mix synthetic/live or backend identities in one graph report.")
    statuses = Counter(result["status"] for result in results)
    execution_failed = bool(statuses["failed"] or statuses["cancelled"])
    completed = [(row, result) for row, result in zip(rows, results) if result["status"] == "completed"]
    quality_by_id = {row.id: (_complete_quality(artifact.source, row, result["decisions"])
                              if result["status"] == "completed" else 0.0)
                     for row, result in zip(rows, results)}
    objective = None if execution_failed else mean(list(quality_by_id.values()))
    per_output = {}
    origin_counts: dict[str, Counter] = {name: Counter() for name in artifact.source.decisions}
    path_errors = Counter()
    for name, declaration in artifact.source.decisions.items():
        accepted = [(row, result["decisions"][name]) for row, result in completed]
        correct = sum(_correct(declaration, prediction["value"], row.expected[name])
                      for row, prediction in accepted)
        summary = {"type": declaration.type, "n_root": len(rows), "completed_n": len(accepted),
                   "correct_n": correct, "all_root_accuracy_or_within_tolerance": correct / len(rows),
                   "completed_only_accuracy_or_within_tolerance": correct / len(accepted) if accepted else None,
                   "probability": _probability_diagnostics(
                       name, declaration, [(row, result["decisions"].get(name), result["status"])
                                           for row, result in zip(rows, results)])}
        if declaration.type == "score":
            differences = [prediction["value"] - row.expected[name] for row, prediction in accepted]
            summary.update(completed_only_mae=mean([abs(value) for value in differences]),
                           completed_only_rmse=(math.sqrt(mean([value * value for value in differences]))
                                                if differences else None),
                           score_tolerance=declaration.score_tolerance)
        per_output[name] = summary
    stage_counts: dict[str, Counter] = {stage.id: Counter() for stage in [*artifact.nodes, *artifact.exports]}
    path_counts, path_error_counts = Counter(), Counter()
    cluster_stage: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for row, result in zip(rows, results):
        path = " > ".join(result.get("path", [])) or "<no-complete-stage>"
        path_counts[path] += 1
        for stage_id, state in result["stages"].items():
            if stage_id not in stage_counts or not isinstance(state, dict) or "status" not in state:
                raise DataError("Hierarchy stage status does not match the frozen graph.")
            stage_counts[stage_id][state["status"]] += 1
            cluster_stage[row.group or row.id][stage_id][state["status"]] += 1
        if result["status"] == "completed":
            for name, declaration in artifact.source.decisions.items():
                prediction = result["decisions"][name]
                origin_counts[name][prediction["origin"]["stage"]] += 1
                if not _correct(declaration, prediction["value"], row.expected[name]):
                    path_errors[name] += 1
                    path_error_counts[path] += 1
    accounting = [result["accounting"] for result in results]
    attempts = [item["requests_attempted"] for item in accounting]
    graph_latency = [item["latency_ms"] for item in accounting]
    network_latency = [state["native_latency_ms"] for result in results for state in result["stages"].values()
                       if state.get("kind") == "leaf" and state.get("status") == "completed"
                       and not state.get("cache_hit") and state.get("native_latency_ms") is not None]
    unknown_usage = sum(item["usage_unknown_calls"] for item in accounting)
    input_tokens = (None if unknown_usage else sum(item["reported_input_tokens"] for item in accounting))
    output_tokens = (None if unknown_usage else sum(item["reported_output_tokens"] for item in accounting))
    if oracle_paths is not None:
        if set(oracle_paths) != {row.id for row in rows}:
            raise DataError("Oracle path diagnostics must align with all root IDs.")
        oracle = {"role": "diagnostic_only_not_primary", "route_match_n": sum(
            result["path"] == oracle_paths[row.id] for row, result in zip(rows, results)),
            "n_root": len(rows)}
    else:
        oracle = None
    return {"format": "systemone-hierarchy-metrics/v1", "graph_sha256": artifact.content_hash,
            "status": "execution_failed" if execution_failed else "complete",
            "synthetic": results[0]["synthetic"], "backend": results[0]["backend"],
            "n_root": len(rows), "n_clusters": len({row.group or row.id for row in rows}),
            "objective": objective,
            "objective_definition": "mean root utility: completed exact Choice/Noul correctness or "
                                    "normalized Score error, source-weighted; review/incomplete=0; "
                                    "execution failure has no score",
            "quality_by_root_id": None if execution_failed else quality_by_id,
            "coverage": {"completed_n": statuses["completed"], "review_n": statuses["review_required"],
                         "failed_n": statuses["failed"], "cancelled_n": statuses["cancelled"],
                         "completed_fraction": statuses["completed"] / len(rows)},
            "outputs": per_output,
            "routes": {"path_counts": dict(path_counts), "selected_origin_counts":
                       {name: dict(counts) for name, counts in origin_counts.items()},
                       "selected_path_error_n": dict(path_error_counts),
                       "final_output_error_n": dict(path_errors),
                       "oracle": oracle},
            "stages": {"status_counts": {name: dict(counts) for name, counts in stage_counts.items()},
                       "status_by_cluster": {cluster: {name: dict(counts) for name, counts in stages.items()}
                                             for cluster, stages in cluster_stage.items()}},
            "usage": {"native_attempts_total": sum(attempts), "native_attempts_mean": mean(attempts),
                      "native_attempts_p95": _p95(attempts),
                      "cache_hits_total": sum(item["cache_hits"] for item in accounting),
                      "retries_total": sum(item["retries"] for item in accounting),
                      "usage_unknown_calls": unknown_usage,
                      "reported_input_tokens": input_tokens,
                      "reported_output_tokens": output_tokens,
                      "dollar_cost": None},
            "latency_ms": {"graph_wall_mean": mean(graph_latency), "graph_wall_p95": _p95(graph_latency),
                           "uncached_native_mean": mean(network_latency),
                           "uncached_native_p95": _p95(network_latency),
                           "uncached_native_n": len(network_latency),
                           "uncached_native_unavailable_reason": ("all_leaf_responses_cached_or_no_success"
                                                                  if not network_latency else None)}}


def evaluate_hierarchy(artifact: HierarchyArtifact, rows: list[Example], backend: ManagedBackend, *,
                       guard: HierarchySplitGuard, split: str,
                       evidence_root: str | Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run each root once; stop on systemic failures before scoring a candidate."""
    runtime = HierarchyRuntime(artifact, backend)
    results = []
    for row in rows:
        bound = guard.bind(artifact, split, row)
        evidence_dir = Path(evidence_root) / row.id if evidence_root is not None else None
        result = runtime.run(row.state, lineage=bound, evidence_dir=evidence_dir)
        results.append(result)
        if result["status"] in {"failed", "cancelled"}:
            # A partial execution is evidence of failure, not a quality row.
            raise HierarchyEvaluationFailure({"status": "execution_failed", "failed_root_id": row.id,
                                              "graph_sha256": artifact.content_hash}, results)
    return report_from_hierarchy_results(artifact, rows, results), results


def replay_hierarchy_evaluation(artifact: HierarchyArtifact, rows: list[Example],
                                backend: ManagedBackend, *, guard: HierarchySplitGuard, split: str,
                                evidence_root: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Revalidate complete evidence with zero provider calls and recompute metrics."""
    runtime = HierarchyRuntime(artifact, backend)
    results = [runtime.run(row.state, lineage=guard.bind(artifact, split, row),
                           evidence_dir=Path(evidence_root) / row.id, evidence_mode="replay") for row in rows]
    return report_from_hierarchy_results(artifact, rows, results), results


def _flat_quality(source, row: Example, result: dict[str, Any]) -> float:
    if result.get("status") in {"failed", "cancelled"}:
        raise DataError("Flat baseline execution failed; no paired quality score is available.")
    if not isinstance(result.get("decisions"), dict) or set(result["decisions"]) != set(source.decisions):
        raise DataError("Flat baseline result lacks the fixed public output contract.")
    for name, declaration in source.decisions.items():
        prediction = result["decisions"][name]
        if not isinstance(prediction, dict) or type(prediction.get("review_required")) is not bool:
            raise DataError("Flat baseline decision lacks a review flag.")
        value = prediction.get("value")
        if declaration.type == "choice":
            valid = isinstance(value, str) and value in declaration.criteria
        elif declaration.type == "noul":
            valid = type(value) is bool
        else:
            valid = type(value) in (int, float) and math.isfinite(value) and (
                0 <= value <= len(declaration.criteria) - 1)
        if not valid:
            raise DataError("Flat baseline decision violates the public output contract.")
    if any(item["review_required"] for item in result["decisions"].values()):
        return 0.0
    return _complete_quality(source, row, result["decisions"])


def paired_flat_hierarchy(artifact: HierarchyArtifact, rows: list[Example],
                          hierarchy_runs: list[list[dict[str, Any]]],
                          flat_runs: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Descriptive paired seed/group comparison; inference belongs to the study protocol."""
    if not hierarchy_runs or len(hierarchy_runs) != len(flat_runs) or not rows:
        raise DataError("Paired comparison requires matching nonempty seeds and root examples.")
    expected = [(row.id, row.group, row.expected) for row in rows]
    clusters = sorted({row.group or row.id for row in rows})
    seed_deltas = []
    grouped_deltas: dict[str, list[float]] = {cluster: [] for cluster in clusters}
    stage_observations: dict[str, Counter] = {cluster: Counter() for cluster in clusters}
    for graph_results, flat_records in zip(hierarchy_runs, flat_runs):
        if len(graph_results) != len(rows) or len(flat_records) != len(rows) or [
            (record.get("id"), record.get("group"), record.get("gold")) for record in flat_records
        ] != expected:
            raise DataError("Flat and hierarchy seeds must use identical ordered roots, groups, and golds.")
        graph_report = report_from_hierarchy_results(artifact, rows, graph_results)
        if graph_report["objective"] is None:
            raise HierarchyEvaluationFailure(graph_report, graph_results)
        by_group: dict[str, list[float]] = defaultdict(list)
        for row, result, record in zip(rows, graph_results, flat_records):
            if not isinstance(record.get("result"), dict):
                raise DataError("Flat paired record lacks a result.")
            if record["result"].get("model") != artifact.source.model or (
                record["result"].get("synthetic") != result["synthetic"]
            ):
                raise DataError("Flat and hierarchy paired identities differ.")
            delta = graph_report["quality_by_root_id"][row.id] - _flat_quality(
                artifact.source, row, record["result"])
            cluster = row.group or row.id
            by_group[cluster].append(delta)
            for stage_id, state in result["stages"].items():
                stage_observations[cluster][(stage_id, state["status"])] += 1
        per_group = {cluster: mean(values) for cluster, values in by_group.items()}
        for cluster in clusters:
            grouped_deltas[cluster].append(per_group[cluster])
        seed_deltas.append(mean([delta for values in by_group.values() for delta in values]))
    return {"format": "systemone-hierarchy-paired/v1", "graph_sha256": artifact.content_hash,
            "n_root": len(rows), "n_clusters": len(clusters), "n_seeds": len(seed_deltas),
            "mean_delta_hierarchy_minus_flat": mean(seed_deltas),
            "per_seed_deltas": seed_deltas,
            "per_cluster_mean_deltas": {cluster: mean(values) for cluster, values in grouped_deltas.items()},
            "stage_observations_by_cluster": {cluster: {f"{stage}:{status}": count
                                                   for (stage, status), count in observations.items()}
                                              for cluster, observations in stage_observations.items()},
            "inference": None, "note": "Descriptive paired roots and groups; no significance claim."}
