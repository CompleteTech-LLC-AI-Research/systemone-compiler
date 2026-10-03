"""GEPA text optimization over complete, split-bound hierarchy executions."""
from __future__ import annotations

import copy
from typing import Any

from .backends import ManagedBackend
from .data import Example
from .errors import CandidateError, ConfigurationError, DataError
from .hierarchy import HierarchyArtifact, HierarchyProvenance
from .hierarchy_data import HierarchySplitGuard, HierarchyTeacherInputs
from .hierarchy_metrics import evaluate_hierarchy
from .hierarchy_validation import validate_hierarchy_artifact
from .io import canonical, fingerprint, json_loads


class _GraphTextCandidateError(CandidateError):
    """Fixed diagnostic vocabulary, never parser messages or candidate content."""

    def __init__(self, reason: str):
        super().__init__("Graph text candidate violates the frozen typed contract.")
        self.reason = reason


def _locations(artifact: HierarchyArtifact) -> dict[str, tuple[int, str, str, int | None]]:
    """Indexed criteria avoid collisions from arbitrary Choice label text."""
    locations = {}
    for index, node in sorted(enumerate(artifact.nodes), key=lambda item: item[1].id):
        for qid, question in sorted(node.program.questions.items()):
            prefix = f"graph/{node.id}/question/{qid}"
            locations[f"{prefix}/instructions"] = (index, qid, "instructions", None)
            if isinstance(question.criteria, dict):
                for offset, _ in enumerate(question.criteria):
                    locations[f"{prefix}/criterion/{offset}"] = (index, qid, "criterion", offset)
            elif isinstance(question.criteria, list):
                for offset, _ in enumerate(question.criteria):
                    locations[f"{prefix}/level/{offset}"] = (index, qid, "level", offset)
    return locations


def components_from_hierarchy(artifact: HierarchyArtifact) -> dict[str, str]:
    values = {}
    for key, (index, qid, section, offset) in _locations(artifact).items():
        question = artifact.nodes[index].program.questions[qid]
        if section == "instructions":
            value = question.instructions
        elif section == "criterion":
            value = list(question.criteria.values())[offset]
        else:
            value = question.criteria[offset]
        values[key] = canonical(value)
    return values


def hierarchy_text_structure_hash(artifact: HierarchyArtifact) -> str:
    """Hash every frozen field except editable prompt entries and provenance."""
    data = artifact.model_dump(mode="json", exclude={"provenance"})
    for node in data["nodes"]:
        node["program"].pop("provenance", None)
        for question in node["program"]["questions"].values():
            question["instructions"] = None
            criteria = question["criteria"]
            if isinstance(criteria, dict):
                question["criteria"] = {name: None for name in criteria}
            elif isinstance(criteria, list):
                question["criteria"] = [None] * len(criteria)
    return fingerprint(data)


def hierarchy_from_components(base: HierarchyArtifact, candidate: dict[str, str]) -> HierarchyArtifact:
    expected = components_from_hierarchy(base)
    if not isinstance(candidate, dict) or set(candidate) != set(expected):
        raise _GraphTextCandidateError("component_addresses")
    if candidate == expected:
        return base.model_copy(deep=True)
    data = base.model_dump(mode="json")
    reason = "typed_contract"
    try:
        for key, (index, qid, section, offset) in _locations(base).items():
            reason = "component_text"
            text = candidate[key]
            if not isinstance(text, str) or len(text) > 24000:
                raise ValueError("Invalid component text or size.")
            reason = "component_json"
            value = json_loads(text)
            reason = "prompt_entry_type"
            if value is not None and not isinstance(value, (str, dict, list)):
                raise ValueError("Invalid prompt entry type.")
            question = data["nodes"][index]["program"]["questions"][qid]
            if section == "instructions":
                question["instructions"] = value
            elif section == "criterion":
                label = list(base.nodes[index].program.questions[qid].criteria)[offset]
                question["criteria"][label] = value
            else:
                question["criteria"][offset] = value
        reason = "typed_contract"
        for node in data["nodes"]:
            node["program"]["provenance"] = {"status": "draft", "deployment_approved": False,
                                               "architect": "graph_text_optimization"}
        data["provenance"] = HierarchyProvenance(
            status="draft", deployment_approved=False,
            evidence={"optimized_from_sha256": base.content_hash,
                      "text_optimization": "gepa", "semantic_review_pending": True}
        ).model_dump(mode="json")
        result = HierarchyArtifact.model_validate(data)
        validate_hierarchy_artifact(result)
        if hierarchy_text_structure_hash(result) != hierarchy_text_structure_hash(base):
            raise ValueError("Graph topology or typed contract changed during text optimization.")
        return result
    except (ValueError, TypeError, KeyError, IndexError, DataError) as exc:
        raise _GraphTextCandidateError(reason) from exc


def _correct(declaration, predicted: Any, gold: Any) -> bool:
    if declaration.type == "score":
        return abs(predicted - gold) <= declaration.score_tolerance
    return predicted == gold


class HierarchyGEPAAdapter:
    """Score full root executions; expose only registered train traces to reflection."""

    def __init__(self, artifact: HierarchyArtifact, splits: dict[str, list[Example]],
                 backend: ManagedBackend, teacher, *, batch_factory=None):
        if set(splits) != {"train", "validation", "calibration", "test"}:
            raise DataError("Graph GEPA requires all four registered splits.")
        self.artifact, self.backend, self.teacher = artifact, backend, teacher
        self.splits = {name: copy.deepcopy(rows) for name, rows in splits.items()}
        HierarchySplitGuard(artifact, self.splits)
        self.registered = {row.id: (name, fingerprint(row.model_dump(mode="json")))
                           for name, rows in self.splits.items() for row in rows}
        self.batch_factory = batch_factory
        self.invalid_candidates = 0
        self.metric_rows = 0
        self.evaluation_batches = 0
        self.native_start = backend.budget.used
        self._issued_traces: set[tuple[str, str]] = set()
        self._issued_reflections: set[tuple[str, str]] = set()

    def _factory(self):
        if self.batch_factory is not None:
            return self.batch_factory
        try:
            from gepa.core.adapter import EvaluationBatch
        except ImportError as exc:
            raise ConfigurationError("GEPA is missing; install the optimize extra.") from exc
        return EvaluationBatch

    def _split(self, batch: list[Example], capture_traces: bool) -> str:
        if not batch:
            raise DataError("GEPA graph evaluation requires a nonempty batch.")
        splits = set()
        for row in batch:
            expected = self.registered.get(row.id)
            if expected is None or expected[1] != fingerprint(row.model_dump(mode="json")):
                raise DataError("GEPA batch contains an unregistered or changed root example.")
            splits.add(expected[0])
        if len(splits) != 1 or not splits <= {"train", "validation"}:
            raise DataError("GEPA batch must be entirely registered train or validation examples.")
        split = next(iter(splits))
        if capture_traces and split != "train":
            raise DataError("GEPA reflection requires registered train examples only.")
        return split

    def evaluate(self, batch, candidate, capture_traces=False):
        rows = list(batch)
        split = self._split(rows, capture_traces)
        factory = self._factory()
        self.evaluation_batches += 1
        self.metric_rows += len(rows)
        try:
            artifact = hierarchy_from_components(self.artifact, candidate)
        except CandidateError as exc:
            self.invalid_candidates += 1
            traces = ([{"error": "invalid_typed_graph_text_candidate",
                        "reason": getattr(exc, "reason", "typed_contract")} for _ in rows]
                      if capture_traces else None)
            if capture_traces:
                self._issued_traces.add((fingerprint(candidate), fingerprint(traces)))
            return factory(outputs=[{"error": "invalid_candidate"} for _ in rows],
                           scores=[0.0] * len(rows), trajectories=traces)
        guard = HierarchySplitGuard(artifact, self.splits)
        report, results = evaluate_hierarchy(artifact, rows, self.backend, guard=guard, split=split)
        scores = [report["quality_by_root_id"][row.id] for row in rows]
        traces = None
        if capture_traces:
            source_traces = HierarchyTeacherInputs(guard).traces(results)
            traces = []
            nodes = {node.id: node for node in artifact.nodes}
            for row, result, source_trace in zip(rows, results, source_traces):
                final_errors = [name for name, declaration in artifact.source.decisions.items()
                                if name not in result["decisions"] or not _correct(
                                    declaration, result["decisions"][name]["value"], row.expected[name])]
                annotated_errors = []
                for annotation in source_trace["human_intermediate_labels"]:
                    prediction = source_trace["stage_predictions"].get(
                        annotation["stage"], {}).get(annotation["decision"])
                    if prediction is not None and not _correct(
                        nodes[annotation["stage"]].program.decisions[annotation["decision"]],
                        prediction["value"], annotation["value"]):
                        annotated_errors.append({"stage": annotation["stage"],
                                                 "decision": annotation["decision"]})
                traces.append({**source_trace, "root_quality": report["quality_by_root_id"][row.id],
                               "final_errors": final_errors, "annotated_stage_errors": annotated_errors,
                               "visited_stages": result["executed"],
                               "stage_statuses": {key: value["status"] for key, value in result["stages"].items()},
                               "path": result["path"]})
            self._issued_traces.add((fingerprint(candidate), fingerprint(traces)))
        return factory(outputs=results, scores=scores, trajectories=traces)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        if (eval_batch.trajectories is None or
                (fingerprint(candidate), fingerprint(eval_batch.trajectories)) not in self._issued_traces):
            raise DataError("Graph reflection requires captured registered train traces.")
        addresses = _locations(self.artifact)
        if not components_to_update or any(key not in addresses for key in components_to_update):
            raise CandidateError("Reflection requested an unknown graph text component.")
        result = {}
        for key in components_to_update:
            stage_id = self.artifact.nodes[addresses[key][0]].id
            result[key] = [{"Inputs": trace.get("input", {}),
                            "Generated Outputs": {"final": trace.get("final_predictions", {}),
                                                  "stage": trace.get("stage_predictions", {}).get(stage_id)},
                            "Feedback": {"root_quality": trace.get("root_quality"),
                                         "error": trace.get("error"),
                                         "reason": trace.get("reason"),
                                         "final_errors": trace.get("final_errors", []),
                                         "annotated_stage_errors": [entry for entry in
                                                                    trace.get("annotated_stage_errors", [])
                                                                    if entry["stage"] == stage_id],
                                         "visited": stage_id in trace.get("visited_stages", []),
                                         "stage_status": trace.get("stage_statuses", {}).get(stage_id),
                                         "path": trace.get("path", [])}}
                           for trace in eval_batch.trajectories]
        self._issued_reflections.add((fingerprint(candidate), fingerprint(result)))
        return result

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        if (not isinstance(candidate, dict) or not components_to_update or
                any(key not in candidate or key not in _locations(self.artifact)
                    for key in components_to_update)):
            raise CandidateError("Proposal requested a missing or unknown graph text component.")
        if (fingerprint(candidate), fingerprint(reflective_dataset)) not in self._issued_reflections:
            raise DataError("Teacher reflection did not originate from registered train traces.")
        try:
            proposed = self.teacher.propose_components(candidate, reflective_dataset, components_to_update)
            if not isinstance(proposed, dict) or set(proposed) != set(components_to_update):
                raise CandidateError("Proposer changed unrequested graph component addresses.")
            hierarchy_from_components(self.artifact, {**candidate, **proposed})
            return proposed
        except CandidateError:
            self.invalid_candidates += 1
            return {key: candidate[key] for key in components_to_update}


def optimize_hierarchy_gepa(artifact: HierarchyArtifact, splits: dict[str, list[Example]],
                            backend: ManagedBackend, teacher, *, max_metric_calls: int = 128,
                            seed: int = 7) -> tuple[HierarchyArtifact, dict[str, Any]]:
    """Run the installed GEPA engine; backend budget meters actual stage calls."""
    if artifact.final_review_gates:
        raise ConfigurationError("Optimize graph text before final review-gate calibration.")
    if max_metric_calls < len(splits.get("validation", [])) + 2:
        raise ConfigurationError("GEPA metric budget must exceed initial validation evaluation.")
    try:
        import gepa
    except ImportError as exc:
        raise ConfigurationError("Install the optimize extra: pip install -e '.[optimize]'.") from exc
    adapter = HierarchyGEPAAdapter(artifact, splits, backend, teacher)
    class _NoPromptLogger:
        def log(self, message: str) -> None:
            # GEPA's default logger prints candidate text. Do not retain it.
            pass

    result = gepa.optimize(
        seed_candidate=components_from_hierarchy(artifact),
        trainset=splits["train"], valset=splits["validation"], adapter=adapter,
        max_metric_calls=max_metric_calls, reflection_minibatch_size=min(3, len(splits["train"])),
        seed=seed, use_merge=False, skip_perfect_score=False, display_progress_bar=False,
        raise_on_exception=True, logger=_NoPromptLogger())
    best = hierarchy_from_components(artifact, result.best_candidate)
    return best, {"engine": "gepa", "seed": seed,
                  "input_graph_sha256": artifact.content_hash,
                  "metric_call_budget": max_metric_calls,
                  "metric_rows_evaluated": adapter.metric_rows,
                  "evaluation_batches": adapter.evaluation_batches,
                  "native_stage_calls": backend.budget.used - adapter.native_start,
                  "invalid_mutations_rejected": adapter.invalid_candidates,
                  "selected_graph_sha256": best.content_hash,
                  "structure_sha256": hierarchy_text_structure_hash(artifact)}
