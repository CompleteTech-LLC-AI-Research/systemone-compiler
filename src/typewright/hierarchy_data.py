"""Split-bound hierarchy data, derived-stage lineage, and teacher input gates."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .data import Example, assert_disjoint, dataset_hash, validate_example
from .errors import DataError
from .hierarchy import HierarchyArtifact, LoweredNode, RootRef
from .io import fingerprint, json_loads
from .models import StrictModel, project_state


SPLITS = frozenset({"train", "validation", "calibration", "test"})


class IntermediateAnnotation(StrictModel):
    """Optional sourced human label, never a model prediction."""

    stage: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    value: Any
    source: Literal["human"]
    annotator: str = Field(min_length=1)


class HierarchyExample(Example):
    annotations: list[IntermediateAnnotation] = Field(default_factory=list)


def read_hierarchy_jsonl(path: str | Path, artifact: HierarchyArtifact) -> list[HierarchyExample]:
    """Read optional human stage labels separately from final gold labels."""
    rows = []
    nodes = {node.id: node for node in artifact.nodes}
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if len(line) > 1_000_000:
                raise DataError(f"Hierarchy JSONL line {line_number} exceeds the 1 MB limit.")
            try:
                row = HierarchyExample.model_validate(json_loads(line))
                validate_example(row, artifact.source)
                for annotation in row.annotations:
                    _check_annotation(annotation, nodes)
            except Exception as exc:
                raise DataError(f"Invalid hierarchy record at line {line_number} of {Path(path).name}.") from exc
            rows.append(row)
    if not rows:
        raise DataError(f"Empty hierarchy dataset: {Path(path).name}")
    return rows


def _check_annotation(annotation: IntermediateAnnotation, nodes: dict[str, LoweredNode]) -> None:
    node = nodes.get(annotation.stage)
    if node is None or annotation.decision not in node.program.decisions:
        raise DataError("Intermediate annotation names an unknown frozen stage decision.")
    declaration = node.program.decisions[annotation.decision]
    value = annotation.value
    if declaration.type == "choice":
        valid = isinstance(value, str) and value in declaration.criteria
    elif declaration.type == "noul":
        valid = type(value) is bool
    else:
        valid = type(value) in (int, float) and math.isfinite(value) and (
            0 <= value <= len(declaration.criteria) - 1
        )
    if not valid:
        raise DataError("Intermediate human annotation violates the stage decision contract.")


def _root_condition_may_run(node: LoweredNode, root: dict[str, Any]) -> bool:
    if node.when is None:
        return True
    for predicate in node.when.all:
        if not isinstance(predicate.ref, RootRef):
            continue  # Model-derived route is unknown before execution.
        value = root.get(predicate.ref.root, predicate.ref.default)
        if value is None:
            return True
        expected = predicate.value
        accepted = {"eq": lambda: value == expected, "in": lambda: value in expected,
                    "lt": lambda: value < expected, "lte": lambda: value <= expected,
                    "gt": lambda: value > expected, "gte": lambda: value >= expected}[predicate.op]()
        if not accepted:
            return False
    return True


def _static_projection(node: LoweredNode, root: dict[str, Any]) -> dict[str, Any] | None:
    if not all(isinstance(ref, RootRef) for ref in node.inputs.values()):
        return None
    if not _root_condition_may_run(node, root):
        return None
    mapped = {}
    for port, ref in node.inputs.items():
        if ref.root in root:
            mapped[port] = root[ref.root]
        elif ref.default is not None:
            mapped[port] = ref.default
        elif node.program.state[port].required:
            # A model-derived condition may skip this stage; defer the actual
            # admission check until the stage becomes eligible at runtime.
            return None
    return project_state(node.program.state, mapped)


class HierarchySplitGuard:
    """Candidate-specific preflight and runtime checks for four disjoint splits."""

    def __init__(self, artifact: HierarchyArtifact, splits: dict[str, list[Example]]):
        if set(splits) != SPLITS:
            raise DataError("Hierarchy requires train, validation, calibration, and test splits.")
        assert_disjoint(splits, artifact.source)
        self.graph_sha256 = artifact.content_hash
        self.source = artifact.source
        self.nodes = {node.id: node for node in artifact.nodes}
        self.splits = {name: tuple(copy.deepcopy(rows)) for name, rows in splits.items()}
        self.by_id: dict[str, tuple[str, Example, str]] = {}
        self.static_inputs: dict[tuple[str, str], str] = {}
        self._result_hashes: dict[str, str] = {}
        self._train_stage_states: dict[str, dict[str, dict[str, Any]]] = {}
        self._train_result_status: dict[str, str] = {}
        projection_versions: dict[str, list[dict[str, str]]] = {node_id: [] for node_id in self.nodes}
        seen_static: dict[tuple[str, str], tuple[str, str]] = {}
        for split, rows in self.splits.items():
            for row in rows:
                if isinstance(row, HierarchyExample):
                    seen_annotations = set()
                    for annotation in row.annotations:
                        key = (annotation.stage, annotation.decision)
                        if key in seen_annotations:
                            raise DataError("Duplicate human annotation for one stage decision.")
                        seen_annotations.add(key)
                        _check_annotation(annotation, self.nodes)
                root = project_state(artifact.source.state, row.state)
                self.by_id[row.id] = (split, row, fingerprint(row.model_dump(mode="json")))
                for node in artifact.nodes:
                    projected = _static_projection(node, root)
                    if projected is None:
                        continue
                    digest = fingerprint(projected)
                    key = (node.id, digest)
                    if key in seen_static:
                        other_split, _ = seen_static[key]
                        raise DataError(f"Duplicate static stage projection for {node.id} in "
                                        f"{other_split} and {split}.")
                    seen_static[key] = (split, row.id)
                    self.static_inputs[(row.id, node.id)] = digest
                    if split == "train":
                        projection_versions[node.id].append({"root_id": row.id, "sha256": digest})
        self.versions = {"graph_sha256": self.graph_sha256,
                         "datasets": {name: {"sha256": dataset_hash(list(rows)), "n": len(rows)}
                                      for name, rows in self.splits.items()},
                         "train_projections": {name: fingerprint(entries)
                                               for name, entries in projection_versions.items()},
                         "observed_train_projections": self._observed_versions()}

    def _observed_versions(self) -> dict[str, Any]:
        entries = {stage_id: [] for stage_id in self.nodes}
        for root_id, states in sorted(self._train_stage_states.items()):
            for stage_id, state in sorted(states.items()):
                entries[stage_id].append({"root_id": root_id, "sha256": fingerprint(state)})
        expected = {row.id for row in self.splits["train"]}
        return {"complete": set(self._train_result_status) == expected and all(
                    status in {"completed", "review_required"} for status in self._train_result_status.values()),
                "rows_seen": len(self._train_result_status),
                "stages": {stage_id: {"n": len(values), "sha256": fingerprint(values)}
                           for stage_id, values in entries.items()}}

    def bind(self, artifact: HierarchyArtifact, split: str, row: Example) -> BoundHierarchyExample:
        if artifact.content_hash != self.graph_sha256:
            raise DataError("Graph candidate changed; rerun split and projection validation.")
        original = self.by_id.get(row.id)
        if original is None or original[0] != split or original[2] != fingerprint(row.model_dump(mode="json")):
            raise DataError("Example identity, split, group, state, or labels changed after preflight.")
        return BoundHierarchyExample(self, split, copy.deepcopy(original[1]))

    def registered_train_result(self, result: dict[str, Any]) -> Example:
        lineage = result.get("lineage")
        if not isinstance(lineage, dict) or lineage.get("split") != "train":
            raise DataError("Teacher feedback may contain only registered train results.")
        root_id = lineage.get("root_id")
        original = self.by_id.get(root_id)
        if original is None or original[0] != "train" or (
            self._result_hashes.get(root_id) != fingerprint(result)
        ):
            raise DataError("Teacher feedback result is not the sealed train execution.")
        if result.get("status") in {"failed", "cancelled"}:
            raise DataError("Failed or cancelled graph execution is not teacher quality feedback.")
        return original[1]


@dataclass
class BoundHierarchyExample:
    guard: HierarchySplitGuard
    split: str
    row: Example
    stage_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    stage_predictions: dict[str, dict[str, Any]] = field(default_factory=dict)
    _stage_states: dict[str, dict[str, Any]] = field(default_factory=dict)

    def assert_run(self, artifact: HierarchyArtifact, state: dict[str, Any]) -> None:
        if artifact.content_hash != self.guard.graph_sha256 or (
            fingerprint(project_state(artifact.source.state, state)) !=
            fingerprint(project_state(artifact.source.state, self.row.state))
        ):
            raise DataError("Bound hierarchy example differs from the validated graph or root state.")

    def validate_stage(self, node: LoweredNode, state: dict[str, Any],
                       values: dict[str, dict[str, dict[str, Any]]]) -> None:
        if node.id not in self.guard.nodes or node.program.content_hash != (
            self.guard.nodes[node.id].program.content_hash
        ):
            raise DataError("Stage changed after split preflight.")
        digest = fingerprint(project_state(node.program.state, state))
        root = project_state(self.guard.source.state, self.row.state)
        for port, ref in node.inputs.items():
            if isinstance(ref, RootRef):
                source_value = root.get(ref.root)
            else:
                source_value = values.get(ref.stage, {}).get(ref.decision, {}).get(ref.field)
            if source_value is None:
                source_value = ref.default
            if (port in state) != (source_value is not None) or (
                port in state and state[port] != source_value
            ):
                raise DataError("Derived stage input differs from validated root or predecessor output.")
        static = self.guard.static_inputs.get((self.row.id, node.id))
        if static is not None and digest != static:
            raise DataError("Static stage projection changed before dispatch.")
        prior = self.stage_inputs.get(node.id)
        if prior is not None and prior["input_sha256"] != digest:
            raise DataError("Stage input changed within one root example.")
        self.stage_inputs[node.id] = {"input_sha256": digest,
                                      "origin": "root_projection" if static else "derived_or_routed"}
        if self.split == "train":
            self._stage_states[node.id] = copy.deepcopy(state)

    def record_prediction(self, node_id: str, decisions: dict[str, Any]) -> None:
        if node_id not in self.stage_inputs:
            raise DataError("Cannot record a stage prediction before its input is validated.")
        self.stage_predictions[node_id] = copy.deepcopy(decisions)

    def seal(self, result: dict[str, Any]) -> None:
        lineage = {"root_id": self.row.id, "group": self.row.group, "split": self.split,
                   "graph_sha256": self.guard.graph_sha256,
                   "stage_inputs": copy.deepcopy(self.stage_inputs),
                   "stage_predictions": copy.deepcopy(self.stage_predictions),
                   "stage_output_kind": "model_predictions_not_labels"}
        result["lineage"] = lineage
        self.guard._result_hashes[self.row.id] = fingerprint(result)
        if self.split == "train":
            self.guard._train_stage_states[self.row.id] = copy.deepcopy(self._stage_states)
            self.guard._train_result_status[self.row.id] = result["status"]
            self.guard.versions["observed_train_projections"] = self.guard._observed_versions()


class HierarchyTeacherInputs:
    """Construct proposal/reflection payloads only from sealed train executions."""

    def __init__(self, guard: HierarchySplitGuard):
        self.guard = guard

    def examples(self, rows: list[Example]) -> list[dict[str, Any]]:
        payload = []
        for row in rows:
            split, original, digest = self.guard.by_id.get(row.id, (None, None, None))
            if split != "train" or digest != fingerprint(row.model_dump(mode="json")):
                raise DataError("Teacher examples must come from the validated train split.")
            payload.append({"root_id": row.id, "group": row.group,
                            "input": project_state(self.guard.source.state, original.state),
                            "expected_final": copy.deepcopy(original.expected),
                            "human_intermediate_labels": [item.model_dump(mode="json") for item in
                                                          getattr(original, "annotations", [])]})
        return payload

    def traces(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = []
        for result in results:
            row = self.guard.registered_train_result(result)
            payload.append({"root_id": row.id, "group": row.group,
                            "status": result["status"],
                            "input": project_state(self.guard.source.state, row.state),
                            "expected_final": copy.deepcopy(row.expected),
                            "final_predictions": copy.deepcopy(result["decisions"]),
                            "stage_predictions": copy.deepcopy(result["lineage"]["stage_predictions"]),
                            "stage_inputs": copy.deepcopy(result["lineage"]["stage_inputs"]),
                            "train_stage_states": copy.deepcopy(self.guard._train_stage_states[row.id]),
                            "human_intermediate_labels": [item.model_dump(mode="json") for item in
                                                          getattr(row, "annotations", [])]})
        return payload

    def reflection(self, results: list[dict[str, Any]], components: list[str]) -> dict[str, list[dict[str, Any]]]:
        traces = self.traces(results)
        return {component: copy.deepcopy(traces) for component in components}

    def routed_train_subset(self, stage_id: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if stage_id not in self.guard.nodes:
            raise DataError("Unknown routed stage.")
        seen = set()
        selected = []
        for result, trace in zip(results, self.traces(results)):
            stage = result["lineage"]["stage_inputs"].get(stage_id)
            if stage is None:
                continue
            group = trace["group"] or trace["root_id"]
            key = (group, stage["input_sha256"], fingerprint(trace["expected_final"]),
                   fingerprint(trace["human_intermediate_labels"]),
                   fingerprint(trace["stage_predictions"].get(stage_id)))
            if key not in seen:
                seen.add(key)
                selected.append(trace)
        return selected
