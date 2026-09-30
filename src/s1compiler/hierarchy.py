"""Versioned, declarative hierarchy source and portable frozen artifacts.

Semantic graph analysis is extended by the validator; this module owns strict
wire shapes, bounded definition expansion, integrity, and provenance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from .errors import DataError
from .io import atomic_json, fingerprint, load_document
from .models import Decision, ID, Program, StateField, StrictModel, UseCase


def _identifier(value: str) -> str:
    if not ID.fullmatch(value):
        raise ValueError("Hierarchy identifiers must be lowercase snake_case.")
    return value


def _qualified(value: str) -> str:
    if not value or any(not ID.fullmatch(segment) for segment in value.split("/")):
        raise ValueError("Qualified hierarchy IDs require safe slash-separated identifiers.")
    return value


def _origin(value: str) -> str:
    parts = value.split(":")
    if len(parts) != 2 or (parts[0] != "root" and not _qualified(parts[0])) or not ID.fullmatch(parts[1]):
        raise ValueError("Source origin requires a local definition path and stage ID.")
    return value


class RootRef(StrictModel):
    root: str
    default: JsonValue | None = None

    _check_root = field_validator("root")(_identifier)


class StageRef(StrictModel):
    stage: str
    decision: str
    field: Literal["value", "probabilities", "p_true", "vendor_confidence", "gate_score", "review_required"]
    default: JsonValue | None = None

    _check_stage = field_validator("stage")(_qualified)
    _check_decision = field_validator("decision")(_identifier)


Reference = RootRef | StageRef


class Predicate(StrictModel):
    ref: Reference
    op: Literal["eq", "in", "lt", "lte", "gt", "gte"]
    value: JsonValue


class Condition(StrictModel):
    all: list[Predicate] = Field(min_length=1)


class Candidate(StrictModel):
    stage: str
    decision: str
    label_map: dict[str, str] | None = None
    distribution_scope: Literal["full_contract", "branch_conditional"]

    _check_stage = field_validator("stage")(_qualified)
    _check_decision = field_validator("decision")(_identifier)


class FinalMapping(StrictModel):
    candidates: list[Candidate] = Field(min_length=1)
    on_missing: Literal["review_required"]


class LeafStage(StrictModel):
    id: str
    kind: Literal["leaf"]
    program: Program
    inputs: dict[str, Reference]
    when: Condition | None = None
    after: list[str] = Field(default_factory=list)
    on_review: Literal["defer", "continue_marked"] = "defer"

    _check_id = field_validator("id")(_identifier)

    @field_validator("after")
    @classmethod
    def check_after(cls, values: list[str]) -> list[str]:
        return [_identifier(value) for value in values]


class SubgraphStage(StrictModel):
    id: str
    kind: Literal["subgraph"]
    definition: str
    inputs: dict[str, Reference]
    when: Condition | None = None
    after: list[str] = Field(default_factory=list)
    on_review: Literal["defer", "continue_marked"] = "defer"

    _check_id = field_validator("id")(_identifier)
    _check_definition = field_validator("definition")(_identifier)

    @field_validator("after")
    @classmethod
    def check_after(cls, values: list[str]) -> list[str]:
        return [_identifier(value) for value in values]


Stage = Annotated[LeafStage | SubgraphStage, Field(discriminator="kind")]


class Graph(StrictModel):
    inputs: dict[str, StateField]
    outputs: dict[str, Decision]
    stages: list[Stage] = Field(min_length=1, max_length=64)
    final: dict[str, FinalMapping]

    @model_validator(mode="after")
    def check_names(self):
        if not self.inputs or not self.outputs or not self.stages:
            raise ValueError("Graph requires inputs, outputs, and stages.")
        names = [stage.id for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate stage IDs are prohibited.")
        if any(not ID.fullmatch(name) for name in names):
            raise ValueError("Stage IDs must be lowercase snake_case.")
        if set(self.final) != set(self.outputs):
            raise ValueError("Every graph output needs exactly one final mapping.")
        for name in (*self.inputs, *self.outputs):
            if not ID.fullmatch(name):
                raise ValueError("Graph port IDs must be lowercase snake_case.")
        return self


class Limits(StrictModel):
    max_expanded_nodes: int = Field(default=64, strict=True, gt=0, le=64)
    max_depth: int = Field(default=8, strict=True, gt=0, le=8)
    max_native_calls_per_example: int = Field(default=64, strict=True, gt=0, le=64)


class HierarchySource(StrictModel):
    format: Literal["systemone-hierarchy-source/v1"]
    source: UseCase
    limits: Limits = Field(default_factory=Limits)
    definitions: dict[str, Graph] = Field(default_factory=dict, max_length=64)
    graph: Graph

    @model_validator(mode="after")
    def check_root(self):
        if self.graph.inputs != self.source.state or self.graph.outputs != self.source.decisions:
            raise ValueError("Root graph interface must equal the source contract.")
        if any(name == "root" or not ID.fullmatch(name) for name in self.definitions):
            raise ValueError("Definition IDs must be lowercase snake_case and cannot use reserved root.")

        checked_depth: dict[str, int] = {}

        def check_definitions(graph: Graph, active: tuple[str, ...]) -> None:
            for stage in graph.stages:
                if not isinstance(stage, SubgraphStage):
                    continue
                if stage.definition in active:
                    raise ValueError("Recursive hierarchy definition.")
                child = self.definitions.get(stage.definition)
                if child is None:
                    raise ValueError(f"Missing local definition: {stage.definition}")
                if len(active) + 1 > self.limits.max_depth:
                    raise ValueError("Hierarchy depth limit exceeded.")
                if checked_depth.get(stage.definition, -1) >= len(active) + 1:
                    continue
                check_definitions(child, (*active, stage.definition))
                checked_depth[stage.definition] = len(active) + 1

        check_definitions(self.graph, ())
        for name, definition in self.definitions.items():
            check_definitions(definition, (name,))
        return self

    @classmethod
    def load(cls, path: str | Path) -> HierarchySource:
        return cls.model_validate(load_document(path))


class LoweredNode(StrictModel):
    id: str
    source_id: str
    program: Program
    inputs: dict[str, Reference]
    when: Condition | None = None
    after: list[str] = Field(default_factory=list)
    on_review: Literal["defer", "continue_marked"] = "defer"

    _check_id = field_validator("id")(_qualified)
    _check_source_id = field_validator("source_id")(_origin)

    @field_validator("after")
    @classmethod
    def check_after(cls, values: list[str]) -> list[str]:
        return [_qualified(value) for value in values]


class LoweredExport(StrictModel):
    id: str
    input_contracts: dict[str, StateField]
    output_contracts: dict[str, Decision]
    outputs: dict[str, FinalMapping]
    inputs: dict[str, Reference]
    when: Condition | None = None
    after: list[str] = Field(default_factory=list)
    on_review: Literal["defer", "continue_marked"] = "defer"

    _check_id = field_validator("id")(_qualified)

    @field_validator("after")
    @classmethod
    def check_after(cls, values: list[str]) -> list[str]:
        return [_qualified(value) for value in values]


class HierarchyProvenance(StrictModel):
    status: Literal["draft", "synthetic", "measured"] = "draft"
    composition_measured: bool = False
    deployment_approved: bool = False
    evidence: dict[str, JsonValue] = Field(default_factory=dict)


class HierarchyArtifact(StrictModel):
    format: Literal["systemone-hierarchy/v1"]
    source: UseCase
    limits: Limits
    nodes: list[LoweredNode]
    exports: list[LoweredExport]
    final: dict[str, FinalMapping]
    source_to_nodes: dict[str, list[str]]
    provenance: HierarchyProvenance = Field(default_factory=HierarchyProvenance)

    @model_validator(mode="after")
    def check_artifact(self):
        ids = [node.id for node in self.nodes]
        export_ids = [export.id for export in self.exports]
        if len(ids) != len(set(ids)) or len(ids) > self.limits.max_expanded_nodes:
            raise ValueError("Invalid or over-limit expanded node IDs.")
        if not ids or len(export_ids) != len(set(export_ids)) or set(ids) & set(export_ids):
            raise ValueError("Expanded node and export IDs must be distinct and nonempty.")
        if len(ids) + len(export_ids) > self.limits.max_expanded_nodes:
            raise ValueError("Expanded stage count exceeds the artifact limit.")
        if len(ids) > self.limits.max_native_calls_per_example:
            raise ValueError("Expanded native calls exceed the artifact limit.")
        if any(node.program.model != self.source.model for node in self.nodes):
            raise ValueError("Leaf target model differs from source model.")
        if set(self.final) != set(self.source.decisions):
            raise ValueError("Artifact final mappings differ from the source contract.")
        output_ports = {node.id: node.program.decisions for node in self.nodes}
        output_ports.update({export.id: export.output_contracts for export in self.exports})
        for node in self.nodes:
            if not set(node.inputs) <= set(node.program.state) or any(
                field.required and port not in node.inputs for port, field in node.program.state.items()
            ):
                raise ValueError(f"Leaf input ports differ from frozen program: {node.id}")
        for export in self.exports:
            if not set(export.inputs) <= set(export.input_contracts) or any(
                field.required and port not in export.inputs for port, field in export.input_contracts.items()
            ):
                raise ValueError(f"Subgraph export inputs differ from their contract: {export.id}")
            if set(export.outputs) != set(export.output_contracts):
                raise ValueError(f"Subgraph export outputs differ from their contract: {export.id}")
        for node in [*self.nodes, *self.exports]:
            refs = list(node.inputs.values())
            if node.when:
                refs.extend(predicate.ref for predicate in node.when.all)
            for ref in refs:
                if isinstance(ref, RootRef):
                    if ref.root not in self.source.state:
                        raise ValueError(f"Unknown root state reference in {node.id}.")
                elif ref.stage not in output_ports or ref.decision not in output_ports[ref.stage]:
                    raise ValueError(f"Unknown stage output reference in {node.id}.")
            if any(after not in output_ports or after == node.id for after in node.after):
                raise ValueError(f"Invalid after edge in {node.id}.")
        for mappings in [self.final, *(export.outputs for export in self.exports)]:
            for final in mappings.values():
                for candidate in final.candidates:
                    if candidate.stage not in output_ports or candidate.decision not in output_ports[candidate.stage]:
                        raise ValueError("Final mapping references an unknown stage output.")
        mapped = [node_id for node_ids in self.source_to_nodes.values() for node_id in node_ids]
        if sorted(mapped) != sorted(ids):
            raise ValueError("Source-to-node mapping does not cover each expanded node once.")
        if set(self.source_to_nodes) != {node.source_id for node in self.nodes}:
            raise ValueError("Source-to-node mapping contains an unknown source origin.")
        if any(node.id not in self.source_to_nodes.get(node.source_id, []) for node in self.nodes):
            raise ValueError("Source-to-node mapping differs from frozen node origins.")
        if self.provenance.status == "measured":
            if not self.provenance.composition_measured:
                raise ValueError("Measured hierarchy requires measured composition evidence.")
            if not self.provenance.evidence.get("composition_report_sha256"):
                raise ValueError("Measured hierarchy requires a composition report checksum.")
            if any(node.program.provenance.get("status") != "measured" for node in self.nodes):
                raise ValueError("Measured hierarchy cannot contain unmeasured leaves.")
        return self

    @property
    def content_hash(self) -> str:
        semantic = self.model_dump(mode="json", exclude={"provenance"})
        for node in semantic["nodes"]:
            node["program"].pop("provenance", None)
        # H03 added optional typed defaults to references. An absent default
        # must keep the H02 v1 semantic hash, including for already saved files.
        for stage in [*semantic["nodes"], *semantic["exports"]]:
            refs = list(stage["inputs"].values())
            if stage["when"]:
                refs.extend(predicate["ref"] for predicate in stage["when"]["all"])
            for ref in refs:
                if ref.get("default") is None:
                    ref.pop("default", None)
        semantic["nodes"].sort(key=lambda node: node["id"])
        semantic["exports"].sort(key=lambda export: export["id"])
        for instances in semantic["source_to_nodes"].values():
            instances.sort()
        return fingerprint(semantic)

    @property
    def provenance_hash(self) -> str:
        return fingerprint({"graph": self.provenance.model_dump(mode="json"),
                            "leaves": {node.id: node.program.provenance for node in self.nodes}})

    def save(self, path: str | Path) -> None:
        from .hierarchy_validation import validate_hierarchy_artifact

        document = self.model_dump(mode="json")
        self.model_validate(document)  # Recheck nested mutable values before sealing the envelope.
        validate_hierarchy_artifact(self)
        atomic_json(path, {"artifact": document, "sha256": fingerprint(document)})

    @classmethod
    def load(cls, path: str | Path) -> HierarchyArtifact:
        data = load_document(path)
        if set(data) != {"artifact", "sha256"}:
            raise DataError("Expected a hierarchy artifact envelope.")
        if not isinstance(data["artifact"], dict) or fingerprint(data["artifact"]) != data["sha256"]:
            raise DataError("Hierarchy artifact checksum mismatch; recompile or verify the file.")
        artifact = cls.model_validate(data["artifact"])
        from .hierarchy_validation import validate_hierarchy_artifact

        validate_hierarchy_artifact(artifact)
        return artifact


def load_artifact(path: str | Path) -> Program | HierarchyArtifact:
    """Dispatch only supported, integrity-checked frozen artifact versions."""
    data = load_document(path)
    if set(data) == {"program", "sha256"} and isinstance(data["program"], dict):
        if data["program"].get("format") != "systemone-program/v1":
            raise DataError("Unsupported flat artifact format.")
        return Program.load(path)
    if set(data) == {"artifact", "sha256"} and isinstance(data["artifact"], dict):
        if data["artifact"].get("format") != "systemone-hierarchy/v1":
            raise DataError("Unsupported hierarchy artifact format.")
        return HierarchyArtifact.load(path)
    raise DataError("Unsupported artifact envelope.")


def lower_hierarchy(source: HierarchySource) -> HierarchyArtifact:
    """Embed local definitions as bounded, qualified leaf instances."""
    from .hierarchy_validation import validate_hierarchy_source

    plan = validate_hierarchy_source(source)
    nodes: list[LoweredNode] = []
    exports: list[LoweredExport] = []
    mapping: dict[str, list[str]] = {}
    limits = source.limits

    def expand(graph: Graph, prefix: str, stack: tuple[str, ...], inherited: dict[str, Reference]) -> None:
        if len(stack) > limits.max_depth:
            raise DataError("Hierarchy depth limit exceeded.")

        def rebase_ref(ref: Reference) -> Reference:
            if isinstance(ref, RootRef):
                if ref.root in inherited:
                    return inherited[ref.root]
                if prefix:
                    raise DataError(f"Unbound subgraph input: {ref.root}")
                return ref
            return StageRef(stage=f"{prefix}{ref.stage}", decision=ref.decision,
                            field=ref.field, default=ref.default)

        def rebase_when(condition: Condition | None) -> Condition | None:
            if condition is None:
                return None
            return Condition(all=[Predicate(ref=rebase_ref(p.ref), op=p.op, value=p.value)
                                  for p in condition.all])

        stages = {stage.id: stage for stage in graph.stages}
        for stage_id in plan.order[stack[-1] if stack else "root"]:
            stage = stages[stage_id]
            qualified = f"{prefix}{stage.id}"
            if len(nodes) + len(exports) >= limits.max_expanded_nodes:
                raise DataError("Expanded node limit exceeded.")
            inputs = {name: rebase_ref(ref) for name, ref in stage.inputs.items()}
            if isinstance(stage, LeafStage):
                if stage.program.model != source.source.model:
                    raise DataError("Leaf target model differs from root model.")
                source_id = f"{'/'.join(stack) if stack else 'root'}:{stage.id}"
                nodes.append(LoweredNode(
                    id=qualified, source_id=source_id,
                    program=stage.program, inputs=inputs, when=rebase_when(stage.when),
                    after=[f"{prefix}{name}" for name in stage.after], on_review=stage.on_review,
                ))
                mapping.setdefault(source_id, []).append(qualified)
            else:
                if stage.definition in stack:
                    raise DataError("Recursive hierarchy definition.")
                child = source.definitions.get(stage.definition)
                if child is None:
                    raise DataError(f"Missing local definition: {stage.definition}")
                if set(inputs) != set(child.inputs):
                    raise DataError("Subgraph invocation input ports differ from definition.")
                expand(child, f"{qualified}/", (*stack, stage.definition), inputs)
                exports.append(LoweredExport(
                    id=qualified, input_contracts=child.inputs, output_contracts=child.outputs, outputs={
                        name: FinalMapping(candidates=[Candidate(
                            stage=f"{qualified}/{candidate.stage}", decision=candidate.decision,
                            label_map=candidate.label_map, distribution_scope=candidate.distribution_scope,
                        ) for candidate in output.candidates], on_missing=output.on_missing)
                        for name, output in child.final.items()
                    }, inputs=inputs, when=rebase_when(stage.when),
                    after=[f"{prefix}{name}" for name in stage.after], on_review=stage.on_review,
                ))
        if len(nodes) > limits.max_native_calls_per_example:
            raise DataError("Expanded native call limit exceeded.")

    expand(source.graph, "", (), {})
    artifact = HierarchyArtifact(
        format="systemone-hierarchy/v1", source=source.source, limits=limits,
        nodes=nodes, exports=sorted(exports, key=lambda export: export.id),
        final={name: FinalMapping(
            candidates=[Candidate(stage=c.stage, decision=c.decision,
                                  label_map=c.label_map, distribution_scope=c.distribution_scope)
                        for c in mapping_.candidates], on_missing=mapping_.on_missing)
            for name, mapping_ in source.graph.final.items()},
        source_to_nodes={name: sorted(instances) for name, instances in mapping.items()},
    )
    from .hierarchy_validation import validate_hierarchy_artifact

    validate_hierarchy_artifact(artifact)
    return artifact
