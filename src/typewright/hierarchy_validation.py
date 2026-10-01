"""Static, provider-free validation of authored hierarchy dataflow."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .data import Example, assert_disjoint
from .errors import DataError
from .hierarchy import (Candidate, Condition, Graph, HierarchyArtifact, HierarchySource, LeafStage,
                        Reference, RootRef, StageRef, lower_hierarchy)
from .hierarchy_data import HierarchySplitGuard
from .models import ID, Decision, StateField, project_state


@dataclass(frozen=True)
class ValidationPlan:
    """Stable stage order for each authoring graph, keyed by definition name."""

    order: dict[str, tuple[str, ...]]
    split_guard: HierarchySplitGuard | None = None


def _fail(scope: str, stage: str, message: str) -> None:
    raise DataError(f"{scope}/{stage}: {message}")


def _output_type(decision: Decision, field: str, scope: str, stage: str) -> tuple[str, bool]:
    if field == "value":
        return {"choice": "string", "noul": "boolean", "score": "number"}[decision.type], False
    if field == "p_true" and decision.type == "noul":
        return "number", False
    if field == "probabilities" and decision.type in {"choice", "score"}:
        return "object", False
    if field == "vendor_confidence":
        return "number", True
    if field == "gate_score":
        return "number", False
    if field == "review_required":
        return "boolean", False
    _fail(scope, stage, f"invalid reference field {field} for {decision.type}")


def _ref_info(ref: Reference, inputs: dict[str, StateField],
              outputs: dict[str, dict[str, Decision]], scope: str, stage: str,
              ) -> tuple[str, bool, Decision | None]:
    if isinstance(ref, RootRef):
        if ref.root not in inputs:
            _fail(scope, stage, f"undeclared root input {ref.root}")
        spec = inputs[ref.root]
        return spec.type, not spec.required, None
    if ref.stage not in outputs:
        _fail(scope, stage, f"unknown predecessor {ref.stage}")
    if ref.decision not in outputs[ref.stage]:
        _fail(scope, stage, f"unknown decision {ref.stage}.{ref.decision}")
    decision = outputs[ref.stage][ref.decision]
    kind, optional = _output_type(decision, ref.field, scope, stage)
    return kind, optional, decision


def _compatible(actual: str, expected: str) -> bool:
    return actual == expected or (actual == "integer" and expected == "number")


def _literal_matches(value: Any, kind: str) -> bool:
    if kind == "boolean":
        return type(value) is bool
    if kind == "string":
        return type(value) is str
    if kind == "integer":
        return type(value) is int
    if kind == "number":
        return type(value) is int or (type(value) is float and math.isfinite(value))
    return False


def _condition(condition: Condition | None, inputs: dict[str, StateField],
               outputs: dict[str, dict[str, Decision]], scope: str, stage: str) -> None:
    if condition is None:
        return
    for predicate in condition.all:
        ref = predicate.ref
        if ref.default is not None:
            _fail(scope, stage, "condition references cannot carry defaults")
        actual, optional, decision = _ref_info(ref, inputs, outputs, scope, stage)
        if optional:
            _fail(scope, stage, "condition cannot read an optional value")
        if isinstance(ref, StageRef):
            if predicate.op in {"eq", "in"} and not (
                ref.field == "value" and decision and decision.type in {"choice", "noul"}
            ):
                _fail(scope, stage, "equality route requires a typed Choice or Noul value")
            if predicate.op in {"lt", "lte", "gt", "gte"} and not (
                ref.field == "value" and decision and decision.type == "score"
            ):
                _fail(scope, stage, "numeric route requires a Score value")
        if predicate.op in {"lt", "lte", "gt", "gte"}:
            if actual not in {"number", "integer"} or not _literal_matches(predicate.value, actual):
                _fail(scope, stage, "numeric condition type mismatch")
        elif predicate.op == "in":
            if not isinstance(predicate.value, list) or not predicate.value or not all(
                _literal_matches(item, actual) for item in predicate.value
            ):
                _fail(scope, stage, "membership condition type mismatch")
        elif not _literal_matches(predicate.value, actual):
            _fail(scope, stage, "equality condition type mismatch")
        if decision is not None and decision.type == "choice":
            values = predicate.value if predicate.op == "in" else [predicate.value]
            if any(value not in decision.criteria for value in values):
                _fail(scope, stage, "unknown Choice label in condition")
        if (decision is not None and decision.type == "score" and
                predicate.op in {"lt", "lte", "gt", "gte"} and
                not 0 <= predicate.value <= len(decision.criteria) - 1):
            # A Score value lies on 0..levels-1, so a constant outside it is always or never true.
            _fail(scope, stage, "Score route constant is outside the declared scale")
    for predicate in condition.all:
        if predicate.op in {"eq", "in"} and not _atom(condition, predicate.ref):
            _fail(scope, stage, "unreachable contradictory route condition")
        if predicate.op in {"lt", "lte", "gt", "gte"}:
            bounds = _bounds(condition, predicate.ref)
            if bounds is not None and _empty_bounds(bounds):
                _fail(scope, stage, "unreachable contradictory numeric route")


def _atom(condition: Condition | None, ref: StageRef | RootRef) -> set[Any] | None:
    """Finite equality domain guaranteed by a conjunction, if one is present."""
    if condition is None:
        return None
    domain = None
    for predicate in condition.all:
        if predicate.ref != ref or predicate.op not in {"eq", "in"}:
            continue
        values = predicate.value if predicate.op == "in" else [predicate.value]
        if any(isinstance(value, (list, dict)) for value in values):
            return None
        current = set(values)
        domain = current if domain is None else domain & current
    return domain


def _bounds(condition: Condition | None, ref: StageRef | RootRef,
            ) -> tuple[float | int | None, bool, float | int | None, bool] | None:
    if condition is None:
        return None
    lower, lower_inc, upper, upper_inc = None, False, None, False
    found = False
    for predicate in condition.all:
        if predicate.ref != ref or predicate.op not in {"lt", "lte", "gt", "gte"}:
            continue
        found = True
        value = predicate.value
        if predicate.op in {"gt", "gte"}:
            inclusive = predicate.op == "gte"
            if lower is None or value > lower or (value == lower and not inclusive):
                lower, lower_inc = value, inclusive
        else:
            inclusive = predicate.op == "lte"
            if upper is None or value < upper or (value == upper and not inclusive):
                upper, upper_inc = value, inclusive
    return (lower, lower_inc, upper, upper_inc) if found else None


def _empty_bounds(bounds: tuple[float | int | None, bool, float | int | None, bool]) -> bool:
    lower, lower_inc, upper, upper_inc = bounds
    return lower is not None and upper is not None and (lower > upper or (
        lower == upper and not (lower_inc and upper_inc)))


def _in_bounds(value: float | int, bounds: tuple[float | int | None, bool, float | int | None, bool]) -> bool:
    lower, lower_inc, upper, upper_inc = bounds
    return (lower is None or value > lower or (value == lower and lower_inc)) and (
        upper is None or value < upper or (value == upper and upper_inc))


def _bounds_subset(left: tuple[float | int | None, bool, float | int | None, bool],
                   right: tuple[float | int | None, bool, float | int | None, bool]) -> bool:
    ll, lli, lu, lui = left
    rl, rli, ru, rui = right
    lower_ok = rl is None or (ll is not None and (ll > rl or (ll == rl and (not lli or rli))))
    upper_ok = ru is None or (lu is not None and (lu < ru or (lu == ru and (not lui or rui))))
    return lower_ok and upper_ok


def _implies(consumer: Condition | None, producer: Condition | None) -> bool:
    if producer is None:
        return True
    if consumer is None:
        return False
    for required in producer.all:
        if required in consumer.all:
            continue
        if required.op not in {"eq", "in"}:
            if required.op not in {"lt", "lte", "gt", "gte"}:
                return False
            actual_bounds = _bounds(consumer, required.ref)
            required_bounds = _bounds(Condition(all=[required]), required.ref)
            finite = _atom(consumer, required.ref)
            if required_bounds is None or not (
                (actual_bounds is not None and _bounds_subset(actual_bounds, required_bounds))
                or (finite is not None and all(_in_bounds(value, required_bounds) for value in finite))
            ):
                return False
            continue
        observed = _atom(consumer, required.ref)
        expected = set(required.value if required.op == "in" else [required.value])
        if observed is None or not observed <= expected:
            return False
    return True


def _disjoint(left: Condition | None, right: Condition | None) -> bool:
    if left is None or right is None:
        return False
    for atom in left.all:
        if atom.op not in {"eq", "in"}:
            continue
        first = _atom(left, atom.ref)
        second = _atom(right, atom.ref)
        if first is not None and second is not None and first.isdisjoint(second):
            return True
        right_bounds = _bounds(right, atom.ref)
        if first is not None and right_bounds is not None and all(not _in_bounds(value, right_bounds)
                                                                 for value in first):
            return True
    for atom in right.all:
        if atom.op in {"eq", "in"}:
            second = _atom(right, atom.ref)
            left_bounds = _bounds(left, atom.ref)
            if second is not None and left_bounds is not None and all(not _in_bounds(value, left_bounds)
                                                                      for value in second):
                return True
    for atom in left.all:
        if atom.op not in {"lt", "lte", "gt", "gte"}:
            continue
        first_bounds = _bounds(left, atom.ref)
        second_bounds = _bounds(right, atom.ref)
        if first_bounds is None or second_bounds is None:
            continue
        first_lower, first_lower_inc, first_upper, first_upper_inc = first_bounds
        second_lower, second_lower_inc, second_upper, second_upper_inc = second_bounds
        if first_upper is not None and second_lower is not None and (
            first_upper < second_lower or (first_upper == second_lower and not (
                first_upper_inc and second_lower_inc))
        ):
            return True
        if second_upper is not None and first_lower is not None and (
            second_upper < first_lower or (second_upper == first_lower and not (
                second_upper_inc and first_lower_inc))
        ):
            return True
    return False


def _validate_final(candidate: Candidate, target: Decision,
                    outputs: dict[str, dict[str, Decision]], scope: str, output: str) -> None:
    if candidate.stage not in outputs or candidate.decision not in outputs[candidate.stage]:
        _fail(scope, output, "final mapping references an unknown stage output")
    child = outputs[candidate.stage][candidate.decision]
    if child.type != target.type:
        _fail(scope, output, "final output type mismatch")
    if target.type == "choice":
        child_labels, root_labels = set(child.criteria), set(target.criteria)
        if candidate.distribution_scope == "branch_conditional":
            labels = candidate.label_map or {}
            if set(labels) != child_labels or not set(labels.values()) <= root_labels or len(set(labels.values())) != len(labels):
                _fail(scope, output, "invalid branch label map")
        elif child_labels != root_labels or candidate.label_map is not None:
            _fail(scope, output, "full Choice labels must equal the public contract")
    elif candidate.label_map is not None or candidate.distribution_scope != "full_contract":
        _fail(scope, output, "non-Choice output cannot have branch label mapping")
    if target.type == "score" and len(child.criteria) != len(target.criteria):
        _fail(scope, output, "Score scale length differs from public contract")


def _validate_graph(graph: Graph, source: HierarchySource, scope: str) -> tuple[str, ...]:
    stages = {stage.id: stage for stage in graph.stages}
    outputs = {name: (stage.program.decisions if isinstance(stage, LeafStage)
                      else source.definitions[stage.definition].outputs)
               for name, stage in stages.items()}
    dependencies: dict[str, set[str]] = {}
    for name, stage in stages.items():
        if isinstance(stage, LeafStage):
            if stage.program.model != source.source.model:
                _fail(scope, name, "target model drift")
            ports = stage.program.state
        else:
            ports = source.definitions[stage.definition].inputs
        if not set(stage.inputs) <= set(ports) or any(
            field.required and port not in stage.inputs for port, field in ports.items()
        ):
            _fail(scope, name, "stage input contract mismatch")
        deps = set(stage.after)
        if name in deps:
            _fail(scope, name, "self-cycle in after edge")
        _condition(stage.when, graph.inputs, outputs, scope, name)
        for port, ref in stage.inputs.items():
            actual, optional, _ = _ref_info(ref, graph.inputs, outputs, scope, name)
            if not _compatible(actual, ports[port].type):
                _fail(scope, name, f"input type mismatch on {port}: {actual} to {ports[port].type}")
            if optional and ports[port].required and ref.default is None:
                _fail(scope, name, f"optional value cannot feed required input {port}")
            if ref.default is not None:
                if ports[port].required:
                    _fail(scope, name, f"default requires optional input {port}")
                try:
                    project_state({port: ports[port]}, {port: ref.default})
                except DataError as exc:
                    _fail(scope, name, f"invalid typed default for {port}: {exc}")
            if isinstance(ref, StageRef):
                deps.add(ref.stage)
                producer = stages[ref.stage]
                if ports[port].required and not _implies(stage.when, producer.when):
                    _fail(scope, name, f"non-dominating predecessor {ref.stage} for {port}")
        if stage.when:
            for predicate in stage.when.all:
                if isinstance(predicate.ref, StageRef):
                    deps.add(predicate.ref.stage)
                    producer = stages[predicate.ref.stage]
                    if not _implies(stage.when, producer.when):
                        _fail(scope, name, f"condition reads non-dominating predecessor {predicate.ref.stage}")
        if not deps <= set(stages):
            _fail(scope, name, "unknown explicit after predecessor")
        dependencies[name] = deps
    order: list[str] = []
    pending = set(stages)
    while pending:
        ready = sorted(name for name in pending if dependencies[name] <= set(order))
        if not ready:
            _fail(scope, ",".join(sorted(pending)), "cycle in stage dependencies")
        selected = ready[0]
        order.append(selected)
        pending.remove(selected)
    for output, mapping in graph.final.items():
        target = graph.outputs[output]
        seen = set()
        for candidate in mapping.candidates:
            identity = (candidate.stage, candidate.decision)
            if identity in seen:
                _fail(scope, output, "duplicate final terminal")
            seen.add(identity)
            _validate_final(candidate, target, outputs, scope, output)
        for index, left in enumerate(mapping.candidates):
            for right in mapping.candidates[index + 1:]:
                if not _disjoint(stages[left.stage].when, stages[right.stage].when):
                    _fail(scope, output, "ambiguous overlapping final routes")
    useful = {candidate.stage for mapping in graph.final.values() for candidate in mapping.candidates}
    pending = list(useful)
    while pending:
        current = pending.pop()
        for dep in dependencies[current] - useful:
            useful.add(dep)
            pending.append(dep)
    if useful != set(stages):
        _fail(scope, ",".join(sorted(set(stages) - useful)), "unreachable stage cannot affect a final output")
    return tuple(order)


def validate_hierarchy_source(source: HierarchySource) -> ValidationPlan:
    """Apply identical source checks to authored and teacher-proposed graphs."""
    order = {"root": _validate_graph(source.graph, source, "root")}
    for name, graph in source.definitions.items():
        order[name] = _validate_graph(graph, source, name)
    used_definitions: set[str] = set()

    def visit(graph: Graph) -> None:
        for stage in graph.stages:
            if not isinstance(stage, LeafStage) and stage.definition not in used_definitions:
                used_definitions.add(stage.definition)
                visit(source.definitions[stage.definition])

    visit(source.graph)
    if used_definitions != set(source.definitions):
        raise DataError("Unused hierarchy definitions: " + ", ".join(sorted(set(source.definitions) - used_definitions)))
    return ValidationPlan(order=order)


def validate_hierarchy_compile_inputs(source: HierarchySource,
                                      splits: dict[str, list[Example]]) -> ValidationPlan:
    """Gate every compile path before constructing a paid backend or teacher."""
    if set(splits) != {"train", "validation", "calibration", "test"}:
        raise DataError("Hierarchy compile requires train, validation, calibration, and test splits.")
    plan = validate_hierarchy_source(source)
    assert_disjoint(splits, source.source)
    return ValidationPlan(order=plan.order, split_guard=HierarchySplitGuard(lower_hierarchy(source), splits))


def _scope(name: str) -> str:
    """Export ID that lexically contains a qualified stage; empty for the root graph."""
    return name.rpartition("/")[0]


def _validate_scopes(artifact: HierarchyArtifact, stages: dict[str, Any], export_ids: set[str]) -> None:
    """Tie each stage to its export chain, as lowering builds it, and keep edges lexical.

    Lowering rebases a stage's references into its own scope; the only way a reference leaves
    that scope is through the inputs its enclosing export passed down. Finals and `after` edges
    never leave their scope. Node origins name the definition path of that same export chain.
    """
    definitions: dict[str, str] = {}
    for node in artifact.nodes:
        parts = node.id.split("/")
        origin, _, local = node.source_id.partition(":")
        path = [] if origin == "root" else origin.split("/")
        if local != parts[-1] or len(path) != len(parts) - 1 or "root" in path:
            _fail("artifact", node.id, "node origin differs from its export chain")
        for depth, definition in enumerate(path):
            if definitions.setdefault("/".join(parts[:depth + 1]), definition) != definition:
                _fail("artifact", node.id, "node origin differs from its export chain")
    inherited = {export.id: list(export.inputs.values()) for export in artifact.exports}
    for name, stage in stages.items():
        scope = _scope(name)
        if name in export_ids and not any(node.id.startswith(f"{name}/") for node in artifact.nodes):
            _fail("artifact", name, "subgraph export has no descendant node")
        refs = list(stage.inputs.values())
        if stage.when:
            refs.extend(predicate.ref for predicate in stage.when.all)
        for ref in refs:
            if isinstance(ref, StageRef) and _scope(ref.stage) == scope:
                continue
            if not (ref in inherited[scope] if scope else isinstance(ref, RootRef)):
                _fail("artifact", name, "reference crosses its subgraph scope")
        if any(_scope(after) != scope for after in stage.after):
            _fail("artifact", name, "after edge crosses its subgraph scope")
    for export in artifact.exports:
        for mapping in export.outputs.values():
            if any(_scope(candidate.stage) != export.id for candidate in mapping.candidates):
                _fail("artifact", export.id, "final candidate crosses its subgraph scope")
    for mapping in artifact.final.values():
        if any(_scope(candidate.stage) for candidate in mapping.candidates):
            _fail("artifact", "final", "final candidate crosses its subgraph scope")


def validate_hierarchy_artifact(artifact: HierarchyArtifact) -> tuple[str, ...]:
    """Recheck a frozen graph's typed transitive edges before any model call."""
    stages = {stage.id: stage for stage in [*artifact.nodes, *artifact.exports]}
    export_ids = {export.id for export in artifact.exports}
    for name in stages:
        parts = name.split("/")
        if not all(ID.fullmatch(part) for part in parts):
            _fail("artifact", name, "invalid qualified ID segment")
        depth = len(parts) - 1 + (name in export_ids)
        if depth > artifact.limits.max_depth:
            _fail("artifact", name, "frozen nesting depth limit exceeded")
        for length in range(1, len(parts)):
            if "/".join(parts[:length]) not in export_ids:
                _fail("artifact", name, "qualified stage has no parent subgraph export")
    _validate_scopes(artifact, stages, export_ids)
    outputs = {node.id: node.program.decisions for node in artifact.nodes}
    outputs.update({export.id: export.output_contracts for export in artifact.exports})
    dependencies: dict[str, set[str]] = {}
    for name, stage in stages.items():
        ports = stage.program.state if hasattr(stage, "program") else stage.input_contracts
        deps = set(stage.after)
        _condition(stage.when, artifact.source.state, outputs, "artifact", name)
        for port, ref in stage.inputs.items():
            actual, optional, _ = _ref_info(ref, artifact.source.state, outputs, "artifact", name)
            if not _compatible(actual, ports[port].type):
                _fail("artifact", name, f"input type mismatch on {port}")
            if optional and ports[port].required and ref.default is None:
                _fail("artifact", name, f"optional value cannot feed required input {port}")
            if ref.default is not None:
                if ports[port].required:
                    _fail("artifact", name, f"default requires optional input {port}")
                project_state({port: ports[port]}, {port: ref.default})
            if isinstance(ref, StageRef):
                deps.add(ref.stage)
                producer = stages[ref.stage]
                if ports[port].required and not _implies(stage.when, producer.when):
                    _fail("artifact", name, f"non-dominating predecessor {ref.stage} for {port}")
        if stage.when:
            for predicate in stage.when.all:
                if isinstance(predicate.ref, StageRef):
                    deps.add(predicate.ref.stage)
                    producer = stages[predicate.ref.stage]
                    if not _implies(stage.when, producer.when):
                        _fail("artifact", name, f"condition reads non-dominating predecessor {predicate.ref.stage}")
        if name in deps:
            _fail("artifact", name, "self-cycle")
        if not deps <= set(stages):
            _fail("artifact", name, "unknown predecessor")
        dependencies[name] = deps
    for export in artifact.exports:
        for output, mapping in export.outputs.items():
            target = export.output_contracts[output]
            for candidate in mapping.candidates:
                _validate_final(candidate, target, outputs, "artifact", export.id)
                dependencies[export.id].add(candidate.stage)
            for index, left in enumerate(mapping.candidates):
                for right in mapping.candidates[index + 1:]:
                    if not _disjoint(stages[left.stage].when, stages[right.stage].when):
                        _fail("artifact", export.id, "ambiguous overlapping final routes")
    for output, mapping in artifact.final.items():
        target = artifact.source.decisions[output]
        for candidate in mapping.candidates:
            _validate_final(candidate, target, outputs, "artifact", output)
        for index, left in enumerate(mapping.candidates):
            for right in mapping.candidates[index + 1:]:
                if not _disjoint(stages[left.stage].when, stages[right.stage].when):
                    _fail("artifact", output, "ambiguous overlapping final routes")
    order: list[str] = []
    pending = set(stages)
    while pending:
        ready = sorted(name for name in pending if dependencies[name] <= set(order))
        if not ready:
            _fail("artifact", ",".join(sorted(pending)), "cycle in frozen dependencies")
        order.append(ready[0])
        pending.remove(ready[0])
    useful = {candidate.stage for mapping in artifact.final.values() for candidate in mapping.candidates}
    pending_useful = list(useful)
    while pending_useful:
        current = pending_useful.pop()
        for dependency in dependencies[current] - useful:
            useful.add(dependency)
            pending_useful.append(dependency)
    if useful != set(stages):
        _fail("artifact", ",".join(sorted(set(stages) - useful)), "unreachable frozen stage")
    return tuple(order)
