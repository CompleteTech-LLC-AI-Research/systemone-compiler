"""Static hierarchy validation and route/dataflow regressions."""
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from s1compiler.errors import DataError
from s1compiler.data import Example
from s1compiler.hierarchy import HierarchySource, load_artifact, lower_hierarchy
from s1compiler.hierarchy_validation import (validate_hierarchy_artifact,
                                             validate_hierarchy_compile_inputs, validate_hierarchy_source)


FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "hierarchy_contract"


def fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["chain", "conditional", "diamond", "nested"])
def test_valid_sources_share_one_static_validator(name):
    source = HierarchySource.model_validate(fixture(name))
    plan = validate_hierarchy_source(source)
    artifact = lower_hierarchy(source)
    assert plan.order["root"]
    assert artifact.nodes


def test_reference_may_name_stage_declared_later_in_source_list():
    data = fixture("chain")
    data["graph"]["stages"].reverse()
    artifact = lower_hierarchy(HierarchySource.model_validate(data))
    assert [node.id for node in artifact.nodes] == ["signal", "priority"]


def test_topological_ties_are_qualified_id_deterministic():
    data = fixture("nested")
    first = lower_hierarchy(HierarchySource.model_validate(data))
    data["graph"]["stages"].reverse()
    second = lower_hierarchy(HierarchySource.model_validate(data))
    assert [node.id for node in first.nodes] == [node.id for node in second.nodes]
    assert first.content_hash == second.content_hash
    assert first.content_hash == "166ce0e62e4479fac9f0f5302be3716dd06706e47444ee251205c5346b67621f"


def _cycle(data):
    data["graph"]["stages"][0]["after"] = ["priority"]


def _undeclared(data):
    data["graph"]["stages"][0]["inputs"]["message"] = {"root": "secret"}


def _bool_number(data):
    data["graph"]["stages"][1]["inputs"]["urgent"] = {
        "stage": "signal", "decision": "urgent", "field": "p_true"}


def _bad_score_scale(data):
    child = data["graph"]["stages"][1]["program"]
    child["decisions"]["priority"]["criteria"] = ["low", "high"]
    child["questions"]["priority"]["criteria"] = ["low", "high"]


@pytest.mark.parametrize("mutation,match", [
    (_cycle, "cycle"),
    (_undeclared, "undeclared root input"),
    (_bool_number, "input type mismatch"),
    (_bad_score_scale, "Score scale length"),
])
def test_chain_rejects_invalid_dataflow_before_lowering(mutation, match):
    data = fixture("chain")
    mutation(data)
    with pytest.raises((DataError, ValidationError), match=match):
        lower_hierarchy(HierarchySource.model_validate(data))


def test_non_dominating_conditional_dependency_rejected():
    data = fixture("conditional")
    technical = data["graph"]["stages"][2]
    technical["inputs"]["message"] = {
        "stage": "billing", "decision": "resolution", "field": "value"}
    with pytest.raises(DataError, match="non-dominating predecessor"):
        lower_hierarchy(HierarchySource.model_validate(data))


def test_consumer_guard_may_narrow_predecessor_guard():
    data = fixture("chain")
    signal, priority = data["graph"]["stages"]
    signal["when"] = {"all": [{"ref": {"root": "message"}, "op": "in", "value": ["urgent", "outage"]}]}
    priority["when"] = {"all": [{"ref": {"root": "message"}, "op": "eq", "value": "urgent"}]}
    lower_hierarchy(HierarchySource.model_validate(data))


def test_bad_choice_label_map_rejected():
    data = fixture("conditional")
    data["graph"]["final"]["resolution"]["candidates"][0]["label_map"]["refund"] = "invented"
    with pytest.raises(DataError, match="invalid branch label map"):
        lower_hierarchy(HierarchySource.model_validate(data))


def test_unknown_choice_route_label_and_score_equality_rejected():
    data = fixture("conditional")
    data["graph"]["stages"][1]["when"]["all"][0]["value"] = "invented"
    with pytest.raises(DataError, match="unknown Choice label"):
        lower_hierarchy(HierarchySource.model_validate(data))
    score = fixture("diamond")
    combined = score["graph"]["stages"][-1]
    combined["when"] = {"all": [{"ref": {"stage": "operational", "decision": "severity",
                                     "field": "value"}, "op": "eq", "value": 1}]}
    with pytest.raises(DataError, match="equality route requires"):
        lower_hierarchy(HierarchySource.model_validate(score))


def test_duplicate_and_overlapping_terminals_rejected():
    data = fixture("conditional")
    mapping = data["graph"]["final"]["resolution"]
    mapping["candidates"].append(copy.deepcopy(mapping["candidates"][0]))
    with pytest.raises(DataError, match="duplicate final terminal"):
        lower_hierarchy(HierarchySource.model_validate(data))
    mapping["candidates"].pop()
    data["graph"]["stages"][2]["when"]["all"][0]["value"] = "billing"
    with pytest.raises(DataError, match="ambiguous overlapping final routes"):
        lower_hierarchy(HierarchySource.model_validate(data))


def test_missing_final_and_recursive_definition_rejected():
    data = fixture("nested")
    del data["graph"]["final"]["second_review"]
    with pytest.raises(ValidationError, match="final mapping"):
        HierarchySource.model_validate(data)
    recursive = fixture("nested")
    recursive["definitions"]["check_note"]["stages"][0] = {
        "id": "again", "kind": "subgraph", "definition": "check_note", "inputs": {"note": {"root": "note"}}}
    with pytest.raises(ValidationError, match="Recursive"):
        HierarchySource.model_validate(recursive)


def test_unreachable_stage_rejected():
    data = fixture("chain")
    extra = copy.deepcopy(data["graph"]["stages"][0])
    extra["id"] = "unused"
    data["graph"]["stages"].append(extra)
    with pytest.raises(DataError, match="unreachable stage"):
        lower_hierarchy(HierarchySource.model_validate(data))


def test_unused_local_definition_is_rejected():
    data = fixture("nested")
    data["definitions"]["unused"] = copy.deepcopy(data["definitions"]["check_note"])
    with pytest.raises(DataError, match="Unused hierarchy definitions: unused"):
        lower_hierarchy(HierarchySource.model_validate(data))


def test_compile_gate_reuses_four_split_isolation_before_provider_work():
    source = HierarchySource.model_validate(fixture("chain"))
    splits = {name: [Example(id=name, state={"message": name}, expected={"priority": 1})]
              for name in ("train", "validation", "calibration", "test")}
    assert validate_hierarchy_compile_inputs(source, splits).order["root"] == ("signal", "priority")
    splits["test"][0].id = "train"
    with pytest.raises(DataError, match="Duplicate example ID"):
        validate_hierarchy_compile_inputs(source, splits)
    del splits["test"]
    with pytest.raises(DataError, match="requires train"):
        validate_hierarchy_compile_inputs(source, splits)


def test_rehashed_frozen_graph_cannot_change_reference_type_or_add_cycle():
    artifact = lower_hierarchy(HierarchySource.model_validate(fixture("chain")))
    data = artifact.model_dump(mode="json")
    data["nodes"][1]["inputs"]["urgent"]["field"] = "p_true"
    from s1compiler.hierarchy import HierarchyArtifact
    with pytest.raises(DataError, match="input type mismatch"):
        validate_hierarchy_artifact(HierarchyArtifact.model_validate(data))
    data = artifact.model_dump(mode="json")
    data["nodes"][0]["after"] = ["priority"]
    with pytest.raises(DataError, match="cycle"):
        validate_hierarchy_artifact(HierarchyArtifact.model_validate(data))


def test_frozen_loader_runs_semantic_validation_after_checksum(tmp_path):
    from s1compiler.io import atomic_json, fingerprint
    artifact = lower_hierarchy(HierarchySource.model_validate(fixture("chain")))
    data = artifact.model_dump(mode="json")
    data["nodes"][1]["inputs"]["urgent"]["field"] = "p_true"
    path = tmp_path / "rehashed-invalid.s1.json"
    atomic_json(path, {"artifact": data, "sha256": fingerprint(data)})
    with pytest.raises(DataError, match="input type mismatch"):
        load_artifact(path)


def test_optional_vendor_confidence_requires_optional_receiving_port():
    data = fixture("chain")
    data["graph"]["stages"][1]["inputs"]["urgent"] = {
        "stage": "signal", "decision": "urgent", "field": "vendor_confidence"}
    with pytest.raises(DataError, match="input type mismatch"):
        lower_hierarchy(HierarchySource.model_validate(data))
    receiving = data["graph"]["stages"][1]["program"]
    receiving["state"]["urgent"]["type"] = "number"
    with pytest.raises(DataError, match="optional value cannot feed required"):
        lower_hierarchy(HierarchySource.model_validate(data))
    receiving["state"]["urgent"]["required"] = False
    lower_hierarchy(HierarchySource.model_validate(data))


def test_skipped_predecessor_needs_optional_port_and_typed_default():
    data = fixture("conditional")
    technical = data["graph"]["stages"][2]
    technical["program"]["state"]["message"]["required"] = False
    technical["inputs"]["message"] = {
        "stage": "billing", "decision": "resolution", "field": "value", "default": "not_billing"}
    lower_hierarchy(HierarchySource.model_validate(data))
    technical["inputs"]["message"]["default"] = 42
    with pytest.raises(DataError, match="invalid typed default"):
        lower_hierarchy(HierarchySource.model_validate(data))


def test_numeric_route_intervals_prove_disjointness_and_detect_overlap():
    data = fixture("nested")
    risk = {"type": "number", "required": True, "description": "Synthetic routing scalar"}
    data["source"]["state"]["risk"] = risk
    data["graph"]["inputs"]["risk"] = risk
    first, second = data["graph"]["stages"]
    first["when"] = {"all": [{"ref": {"root": "risk"}, "op": "lt", "value": 3}]}
    second["when"] = {"all": [{"ref": {"root": "risk"}, "op": "gte", "value": 3}]}
    data["graph"]["final"]["first_review"]["candidates"].append({
        "stage": "second", "decision": "review", "distribution_scope": "full_contract"})
    lower_hierarchy(HierarchySource.model_validate(data))
    second["when"]["all"][0]["value"] = 2
    with pytest.raises(DataError, match="ambiguous overlapping final routes"):
        lower_hierarchy(HierarchySource.model_validate(data))
    second["when"]["all"][0]["value"] = 3
    first["when"]["all"].append({"ref": {"root": "risk"}, "op": "gte", "value": 3})
    with pytest.raises(DataError, match="unreachable contradictory numeric route"):
        lower_hierarchy(HierarchySource.model_validate(data))


# --- graph size, call, and nesting-depth limits (bounded acyclic plans) -------------------------
_CONTRACT = Path(__file__).resolve().parents[1] / "examples" / "hierarchy_contract"


def _contract(name):
    return json.loads((_CONTRACT / name).read_text(encoding="utf-8"))


def _nested_chain(levels, max_depth=None):
    """Root stages call level_1, which calls level_2, and so on; the last level holds the leaf."""
    data = _contract("nested.json")
    leaf = data["definitions"]["check_note"]
    definitions = {}
    for level in range(levels, 0, -1):
        if level == levels:
            definitions[f"level_{level}"] = copy.deepcopy(leaf)
            continue
        definitions[f"level_{level}"] = {
            "inputs": copy.deepcopy(leaf["inputs"]), "outputs": copy.deepcopy(leaf["outputs"]),
            "stages": [{"id": "inner", "kind": "subgraph", "definition": f"level_{level + 1}",
                        "inputs": {"note": {"root": "note"}}}],
            "final": {"review": {"candidates": [{"stage": "inner", "decision": "review",
                                                  "distribution_scope": "full_contract"}],
                                 "on_missing": "review_required"}}}
    data["definitions"] = definitions
    for stage in data["graph"]["stages"]:
        stage["definition"] = "level_1"
    if max_depth is not None:
        data["limits"] = {**data["limits"], "max_depth": max_depth}
    return data


@pytest.mark.parametrize("field,value", [("max_expanded_nodes", 65), ("max_depth", 9),
                                         ("max_native_calls_per_example", 65)])
def test_limits_cannot_be_raised_above_the_hard_ceilings(field, value):
    data = _contract("chain.json")
    data["limits"] = {**data["limits"], field: value}
    with pytest.raises(ValidationError):
        HierarchySource.model_validate(data)


@pytest.mark.parametrize("field,message", [("max_expanded_nodes", "Expanded node limit exceeded"),
                                           ("max_native_calls_per_example", "native call limit exceeded")])
def test_lowering_rejects_a_graph_larger_than_its_declared_limits(field, message):
    data = _contract("chain.json")
    data["limits"] = {**data["limits"], field: 1}  # chain.json expands to two leaf calls
    with pytest.raises(DataError, match=message):
        lower_hierarchy(HierarchySource.model_validate(data))


def test_nested_expansion_counts_toward_the_node_limit():
    data = _contract("nested.json")
    data["limits"] = {**data["limits"], "max_expanded_nodes": 2}
    with pytest.raises(DataError, match="Expanded node limit exceeded"):
        lower_hierarchy(HierarchySource.model_validate(data))


@pytest.mark.parametrize("levels,accepted", [(1, True), (3, True), (8, True), (9, False)])
def test_default_nesting_depth_boundary_is_eight(levels, accepted):
    data = _nested_chain(levels)
    if accepted:
        assert len(lower_hierarchy(HierarchySource.model_validate(data)).nodes) == 2
    else:
        with pytest.raises(ValidationError):
            HierarchySource.model_validate(data)


@pytest.mark.parametrize("max_depth,accepted", [(2, False), (3, True), (4, True)])
def test_declared_depth_limit_is_enforced_exactly(max_depth, accepted):
    data = _nested_chain(3, max_depth)
    if accepted:
        assert len(lower_hierarchy(HierarchySource.model_validate(data)).nodes) == 2
    else:
        with pytest.raises(ValidationError):
            HierarchySource.model_validate(data)
