"""Hierarchy split, derived-stage, annotation, and teacher-isolation gates."""
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from typewright.backends import ManagedBackend, MockBackend
from typewright.data import Example
from typewright.errors import DataError
from typewright.hierarchy import HierarchySource, RootRef, StageRef, lower_hierarchy
from typewright.hierarchy_data import (HierarchyExample, HierarchySplitGuard,
                                       HierarchyTeacherInputs, IntermediateAnnotation,
                                       read_hierarchy_jsonl)
from typewright.hierarchy_runtime import HierarchyRuntime
from typewright.hierarchy_validation import validate_hierarchy_compile_inputs


FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "hierarchy_contract"
NAMES = ("train", "validation", "calibration", "test")


def artifact(name="chain"):
    return lower_hierarchy(HierarchySource.load(FIXTURES / f"{name}.json"))


def splits():
    return {name: [Example(id=name, group=name, state={"message": f"{name} unique message"},
                           expected={"priority": 1})] for name in NAMES}


def test_preflight_versions_and_bound_stage_lineage_are_root_scoped():
    frozen = artifact()
    rows = splits()
    guard = HierarchySplitGuard(frozen, rows)
    assert set(guard.versions["datasets"]) == set(NAMES)
    assert guard.versions["train_projections"]["signal"]
    assert guard.versions["observed_train_projections"]["complete"] is False
    backend = ManagedBackend(MockBackend())
    result = HierarchyRuntime(frozen, backend).run(
        rows["train"][0].state, lineage=guard.bind(frozen, "train", rows["train"][0]))
    assert result["status"] == "completed"
    assert result["lineage"]["root_id"] == "train"
    assert result["lineage"]["group"] == "train"
    assert result["lineage"]["split"] == "train"
    assert result["lineage"]["stage_inputs"]["signal"]["origin"] == "root_projection"
    assert result["lineage"]["stage_inputs"]["priority"]["origin"] == "derived_or_routed"
    assert result["lineage"]["stage_output_kind"] == "model_predictions_not_labels"
    assert set(result["lineage"]["stage_predictions"]) == set(result["executed"])
    assert guard.versions["observed_train_projections"]["complete"] is True
    assert guard.versions["observed_train_projections"]["stages"]["priority"]["n"] == 1
    backend.close()


def test_static_stage_projection_duplicates_fail_before_any_provider_call():
    frozen = artifact("nested")
    rows = {name: [Example(id=name, group=name,
                           state={"first_note": "same across splits", "second_note": name},
                           expected={"first_review": True, "second_review": False})]
            for name in NAMES}
    # The full root states differ, but the first nested leaf sees identical input.
    with pytest.raises(DataError, match="Duplicate static stage projection for first/check"):
        HierarchySplitGuard(frozen, rows)


def test_equal_mixed_stage_projections_remain_bound_to_distinct_split_roots():
    source = HierarchySource.load(FIXTURES / "chain.json")
    field = source.source.state["message"].model_copy(deep=True)
    source.source.state["tag"] = field
    source.graph.inputs["tag"] = field.model_copy(deep=True)
    child = source.graph.stages[1]
    child.program.state["tag"] = field.model_copy(deep=True)
    child.inputs["tag"] = RootRef(root="tag")
    child.inputs["message"] = RootRef(root="tag")
    frozen = lower_hierarchy(source)
    rows = splits()
    for batch in rows.values():
        batch[0].state["tag"] = "same root-derived combination"
    guard = HierarchySplitGuard(frozen, rows)
    assert all((name, "priority") not in guard.static_inputs for name in NAMES)
    backend = ManagedBackend(MockBackend())
    results = [HierarchyRuntime(frozen, backend).run(
        rows[name][0].state, lineage=guard.bind(frozen, name, rows[name][0])) for name in NAMES]
    assert all(result["status"] == "completed" for result in results)
    assert all(result["lineage"]["stage_inputs"]["priority"]["origin"] == "derived_or_routed"
               for result in results)
    assert len({result["lineage"]["stage_inputs"]["priority"]["input_sha256"]
                for result in results}) == 1
    assert [result["lineage"]["split"] for result in results] == list(NAMES)
    assert [result["lineage"]["root_id"] for result in results] == list(NAMES)
    teacher = HierarchyTeacherInputs(guard)
    assert len(teacher.traces(results[:1])) == 1
    with pytest.raises(DataError, match="train"):
        teacher.traces(results[1:])
    backend.close()


def test_empty_input_leaf_is_rejected_before_split_projection():
    data = json.loads((FIXTURES / "chain.json").read_text(encoding="utf-8"))
    data["graph"]["stages"][0]["program"]["state"] = {}
    data["graph"]["stages"][0]["inputs"] = {}
    with pytest.raises(ValidationError, match="program needs state"):
        HierarchySource.model_validate(data)


@pytest.mark.parametrize("fault", ["id", "projected_input", "group"])
def test_original_duplicate_and_group_checks_remain_fatal(fault):
    frozen = artifact()
    rows = splits()
    if fault == "id":
        rows["test"][0].id = "train"
    elif fault == "projected_input":
        rows["test"][0].state = copy.deepcopy(rows["train"][0].state)
    else:
        rows["test"][0].group = "train"
    with pytest.raises(DataError):
        HierarchySplitGuard(frozen, rows)


def test_candidate_change_requires_new_projection_preflight():
    frozen = artifact()
    rows = splits()
    guard = HierarchySplitGuard(frozen, rows)
    changed = frozen.model_copy(deep=True)
    changed.nodes[0].program.questions["urgent"].instructions = "changed candidate wording"
    with pytest.raises(DataError, match="rerun split and projection validation"):
        guard.bind(changed, "train", rows["train"][0])
    bound = guard.bind(frozen, "train", rows["train"][0])
    with pytest.raises(DataError, match="root state"):
        HierarchyRuntime(frozen, ManagedBackend(MockBackend())).run(
            {"message": "altered after preflight"}, lineage=bound)


def test_changed_derived_value_fails_before_child_dispatch(monkeypatch):
    frozen = artifact()
    rows = splits()
    guard = HierarchySplitGuard(frozen, rows)

    class Recording(MockBackend):
        def __init__(self):
            self.calls = []

        def evaluate(self, program, state):
            self.calls.append(program.name)
            return super().evaluate(program, state)

    recorder = Recording()
    original = HierarchyRuntime._read

    def corrupt(self, ref, root, values):
        value = original(self, ref, root, values)
        return not value if isinstance(ref, StageRef) and ref.field == "value" else value

    monkeypatch.setattr(HierarchyRuntime, "_read", corrupt)
    result = HierarchyRuntime(frozen, ManagedBackend(recorder)).run(
        rows["train"][0].state, lineage=guard.bind(frozen, "train", rows["train"][0]))
    assert result["status"] == "failed"
    assert result["error"]["type"] == "DataError"
    assert recorder.calls == ["signal"]
    with pytest.raises(DataError, match="not teacher quality feedback"):
        HierarchyTeacherInputs(guard).traces([result])


def test_human_stage_annotations_are_distinct_from_generated_predictions():
    frozen = artifact()
    rows = splits()
    rows["train"][0] = HierarchyExample(
        **rows["train"][0].model_dump(mode="json"),
        annotations=[IntermediateAnnotation(stage="signal", decision="urgent", value=True,
                                            source="human", annotator="synthetic-reviewer")])
    guard = HierarchySplitGuard(frozen, rows)
    result = HierarchyRuntime(frozen, ManagedBackend(MockBackend())).run(
        rows["train"][0].state, lineage=guard.bind(frozen, "train", rows["train"][0]))
    payload = HierarchyTeacherInputs(guard).traces([result])[0]
    assert payload["human_intermediate_labels"][0]["value"] is True
    assert payload["human_intermediate_labels"][0]["source"] == "human"
    assert payload["stage_predictions"]["signal"] == result["lineage"]["stage_predictions"]["signal"]
    assert payload["train_stage_states"]["signal"]["message"] == rows["train"][0].state["message"]
    assert "train_stage_states" not in result["lineage"]
    with pytest.raises(ValidationError):
        IntermediateAnnotation(stage="signal", decision="urgent", value=True,
                               source="model", annotator="invalid")
    rows["train"][0].annotations[0].value = "not a bool"
    with pytest.raises(DataError, match="human annotation"):
        HierarchySplitGuard(frozen, rows)


def test_hierarchy_jsonl_loads_only_explicit_sourced_stage_labels(tmp_path):
    frozen = artifact()
    row = HierarchyExample(id="train-1", state={"message": "urgent outage"},
                           expected={"priority": 1}, annotations=[
        IntermediateAnnotation(stage="signal", decision="urgent", value=True,
                               source="human", annotator="synthetic-reviewer")])
    path = tmp_path / "train.jsonl"
    path.write_text(row.model_dump_json() + "\n", encoding="utf-8")
    assert read_hierarchy_jsonl(path, frozen) == [row]
    path.write_text(row.model_dump_json().replace('"source":"human"', '"source":"model"') + "\n",
                    encoding="utf-8")
    with pytest.raises(DataError, match="Invalid hierarchy record"):
        read_hierarchy_jsonl(path, frozen)


@pytest.mark.parametrize("split", ["validation", "calibration", "test"])
def test_nontrain_sentinels_never_reach_proposal_or_reflection(split):
    frozen = artifact()
    rows = splits()
    rows[split][0].state["message"] = f"{split}-sentinel-never-share"
    guard = HierarchySplitGuard(frozen, rows)
    backend = ManagedBackend(MockBackend())
    result = HierarchyRuntime(frozen, backend).run(
        rows[split][0].state, lineage=guard.bind(frozen, split, rows[split][0]))
    teacher = HierarchyTeacherInputs(guard)
    with pytest.raises(DataError, match="train split"):
        teacher.examples(rows[split])
    with pytest.raises(DataError, match="train results"):
        teacher.traces([result])
    with pytest.raises(DataError, match="train results"):
        teacher.reflection([result], ["signal/urgent/instructions"])
    assert f"{split}-sentinel-never-share" not in json.dumps(teacher.examples(rows["train"]))
    backend.close()


def test_nested_stage_exception_cannot_become_teacher_feedback():
    frozen = artifact("nested")
    rows = {name: [Example(id=name, group=name,
                           state={"first_note": f"{name} first", "second_note": f"{name} second"},
                           expected={"first_review": True, "second_review": False})]
            for name in NAMES}
    guard = HierarchySplitGuard(frozen, rows)

    class Failing(MockBackend):
        def evaluate(self, program, state):
            raise RuntimeError("test-only nested-stage failure sentinel")

    bound = guard.bind(frozen, "validation", rows["validation"][0])
    with pytest.raises(RuntimeError, match="nested-stage failure"):
        HierarchyRuntime(frozen, ManagedBackend(Failing())).run(rows["validation"][0].state,
                                                             lineage=bound)
    with pytest.raises(DataError, match="train results"):
        HierarchyTeacherInputs(guard).traces([{
            "lineage": {"root_id": "validation", "split": "validation"},
            "error": "test-only nested-stage failure sentinel"}])


def test_compile_input_gate_exposes_candidate_specific_split_guard():
    source = HierarchySource.load(FIXTURES / "chain.json")
    plan = validate_hierarchy_compile_inputs(source, splits())
    assert plan.order["root"] == ("signal", "priority")
    assert plan.split_guard.graph_sha256 == artifact().content_hash


def test_coarse_model_derived_inputs_retain_root_groups_and_dedup_only_train():
    data = json.loads((FIXTURES / "chain.json").read_text(encoding="utf-8"))
    child = data["graph"]["stages"][1]
    child["program"]["state"].pop("message")
    child["inputs"].pop("message")
    frozen = lower_hierarchy(HierarchySource.model_validate(data))
    rows = splits()
    rows["train"] = [
        Example(id=f"train-{index}", group=group,
                state={"message": f"urgent outage variant {index}"}, expected={"priority": 1})
        for index, group in ((1, "shared"), (2, "shared"), (3, "separate"))]
    guard = HierarchySplitGuard(frozen, rows)
    backend = ManagedBackend(MockBackend())
    results = [HierarchyRuntime(frozen, backend).run(row.state,
               lineage=guard.bind(frozen, "train", row)) for row in rows["train"]]
    assert all(result["status"] == "completed" for result in results)
    assert len({result["lineage"]["stage_inputs"]["priority"]["input_sha256"]
                for result in results}) == 1
    selected = HierarchyTeacherInputs(guard).routed_train_subset("priority", results)
    assert [item["root_id"] for item in selected] == ["train-1", "train-3"]
    validation = HierarchyRuntime(frozen, backend).run(rows["validation"][0].state,
        lineage=guard.bind(frozen, "validation", rows["validation"][0]))
    with pytest.raises(DataError, match="train results"):
        HierarchyTeacherInputs(guard).routed_train_subset("priority", results + [validation])
    backend.close()


def test_durable_resume_pins_root_identity_even_for_same_projected_input(tmp_path):
    frozen = artifact()
    rows = splits()
    guard = HierarchySplitGuard(frozen, rows)
    directory = tmp_path / "lineage-evidence"
    cancelled = HierarchyRuntime(frozen, ManagedBackend(MockBackend())).run(
        rows["train"][0].state, lineage=guard.bind(frozen, "train", rows["train"][0]),
        evidence_dir=directory, cancel_requested=lambda: True)
    assert cancelled["status"] == "cancelled"
    different = splits()
    different["train"][0].id = "different-root"
    different_guard = HierarchySplitGuard(frozen, different)
    with pytest.raises(DataError, match="plan, data"):
        HierarchyRuntime(frozen, ManagedBackend(MockBackend())).run(
            different["train"][0].state,
            lineage=different_guard.bind(frozen, "train", different["train"][0]),
            evidence_dir=directory, evidence_mode="resume")
    resumed = HierarchyRuntime(frozen, ManagedBackend(MockBackend())).run(
        rows["train"][0].state, lineage=guard.bind(frozen, "train", rows["train"][0]),
        evidence_dir=directory, evidence_mode="resume")
    assert resumed["status"] == "completed"
    replay = HierarchyRuntime(frozen, ManagedBackend(MockBackend())).run(
        rows["train"][0].state, lineage=guard.bind(frozen, "train", rows["train"][0]),
        evidence_dir=directory, evidence_mode="replay")
    assert replay == resumed
