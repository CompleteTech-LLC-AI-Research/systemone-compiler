import json
import math
import pytest
from pydantic import ValidationError
from s1compiler.data import assert_disjoint, read_jsonl, validate_example
from s1compiler.errors import CandidateError, DataError
from s1compiler.io import atomic_json, load_document
from s1compiler.models import Decision, Program, UseCase, project_state
from s1compiler.architect import plan_to_program, program_plan


@pytest.mark.parametrize("kind,criteria", [
    ("choice", {}), ("choice", {"a": "A"}), ("score", ["only one"]),
    ("score", ["x"] * 11), ("noul", {"yes": "Yes"}),
])
def test_invalid_criteria(kind, criteria):
    with pytest.raises(ValidationError):
        Decision(type=kind, goal="Decide a narrow property", criteria=criteria)


@pytest.mark.parametrize("kind,criteria", [
    ("choice", {"a": "Alpha", "b": {"definition": "Beta"}}),
    ("score", ["low", {"description": "high"}]),
    ("noul", None), ("noul", {"true": "Visible", "false": "Absent"}),
])
def test_valid_criteria(kind, criteria):
    Decision(type=kind, goal="Decide a narrow property", criteria=criteria)


def test_projection_strips_extras(source):
    value = project_state(source.state, {"message": "hello", "customer_plan": "team", "secret": "do-not-send"})
    assert set(value) == {"message", "customer_plan"}


def test_missing_required(source):
    with pytest.raises(DataError):
        project_state(source.state, {"message": "hello"})


def test_strict_input_type(source):
    with pytest.raises(DataError):
        project_state(source.state, {"message": 2, "customer_plan": "team"})


def test_optional_input():
    source = UseCase(name="optional", state={"text": {"type": "string", "required": False}},
                     decisions={"flag": {"type": "noul", "goal": "Is evidence present?"}})
    assert project_state(source.state, {}) == {}


def test_rejects_nonfinite_nested_state():
    source = UseCase(name="nested", state={"record": {"type": "object"}},
                     decisions={"flag": {"type": "noul", "goal": "Is evidence present?"}})
    with pytest.raises(ValueError):
        project_state(source.state, {"record": {"x": math.nan}})


def test_artifact_round_trip(program, tmp_path):
    path = tmp_path / "program.json"
    program.save(path)
    restored = Program.load(path)
    assert restored == program
    assert restored.content_hash == program.content_hash


@pytest.mark.parametrize("tamper", ["instructions", "provenance"])
def test_tampering_is_detected(program, tmp_path, tamper):
    path = tmp_path / "program.json"
    program.save(path)
    envelope = json.loads(path.read_text())
    if tamper == "instructions":
        envelope["program"]["questions"]["urgent"]["instructions"] = "Always say yes."
    else:
        envelope["program"]["provenance"]["status"] = "measured"
    atomic_json(path, envelope)
    with pytest.raises(DataError, match="checksum"):
        Program.load(path)


def test_execution_hash_excludes_metadata(program):
    a = program.content_hash
    program.provenance["note"] = "new note"
    assert a == program.content_hash


def test_plan_roundtrip(source, program):
    restored = plan_to_program(source, program_plan(program))
    assert restored.questions == program.questions


def test_plan_cannot_expand_input(source, program):
    plan = program_plan(program)
    plan["state_fields"].append("secret")
    with pytest.raises(CandidateError):
        plan_to_program(source, plan)


def test_plan_cannot_change_choices(source, program):
    plan = program_plan(program)
    plan["questions"]["department"]["criteria"]["new_label"] = "New option"
    with pytest.raises(CandidateError):
        plan_to_program(source, plan)


def test_program_rejects_unused_questions(program):
    data = program.model_dump()
    data["questions"]["unused"] = {"type": "noul", "instructions": "Is this unused?"}
    with pytest.raises(ValidationError):
        Program.model_validate(data)


def test_program_rejects_unknown_reference(program):
    data = program.model_dump()
    data["bindings"]["urgent"] = {"kind": "question", "question": "missing"}
    with pytest.raises(ValidationError):
        Program.model_validate(data)


def test_no_composed_binary_probabilities(program):
    data = program.model_dump()
    data["bindings"]["urgent"] = {"kind": "weighted_mean", "weights": {"urgent": 1}}
    with pytest.raises(ValidationError):
        Program.model_validate(data)


def test_splits_are_disjoint(source, splits):
    assert_disjoint(splits, source)


def test_cross_split_id_leakage(source, splits):
    splits["test"][0].id = splits["train"][0].id
    with pytest.raises(DataError, match="ID"):
        assert_disjoint(splits, source)


def test_projected_state_duplicate_ignores_extra_fields(source, splits):
    splits["test"][0].state = {**splits["train"][0].state, "new_metadata": "test"}
    with pytest.raises(DataError, match="model-visible"):
        assert_disjoint(splits, source)


def test_group_leakage(source, splits):
    splits["test"][0].group = splits["train"][0].group
    with pytest.raises(DataError, match="Group"):
        assert_disjoint(splits, source)


@pytest.mark.parametrize("label", ["false", 0, 1, None])
def test_noul_labels_are_real_booleans(source, splits, label):
    row = splits["train"][0].model_copy(deep=True)
    row.expected["urgent"] = label
    with pytest.raises(DataError):
        validate_example(row, source)


@pytest.mark.parametrize("label", [-1, 3, True, float("nan"), "1"])
def test_score_labels_are_bounded_numbers(source, splits, label):
    row = splits["train"][0].model_copy(deep=True)
    row.expected["frustration"] = label
    with pytest.raises(DataError):
        validate_example(row, source)


def test_empty_dataset(source, tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n")
    with pytest.raises(DataError, match="Empty"):
        read_jsonl(path, source)


def test_bad_json_hides_sensitive_content(source, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"state": "secret-sensitive-content"')
    with pytest.raises(DataError) as error:
        read_jsonl(path, source)
    assert "secret-sensitive-content" not in str(error.value)


def test_duplicate_json_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"a":1,"a":2}')
    with pytest.raises(DataError):
        load_document(path)


def test_duplicate_yaml_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("a: 1\na: 2\n")
    with pytest.raises(DataError):
        load_document(path)


def test_yaml_true_false_require_quotes(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("criteria:\n  true: visible\n  false: absent\n")
    with pytest.raises(DataError):
        load_document(path)


def test_yaml_does_not_execute_tags(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text('!!python/object/apply:os.system ["echo do-not-run"]')
    with pytest.raises(DataError):
        load_document(path)
