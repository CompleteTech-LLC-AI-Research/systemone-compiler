"""Portable hierarchy artifact and v1 dispatch regression checks."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from s1compiler.cli import main
from s1compiler.errors import DataError
from s1compiler.hierarchy import HierarchyArtifact, HierarchySource, load_artifact, lower_hierarchy
from s1compiler.io import atomic_json
from s1compiler.models import Program


FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "hierarchy_contract"


@pytest.mark.parametrize("name", ["chain", "conditional", "diamond", "nested"])
def test_source_and_frozen_artifact_round_trip_without_source_directory(name, tmp_path):
    source = HierarchySource.load(FIXTURES / f"{name}.json")
    artifact = lower_hierarchy(source)
    target = tmp_path / "portable.json"
    artifact.save(target)
    reloaded = load_artifact(target)
    assert isinstance(reloaded, HierarchyArtifact)
    assert reloaded == artifact
    assert reloaded.content_hash == artifact.content_hash
    assert all(node.program.model == source.source.model for node in reloaded.nodes)
    assert not any(key in reloaded.model_dump_json() for key in ("definitions", "include", "file://"))


def test_nested_definitions_have_distinct_instances_and_source_map():
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / "nested.json"))
    assert [node.id for node in artifact.nodes] == ["first/check", "second/check"]
    assert artifact.source_to_nodes == {"check_note:check": ["first/check", "second/check"]}
    assert [export.id for export in artifact.exports] == ["first", "second"]


@pytest.mark.parametrize("mutation", [
    lambda data: data["definitions"]["check_note"]["stages"][0]["program"]["questions"]["review"].update(
        instructions="Different prompt"),
    lambda data: data["graph"]["stages"][0]["inputs"].update(note={"root": "second_note"}),
    lambda data: data["definitions"]["check_note"]["stages"][0]["program"]["policies"]["review"].update(
        min_gate=0.8),
])
def test_executable_change_changes_content_hash(mutation):
    original = json.loads((FIXTURES / "nested.json").read_text(encoding="utf-8"))
    changed = copy.deepcopy(original)
    mutation(changed)
    assert lower_hierarchy(HierarchySource.model_validate(changed)).content_hash != lower_hierarchy(
        HierarchySource.model_validate(original)).content_hash


def test_changed_route_condition_changes_content_hash():
    original = json.loads((FIXTURES / "conditional.json").read_text(encoding="utf-8"))
    changed = copy.deepcopy(original)
    changed["graph"]["stages"][1]["when"]["all"][0]["value"] = "other"
    assert lower_hierarchy(HierarchySource.model_validate(changed)).content_hash != lower_hierarchy(
        HierarchySource.model_validate(original)).content_hash


def test_stage_list_order_does_not_change_semantic_hash():
    original = json.loads((FIXTURES / "nested.json").read_text(encoding="utf-8"))
    reordered = copy.deepcopy(original)
    reordered["graph"]["stages"].reverse()
    assert lower_hierarchy(HierarchySource.model_validate(reordered)).content_hash == lower_hierarchy(
        HierarchySource.model_validate(original)).content_hash


def test_subgraph_guard_is_frozen_into_execution_hash():
    original = json.loads((FIXTURES / "nested.json").read_text(encoding="utf-8"))
    guarded = copy.deepcopy(original)
    guarded["graph"]["stages"][0]["when"] = {"all": [{
        "ref": {"root": "first_note"}, "op": "eq", "value": "urgent",
    }]}
    first = lower_hierarchy(HierarchySource.model_validate(original))
    second = lower_hierarchy(HierarchySource.model_validate(guarded))
    assert second.content_hash != first.content_hash
    assert second.exports[0].when is not None


def test_tampered_and_unsupported_artifacts_fail_before_load(tmp_path):
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / "nested.json"))
    target = tmp_path / "artifact.json"
    artifact.save(target)
    data = json.loads(target.read_text(encoding="utf-8"))
    data["artifact"]["nodes"][0]["program"]["questions"]["review"]["instructions"] = "tampered"
    atomic_json(target, data)
    with pytest.raises(DataError, match="checksum"):
        load_artifact(target)
    data["artifact"]["format"] = "systemone-hierarchy/v2"
    atomic_json(target, data)
    with pytest.raises(DataError, match="Unsupported"):
        load_artifact(target)


def test_rehashed_but_inconsistent_artifact_is_rejected(tmp_path):
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / "nested.json"))
    data = artifact.model_dump(mode="json")
    data["nodes"][0]["inputs"]["note"] = {"root": "undeclared"}
    target = tmp_path / "bad.json"
    from s1compiler.io import fingerprint
    atomic_json(target, {"artifact": data, "sha256": fingerprint(data)})
    with pytest.raises(ValidationError, match="Unknown root state reference"):
        load_artifact(target)
    data["nodes"][0]["inputs"]["note"] = {"root": "first_note"}
    data["nodes"][0]["id"] = data["nodes"][1]["id"]
    atomic_json(target, {"artifact": data, "sha256": fingerprint(data)})
    with pytest.raises(ValidationError, match="node IDs"):
        load_artifact(target)


def test_recursive_missing_and_nonfinite_source_rejected():
    original = json.loads((FIXTURES / "nested.json").read_text(encoding="utf-8"))
    recursive = copy.deepcopy(original)
    recursive["definitions"]["check_note"]["stages"] = [{
        "id": "again", "kind": "subgraph", "definition": "check_note", "inputs": {"note": {"root": "note"}},
    }]
    with pytest.raises((DataError, ValidationError), match="Recursive"):
        lower_hierarchy(HierarchySource.model_validate(recursive))
    missing = copy.deepcopy(original)
    missing["graph"]["stages"][0]["definition"] = "unknown"
    with pytest.raises((DataError, ValidationError), match="Missing"):
        lower_hierarchy(HierarchySource.model_validate(missing))
    nonfinite = copy.deepcopy(original)
    nonfinite["limits"]["max_expanded_nodes"] = float("nan")
    with pytest.raises(ValidationError):
        HierarchySource.model_validate(nonfinite)


def test_measured_wrapper_cannot_launder_draft_children():
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / "nested.json"))
    data = artifact.model_dump(mode="json")
    data["provenance"] = {"status": "measured", "composition_measured": True}
    with pytest.raises(ValidationError, match="composition report checksum"):
        HierarchyArtifact.model_validate(data)
    data["provenance"]["evidence"] = {"composition_report_sha256": "f" * 64}
    with pytest.raises(ValidationError, match="unmeasured leaves"):
        HierarchyArtifact.model_validate(data)


def test_provenance_hash_is_separate_from_semantic_hash():
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / "nested.json"))
    data = artifact.model_dump(mode="json")
    data["nodes"][0]["program"]["provenance"]["architect"] = "reviewed-template"
    revised = HierarchyArtifact.model_validate(data)
    assert revised.content_hash == artifact.content_hash
    assert revised.provenance_hash != artifact.provenance_hash


@pytest.mark.parametrize("bad", ["../secret", "https://example.org/include", "foo/bar"])
def test_unsafe_definition_reference_rejected(bad):
    data = json.loads((FIXTURES / "nested.json").read_text(encoding="utf-8"))
    data["graph"]["stages"][0]["definition"] = bad
    with pytest.raises(ValidationError, match="identifier"):
        HierarchySource.model_validate(data)


def test_unbound_subgraph_input_cannot_fall_through_to_root():
    data = json.loads((FIXTURES / "nested.json").read_text(encoding="utf-8"))
    data["definitions"]["check_note"]["stages"][0]["inputs"]["note"] = {"root": "first_note"}
    with pytest.raises(DataError, match="Unbound subgraph input"):
        lower_hierarchy(HierarchySource.model_validate(data))


def test_schema_export_and_flat_dispatch(tmp_path):
    assert main(["schema", "--out", str(tmp_path / "schemas")]) == 0
    for name in ("hierarchy-source", "hierarchy-artifact"):
        schema = json.loads((tmp_path / "schemas" / f"{name}.schema.json").read_text())
        assert schema["type"] == "object"
    leaf = HierarchySource.load(FIXTURES / "chain.json").graph.stages[0].program
    flat = tmp_path / "flat.json"
    leaf.save(flat)
    loaded = load_artifact(flat)
    assert isinstance(loaded, Program)
    assert loaded.content_hash == leaf.content_hash
    assert loaded.content_hash == "da59a9ebe99448503ca3f9916c95cfef4dcbdbd64f594b5b184c1d682fa4a7c4"


def test_import_and_frozen_load_need_no_optimizer_or_vendor_sdk(tmp_path):
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / "nested.json"))
    path = tmp_path / "artifact.json"
    artifact.save(path)
    probe = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'dspy', 'gepa', 'typesafe_sdk', 'typesafe'}:
        raise AssertionError('optional dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from s1compiler import load_artifact
assert len(load_artifact(__import__('sys').argv[1]).nodes) == 2
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(FIXTURES.parents[1] / "src")
    completed = subprocess.run([sys.executable, "-c", probe, str(path)], capture_output=True, text=True,
                               env=env)
    assert completed.returncode == 0, completed.stderr
