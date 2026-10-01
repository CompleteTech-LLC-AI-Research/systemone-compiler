"""Packaging metadata regression tests. No build, network, or inference required."""
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_license_uses_spdx_string_not_deprecated_table(pyproject):
    # setuptools removes `project.license` as a TOML table after 2027-02-18.
    license_value = pyproject["project"]["license"]
    assert isinstance(license_value, str), "Use an SPDX string, not the deprecated {text = ...} table."
    assert license_value == "MIT"
    assert pyproject["project"]["license-files"] == ["LICENSE"]
    assert (ROOT / "LICENSE").is_file()


def test_build_backend_requires_setuptools_supporting_spdx(pyproject):
    # The SPDX string form of project.license needs setuptools>=77.
    requires = pyproject["build-system"]["requires"]
    assert "setuptools>=77" in requires


def test_runtime_dependencies_exclude_optimizer_and_sdk(pyproject):
    runtime = " ".join(pyproject["project"]["dependencies"]).lower()
    for package in ("dspy", "gepa", "typesafe-sdk"):
        assert package not in runtime, f"{package} must stay an optional extra, not a runtime dependency."


def test_optional_extras_pin_documented_integration_targets(pyproject):
    extras = pyproject["project"]["optional-dependencies"]
    assert "typesafe-sdk==0.7.0" in extras["live"]
    assert "dspy[litellm]==3.3.1" in extras["optimize"]
    assert "gepa==0.1.4" in extras["optimize"]


def test_checked_in_json_schemas_match_the_models():
    import json
    from pathlib import Path
    from typewright.hierarchy import HierarchyArtifact, HierarchySource
    from typewright.models import Program, UseCase

    directory = Path(__file__).resolve().parents[1] / "schemas"
    expected = {"usecase.schema.json": UseCase, "program.schema.json": Program,
                "hierarchy-source.schema.json": HierarchySource,
                "hierarchy-artifact.schema.json": HierarchyArtifact}
    assert {path.name for path in directory.glob("*.json")} == set(expected)
    for name, model in expected.items():
        stored = json.loads((directory / name).read_text(encoding="utf-8"))
        assert stored == model.model_json_schema(), f"{name} is stale; run `typewright schema --out schemas`"
