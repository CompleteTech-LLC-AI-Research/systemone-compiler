"""Keep generated hierarchy content declarative across compiler and runtime modules."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from s1compiler.errors import DataError
from s1compiler.hierarchy import HierarchySource


ROOT = Path(__file__).resolve().parents[1]


def test_hierarchy_modules_have_no_dynamic_execution_or_pickle_loading():
    modules = sorted((ROOT / "src" / "s1compiler").glob("hierarchy*.py"))
    assert modules
    forbidden_imports = {"pickle", "subprocess", "shutil"}
    forbidden_calls = {"eval", "exec", "compile", "__import__"}
    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not {alias.name.split(".")[0] for alias in node.names} & forbidden_imports, module
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in forbidden_imports, module
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in forbidden_calls, module
                elif isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in {"system", "popen", "loads"} or not (
                        isinstance(node.func.value, ast.Name) and
                        node.func.value.id in {"os", "subprocess", "pickle"}), module


@pytest.mark.parametrize("payload", [
    '{"format":"systemone-hierarchy-source/v1","graph":',
    '{"format":"systemone-hierarchy-source/v1","source":{},"graph":{},"extra":"__import__(\'os\')"}',
])
def test_malformed_or_executable_looking_json_never_becomes_a_graph(tmp_path, payload):
    path = tmp_path / "candidate.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises((DataError, ValueError)):
        HierarchySource.load(path)
