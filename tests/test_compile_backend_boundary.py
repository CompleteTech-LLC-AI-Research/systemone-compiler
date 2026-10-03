"""Compile/runtime backend selection using authored no-dispatch provider doubles."""
from types import SimpleNamespace

import pytest

from typewright import banking77, cli, hierarchy_study, research
from typewright import dspy_typesafe_backend


class ProviderDouble:
    synthetic = False
    identity = "authored-provider-double"

    def __init__(self, **kwargs):
        self.configuration = kwargs

    def close(self):
        pass


@pytest.fixture
def providers(monkeypatch):
    class CompileDouble(ProviderDouble):
        pass

    class RuntimeDouble(ProviderDouble):
        pass

    monkeypatch.setattr(dspy_typesafe_backend, "DSPyTypeSafeBackend", CompileDouble)
    monkeypatch.setattr(cli, "TypeSafeBackend", RuntimeDouble)
    monkeypatch.setattr(banking77, "TypeSafeBackend", RuntimeDouble)
    monkeypatch.setattr(hierarchy_study, "TypeSafeBackend", RuntimeDouble)
    return CompileDouble, RuntimeDouble


@pytest.mark.parametrize("command", ["compile", "run", "evaluate"])
def test_cli_native_backend_preserves_compile_runtime_boundary(command, providers):
    compile_type, runtime_type = providers
    args = SimpleNamespace(command=command, backend="typesafe", allow_paid=True,
                           request_timeout=7, max_calls=3, cache=None, no_cache=True)
    backend = cli.make_backend(args)
    try:
        assert type(backend.backend) is (compile_type if command == "compile" else runtime_type)
        assert backend.backend.configuration == {"allow_paid": True, "timeout": 7}
        assert backend.budget.used == 0
        assert backend.budget.maximum == 3
        assert backend.cache is None
    finally:
        backend.close()


@pytest.mark.parametrize("compile_time", [False, True])
def test_flat_research_factory_phase_is_explicit(compile_time, providers):
    compile_type, runtime_type = providers
    backend = banking77.make_backend(True, 2, compile_time=compile_time)
    try:
        assert type(backend.backend) is (compile_type if compile_time else runtime_type)
        assert backend.budget.used == 0
        assert backend.budget.maximum == 2
        assert backend.cache is None
    finally:
        backend.close()


@pytest.mark.parametrize("phase", ["selection", "test"])
def test_hierarchy_study_backend_phase_and_ceiling(phase, providers):
    compile_type, runtime_type = providers
    protocol = {"mode": "typesafe", "budgets": {"by_arm": {"arm": {
        "allocated_selection_ceiling": 2, "allocated_test_ceiling": 3}}}}
    backend = hierarchy_study._study_backend(protocol, "arm", phase=phase, allow_paid=True)
    try:
        assert type(backend.backend) is (compile_type if phase == "selection" else runtime_type)
        assert backend.budget.maximum == (2 if phase == "selection" else 3)
        assert backend.budget.used == 0
        assert backend.cache is None
    finally:
        backend.close()


def test_research_selection_default_factory_uses_compile_phase(tmp_path, monkeypatch):
    class FactoryObserved(Exception):
        pass

    observed = []

    def factory(live, ceiling, *, compile_time=False):
        observed.append((live, ceiling, compile_time))
        raise FactoryObserved

    monkeypatch.setattr(research, "validate_protocol", lambda *_: {"task": (None, {}, {}, {})})
    monkeypatch.setattr(research, "make_backend", factory)
    protocol = {"backend": "typesafe", "seeds": [7], "search_calls_per_arm_seed": 3}
    with pytest.raises(FactoryObserved):
        research.select(protocol, tmp_path / "selection")
    assert observed == [(True, 3, True)]
