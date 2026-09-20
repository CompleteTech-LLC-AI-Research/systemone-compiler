from pathlib import Path
import pytest
from s1compiler.architect import template_program
from s1compiler.backends import AnswerCache, ManagedBackend, MockBackend
from s1compiler.data import read_jsonl
from s1compiler.models import UseCase

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source():
    return UseCase.load(ROOT / "examples/support_triage/usecase.yaml")


@pytest.fixture
def program(source):
    return template_program(source)


@pytest.fixture
def splits(source):
    directory = ROOT / "examples/support_triage"
    return {name: read_jsonl(directory / f"{name}.jsonl", source)
            for name in ("train", "validation", "calibration", "test")}


@pytest.fixture
def backend():
    managed = ManagedBackend(MockBackend(), max_calls=1000, cache=AnswerCache())
    yield managed
    managed.close()
