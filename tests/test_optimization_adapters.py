import dataclasses
import json
import sys
import types
import pytest
from typewright.architect import DSPyTeacher
from typewright.errors import CandidateError, ConfigurationError, DataError
from typewright.gepa_adapter import JevGEPAAdapter, components_from_program, program_from_components, optimize_gepa


@dataclasses.dataclass
class Batch:
    outputs: list
    scores: list
    trajectories: list | None


class FixedTeacher:
    def propose_components(self, candidate, reflective_dataset, components):
        return {k: candidate[k] for k in components}


def test_component_roundtrip(program):
    parts = components_from_program(program)
    restored = program_from_components(program, parts)
    assert restored == program
    assert all(isinstance(v, str) for v in parts.values())


def test_component_edit_does_not_mutate_seed(program):
    parts = components_from_program(program)
    parts["urgent/instructions"] = json.dumps("Is the customer currently blocked?")
    restored = program_from_components(program, parts)
    assert restored.questions["urgent"].instructions == "Is the customer currently blocked?"
    assert restored.questions["urgent"].instructions != program.questions["urgent"].instructions


@pytest.mark.parametrize("invalid", ["not json", "42", "true", "null"])
def test_bad_instruction_mutation_rejected(program, invalid):
    parts = components_from_program(program)
    parts["urgent/instructions"] = invalid
    with pytest.raises(CandidateError):
        program_from_components(program, parts)


def test_component_key_contract(program):
    parts = components_from_program(program)
    parts["injected/extra"] = '"bad"'
    with pytest.raises(CandidateError):
        program_from_components(program, parts)


def test_gepa_adapter_shapes_and_feedback(program, backend, splits):
    adapter = JevGEPAAdapter(program, backend, FixedTeacher(), batch_factory=Batch,
                             train_rows=splits["train"])
    candidate = components_from_program(program)
    evaluated = adapter.evaluate(splits["train"][:2], candidate, capture_traces=True)
    assert len(evaluated.outputs) == len(evaluated.scores) == len(evaluated.trajectories) == 2
    assert all(0 <= score <= 1 for score in evaluated.scores)
    feedback = adapter.make_reflective_dataset(candidate, evaluated, ["urgent/instructions"])
    assert "Feedback" in feedback["urgent/instructions"][0]
    assert "expected" in feedback["urgent/instructions"][0]["Feedback"]
    proposed = adapter.propose_new_texts(candidate, feedback, ["urgent/instructions"])
    assert set(proposed) == {"urgent/instructions"}


def test_gepa_no_traces_when_not_requested(program, backend, splits):
    adapter = JevGEPAAdapter(program, backend, FixedTeacher(), batch_factory=Batch)
    result = adapter.evaluate(splits["train"][:1], components_from_program(program))
    assert result.trajectories is None


@pytest.mark.parametrize("split", ["validation", "calibration", "test"])
def test_gepa_reflection_rejects_every_nontrain_split_before_teacher_or_backend(program, backend, splits, split):
    class SpyTeacher:
        calls = 0

        def propose_components(self, *_):
            self.calls += 1
            return {}

    teacher = SpyTeacher()
    adapter = JevGEPAAdapter(program, backend, teacher, batch_factory=Batch,
                             train_rows=splits["train"])
    candidate = components_from_program(program)
    before = backend.budget.used
    with pytest.raises(DataError, match="registered train"):
        adapter.evaluate(splits[split][:1], candidate, capture_traces=True)
    assert backend.budget.used == before
    fake = Batch(outputs=[], scores=[], trajectories=[{"input": f"{split}-sentinel"}])
    with pytest.raises(DataError, match="not captured"):
        adapter.make_reflective_dataset(candidate, fake, ["urgent/instructions"])
    with pytest.raises(DataError, match="not built"):
        adapter.propose_new_texts(candidate, {"urgent/instructions": [{"Inputs": f"{split}-sentinel"}]},
                                  ["urgent/instructions"])
    assert teacher.calls == 0


def test_invalid_gepa_candidate_no_backend_call(program, backend, splits):
    adapter = JevGEPAAdapter(program, backend, FixedTeacher(), batch_factory=Batch,
                             train_rows=splits["train"])
    result = adapter.evaluate(splits["train"][:2], {"bad": "bad"}, capture_traces=True)
    assert result.scores == [0, 0]
    assert len(result.trajectories) == 2
    assert backend.budget.used == 0


def test_mutation_rollback(program, backend, splits):
    class BadTeacher:
        def propose_components(self, *args):
            return {"urgent/instructions": "5"}
    adapter = JevGEPAAdapter(program, backend, BadTeacher(), batch_factory=Batch,
                             train_rows=splits["train"])
    candidate = components_from_program(program)
    batch = adapter.evaluate(splits["train"][:1], candidate, capture_traces=True)
    reflection = adapter.make_reflective_dataset(candidate, batch, ["urgent/instructions"])
    result = adapter.propose_new_texts(candidate, reflection, ["urgent/instructions"])
    assert result["urgent/instructions"] == candidate["urgent/instructions"]
    assert adapter.rejected == 1


def test_gepa_invocation_uses_native_adapter_not_chat_lm(program, backend, splits, monkeypatch):
    captured = {}
    def optimize(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(best_candidate=kwargs["seed_candidate"])
    monkeypatch.setitem(sys.modules, "gepa", types.SimpleNamespace(optimize=optimize))
    restored, report = optimize_gepa(program, splits["train"], splits["validation"], backend,
                                    FixedTeacher(), max_metric_calls=32)
    assert restored == program
    assert isinstance(captured["adapter"], JevGEPAAdapter)
    assert "task_lm" not in captured and "run_dir" not in captured
    assert captured["valset"] is splits["validation"]
    assert report["engine"] == "gepa"


def test_teacher_consent_before_import():
    with pytest.raises(ConfigurationError, match="consent"):
        DSPyTeacher("test/model")
    with pytest.raises(ConfigurationError, match="consent"):
        DSPyTeacher("test/model", allow_paid=True)


def test_teacher_requires_model():
    with pytest.raises(ConfigurationError, match="Specify"):
        DSPyTeacher("", allow_paid=True, share_feedback=True)


def test_dspy_component_serialization_without_model_call():
    teacher = object.__new__(DSPyTeacher)
    teacher.rejected = 0
    teacher.revise = object()
    teacher._predict = lambda *a, **kw: types.SimpleNamespace(revised_components_json='{"flag/instructions":"Is it present?"}')
    result = teacher.propose_components({"flag/instructions": '"Check it"'}, {}, ["flag/instructions"])
    assert json.loads(result["flag/instructions"]) == "Is it present?"


def test_dspy_bad_component_keys_rejected():
    teacher = object.__new__(DSPyTeacher)
    teacher.rejected = 0
    teacher.revise = object()
    teacher._predict = lambda *a, **kw: types.SimpleNamespace(revised_components_json='{"new":"bad"}')
    with pytest.raises(CandidateError):
        teacher.propose_components({"flag/instructions": '"Check it"'}, {}, ["flag/instructions"])
    assert teacher.rejected == 1
