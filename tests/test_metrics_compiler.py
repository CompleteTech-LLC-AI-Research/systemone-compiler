import pytest
from s1compiler.compiler import Compiler, CompileOptions
from s1compiler.backends import ManagedBackend, MockBackend
from s1compiler.errors import ConfigurationError, DataError
from s1compiler.metrics import calibration_error, example_quality, report_from_results, wilson_upper
from s1compiler.policy import fit_policies
from s1compiler.runtime import Runtime


def perfect_result(program, row):
    decisions = {}
    for key, spec in program.decisions.items():
        gold = row.expected[key]
        probs = None
        if spec.type == "choice":
            probs = {label: float(label == gold) for label in spec.criteria}
        decisions[key] = {"value": gold, "probabilities": probs,
                          "p_true": float(gold) if spec.type == "noul" else None, "review_required": False}
    return {"decisions": decisions, "synthetic": True, "cache_hit": False, "latency_ms": 1}


def test_perfect_objective_and_metrics(program, splits):
    row = splits["train"][0]
    result = perfect_result(program, row)
    assert example_quality(program, row, result) == 1
    report = report_from_results(program, [row], [result])
    assert report["decisions"]["department"]["brier"] == 0
    assert report["decisions"]["frustration"]["mae"] == 0


def test_binary_probability_quality_is_smooth(program, splits):
    row = splits["train"][0]
    result = perfect_result(program, row)
    result["decisions"]["urgent"]["p_true"] = .4
    a = example_quality(program, row, result)
    result["decisions"]["urgent"]["p_true"] = .49
    b = example_quality(program, row, result)
    assert a > b


def test_abstention_does_not_improve_objective(program, splits):
    row = splits["train"][0]
    result = perfect_result(program, row)
    before = example_quality(program, row, result)
    for value in result["decisions"].values():
        value["review_required"] = True
    assert example_quality(program, row, result) == before
    report = report_from_results(program, [row], [result])
    assert report["decisions"]["department"]["selective_error"] is None
    assert report["decisions"]["department"]["coverage"] == 0


def test_calibration_error_known_case():
    assert calibration_error([(1.0, True), (1.0, True)]) == 0
    assert calibration_error([(1.0, False), (1.0, False)]) == 1
    assert calibration_error([]) is None


def test_wilson_small_samples_not_zero():
    assert 0 < wilson_upper(0, 10) < .5
    assert wilson_upper(0, 0) is None


def test_policy_insufficient_evidence_means_review(program, splits, backend):
    rows = splits["calibration"][:2]
    predictions = [Runtime(program, backend).run(r.state) for r in rows]
    fitted, report = fit_policies(program, rows, predictions, min_samples=10)
    assert all(p.force_review for p in fitted.policies.values())
    assert all(value["status"] == "review_only" for value in report.values())
    assert not any(p.force_review for p in program.policies.values())


def test_compile_artifact_and_no_fake_improvement(source, splits, backend):
    program, report = Compiler(backend, options=CompileOptions(min_calibration_samples=3)).compile(source, **splits)
    assert report["status"] == "synthetic"
    assert program.provenance["deployment_approved"] is False
    assert report["optimization"]["engine"] == "none"
    assert report["test_objective_delta"] == 0
    assert set(program.provenance["splits"]) == set(splits)
    assert report["teacher"] is None
    assert backend.cache_hits > 0


def test_leakage_fails_before_any_calls(source, splits, backend):
    splits["test"][0].id = splits["train"][0].id
    with pytest.raises(DataError):
        Compiler(backend).compile(source, **splits)
    assert backend.budget.used == 0


def test_teacher_required(source, splits, backend):
    with pytest.raises(ConfigurationError, match="teacher"):
        Compiler(backend, options=CompileOptions(architect="dspy")).compile(source, **splits)
    assert backend.budget.used == 0


def test_teacher_never_receives_test_or_extra_state(source, splits, backend):
    captured = []
    class Teacher:
        model = "test-double"
        def propose_plan(self, source, program, training_feedback):
            captured.append(str(training_feedback))
            return program.model_copy(deep=True)
        def accounting(self):
            return {"test_double": True}
    for row in splits["train"]:
        row.state["secret"] = "DO_NOT_SEND_EXTRA"
    for row in splits["test"]:
        row.state["message"] += " HOLDOUT_SENTINEL_884"
    Compiler(backend, teacher=Teacher(), options=CompileOptions(architect="dspy", structural_rounds=1)).compile(source, **splits)
    assert len(captured) == 2
    assert all("HOLDOUT_SENTINEL_884" not in trace for trace in captured)
    assert all("DO_NOT_SEND_EXTRA" not in trace for trace in captured)


def test_testset_runs_after_calibration(source, splits):
    visited = []
    class Recorder(MockBackend):
        def evaluate(self, program, state):
            visited.append(state["message"])
            return super().evaluate(program, state)
    backend = ManagedBackend(Recorder())
    Compiler(backend).compile(source, **splits)
    test_messages = {r.state["message"] for r in splits["test"]}
    calibration_messages = {r.state["message"] for r in splits["calibration"]}
    first_test = min(i for i, msg in enumerate(visited) if msg in test_messages)
    last_calibration = max(i for i, msg in enumerate(visited) if msg in calibration_messages)
    assert last_calibration < first_test
    backend.close()


@pytest.mark.parametrize("kwargs", [{"structural_rounds": -1}, {"max_calibration_error": 2}, {"optimizer": "fake"}])
def test_invalid_compile_options(kwargs):
    with pytest.raises(ConfigurationError):
        CompileOptions(**kwargs)
