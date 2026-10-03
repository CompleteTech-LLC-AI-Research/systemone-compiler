"""Synthetic policy mechanics; these are not measured Jev gains."""
import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from typewright.architect import template_program
from typewright.data import Example
from typewright.io import fingerprint
from typewright.models import Decision, Policy, Program, StateField, UseCase
from typewright.policy import fit_policies
from typewright.runtime import apply_policy


def fixture(kind, n=100):
    criteria = {"a": "A", "b": "B"} if kind == "choice" else ["low", "high"]
    source = UseCase(name="synthetic", state={"text": StateField()},
                     decisions={"output": Decision(type=kind, goal="Synthetic test", criteria=criteria)})
    program = template_program(source)
    qid = program.bindings["output"].question
    rows, results = [], []
    for i in range(n):
        if kind == "choice":
            # A shifted decision boundary: native argmax is A, policy should pick B.
            probabilities = {"a": .55, "b": .45} if i < n * .8 else {"a": .9, "b": .1}
            answer = {"type": kind, "choice": "a", "probabilities": probabilities, "confidence": .7}
            gold = "b" if i < n * .8 else "a"
        else:
            # A high ordinal level despite a continuous native expectation of .3.
            answer = {"type": kind, "score": .3, "probabilities": {"0": .7, "1": .3}, "confidence": .7}
            gold = 1
        rows.append(Example(id=f"cal-{i}", state={"text": str(i)}, expected={"output": gold}))
        answers = {qid: answer}
        results.append({"answers": answers, "decisions": apply_policy(program, answers), "synthetic": True})
    return program, rows, results


@pytest.mark.parametrize("kind", ["choice", "score"])
def test_synthetic_shift_fits_with_held_out_improvement(kind):
    program, rows, results = fixture(kind)
    fitted, report = fit_policies(program, rows, results, fit_decision_knobs=True, max_error=1)
    detail = report["output"]["decision_fit"]
    assert detail["status"] == "fitted"
    assert detail["synthetic"] is True
    assert detail["calibration_n"] == 100
    assert len(detail["folds"]) == 5
    assert all(f["candidate_loss"] < f["default_loss"] for f in detail["folds"])
    prediction = apply_policy(fitted, results[0]["answers"])["output"]
    assert prediction["value"] == rows[0].expected["output"]
    assert prediction["probabilities"] == results[0]["decisions"]["output"]["probabilities"]
    assert prediction["gate_score"] == (.45 if kind == "choice" else .3)
    assert "native_probability" in prediction["gate_basis"]
    assert program.policies["output"].fitting_version is None


@pytest.mark.parametrize("kind", ["choice", "score"])
def test_ties_and_small_samples_keep_defaults(kind):
    program, rows, results = fixture(kind)
    for row in rows:
        row.expected["output"] = "a" if kind == "choice" else .3
    fitted, report = fit_policies(program, rows, results, fit_decision_knobs=True)
    assert report["output"]["decision_fit"]["status"] == "unchanged"
    assert fitted.policies["output"].fitting_version is None
    fitted, report = fit_policies(program, rows[:4], results[:4], fit_decision_knobs=True)
    assert report["output"]["decision_fit"]["status"] == "unchanged"
    assert fitted.policies["output"].force_review


def test_grouped_fold_overfit_declined():
    program, rows, results = fixture("choice")
    # Four folds favor B; one independent group favors native A. Each fold has
    # enough support, but the held-out harmed fold rejects the fitted policy.
    for i, row in enumerate(rows):
        row.group = f"group-{i // 20}"
        if i < 20:
            row.expected["output"] = "a"
    fitted, report = fit_policies(program, rows, results, fit_decision_knobs=True)
    detail = report["output"]["decision_fit"]
    assert detail["status"] == "unchanged"
    assert len(detail["folds"]) == 5
    assert any(f["candidate_loss"] > f["default_loss"] for f in detail["folds"])
    assert fitted.policies["output"].choice_weights is None


@pytest.mark.parametrize("kwargs", [
    {"choice_weights": {"a": 0}}, {"score_cuts": [1, .5]},
    {"score_cuts": [float("nan")]}, {"choice_weights": {"a": float("inf")}},
])
def test_strict_knobs(kwargs):
    with pytest.raises(ValidationError):
        Policy(fitting_version="native-policy/v1", **kwargs)


def test_contract_bound_knobs():
    program, _, _ = fixture("score")
    data = program.model_dump()
    for cuts in ([.2, .8], [-.1], [1.1]):
        data["policies"]["output"] = {"fitting_version": "native-policy/v1", "score_cuts": cuts}
        with pytest.raises(ValidationError):
            Program.model_validate(data)
    program, _, _ = fixture("choice")
    data = program.model_dump()
    data["policies"]["output"] = {"fitting_version": "native-policy/v1", "choice_weights": {"a": 1}}
    with pytest.raises(ValidationError):
        Program.model_validate(data)


def test_legacy_checksum_serialization_and_new_roundtrip(tmp_path):
    program, rows, results = fixture("choice")
    legacy = program.model_dump(mode="json")
    assert set(legacy["policies"]["output"]) == {"noul_threshold", "min_gate", "force_review"}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"program": legacy, "sha256": fingerprint(legacy)}))
    loaded = Program.load(path)
    assert loaded.content_hash == program.content_hash
    assert loaded.model_dump(mode="json") == legacy
    assert apply_policy(loaded, results[0]["answers"]) == results[0]["decisions"]
    fitted, _ = fit_policies(program, rows, results, fit_decision_knobs=True, max_error=1)
    fitted.save(path)
    assert Program.load(path) == fitted


def test_runtime_import_boundary():
    subprocess.run([sys.executable, "-c", "import typewright.runtime; import sys; "
                    "assert 'dspy' not in sys.modules; assert 'gepa' not in sys.modules"], check=True)


def test_compile_validation_and_test_cannot_influence_fit(source, splits, backend, monkeypatch):
    import typewright.compiler as module
    from typewright.compiler import CompileOptions, Compiler
    seen = []
    original = module.fit_policies

    def spy(program, rows, results, **kwargs):
        assert [r.id for r in rows] == [r.id for r in splits["calibration"]]
        seen.append(original(program, rows, results, **kwargs)[1])
        return original(program, rows, results, **kwargs)

    monkeypatch.setattr(module, "fit_policies", spy)
    options = CompileOptions(min_calibration_samples=3)
    Compiler(backend, options=options).compile(source, **splits)
    # With template/no optimizer, neither validation nor test labels select any
    # prompt. Deliberately corrupt those labels to prove they cannot fit knobs.
    for split in ("validation", "test"):
        for row in splits[split]:
            row.expected["urgent"] = not row.expected["urgent"]
            row.expected["department"] = next(iter(source.decisions["department"].criteria))
            row.expected["frustration"] = 0
    Compiler(backend, options=options).compile(source, **splits)
    assert seen[0] == seen[1]


def test_fitted_inspect_and_schema(tmp_path, capsys):
    from typewright.cli import main
    program, rows, results = fixture("score")
    fitted, _ = fit_policies(program, rows, results, fit_decision_knobs=True, max_error=1)
    path = tmp_path / "fitted.json"
    fitted.save(path)
    assert main(["inspect", str(path)]) == 0
    policy = json.loads(capsys.readouterr().out)["policies"]["output"]
    assert policy["fitting_version"] == "native-policy/v1"
    assert policy["score_cuts"] == fitted.policies["output"].score_cuts
    assert main(["schema", "--out", str(tmp_path / "schemas")]) == 0
    schema = json.loads((tmp_path / "schemas/program.schema.json").read_text())
    assert {"score_cuts", "choice_weights", "fitting_version"} <= set(schema["$defs"]["Policy"]["properties"])


def test_choice_weight_tie_preserves_native_label():
    program, _, results = fixture("choice")
    program.policies["output"] = Policy(fitting_version="native-policy/v1", choice_weights={"a": 1, "b": 1})
    answers = results[0]["answers"]
    answer = next(iter(answers.values()))
    answer.update(choice="b", probabilities={"a": .5, "b": .5})
    assert apply_policy(program, answers)["output"]["value"] == "b"


def test_composite_score_cuts_keep_heuristic_gate():
    from typewright.models import Binding, Question
    program, _, results = fixture("score")
    qid = program.bindings["output"].question
    program.questions["extra"] = Question(type="noul", instructions="Synthetic component")
    program.bindings["output"] = Binding(kind="weighted_mean", weights={qid: 1, "extra": 1})
    program.policies["output"] = Policy(fitting_version="native-policy/v1", score_cuts=[.2])
    answers = {**results[0]["answers"], "extra": {"type": "noul", "noul": .3}}
    prediction = apply_policy(program, answers)["output"]
    assert prediction["value"] == 1
    assert prediction["probabilities"] is None
    assert prediction["vendor_confidence"] is None
    assert prediction["gate_score"] == .7
    assert prediction["gate_basis"] == "minimum_component_gate_heuristic_after_score_cuts"


def test_checked_in_pre_extension_artifact_keeps_hash_and_payload(tmp_path):
    from pathlib import Path
    # This committed artifact predates the extension; its stored checksum and
    # payload are independent of the new Policy serializer.
    path = Path(__file__).resolve().parents[1] / "examples/compiled_demo/program.s1.json"
    original = json.loads(path.read_text())
    assert all(set(policy) == {"noul_threshold", "min_gate", "force_review"}
               for policy in original["program"]["policies"].values())
    loaded = Program.load(path)
    assert fingerprint(loaded.model_dump(mode="json")) == original["sha256"]
    output = tmp_path / "saved.json"
    loaded.save(output)
    assert json.loads(output.read_text()) == original


@pytest.mark.parametrize("kind", ["choice", "score"])
def test_successful_compile_fit_isolated_from_validation_and_test(kind, monkeypatch):
    import typewright.compiler as module
    from typewright.backends import ManagedBackend, MockBackend
    from typewright.compiler import CompileOptions, Compiler
    from typewright.metrics import report_from_results

    program, calibration, cached = fixture(kind)
    source = UseCase(name=program.name, model=program.model, state=program.state, decisions=program.decisions)
    splits = {"calibration": calibration}
    for split in ("train", "validation", "test"):
        splits[split] = [Example(id=f"{split}-{i}", state={"text": f"{split}-{i}"},
                                expected={"output": "a" if kind == "choice" else 0}) for i in range(10)]
    seen = []
    original_fit = module.fit_policies

    def cached_evaluation(selected, rows, backend):
        answers = cached[0]["answers"]
        results = [{"answers": answers, "decisions": apply_policy(selected, answers),
                    "synthetic": True, "latency_ms": 0, "cache_hit": True} for row in rows]
        return report_from_results(selected, rows, results), results

    def capture(selected, rows, results, **kwargs):
        assert [row.id for row in rows] == [row.id for row in calibration]
        fitted, report = original_fit(selected, rows, results, **kwargs)
        assert report["output"]["decision_fit"]["status"] == "fitted"
        seen.append((fitted.policies["output"].model_dump(), report))
        return fitted, report

    monkeypatch.setattr(module, "evaluate", cached_evaluation)
    monkeypatch.setattr(module, "fit_policies", capture)
    backend = ManagedBackend(MockBackend())
    try:
        compiler = Compiler(backend, options=CompileOptions(min_calibration_samples=10, fit_decision_knobs=True))
        compiler.compile(source, **splits)
        for split in ("validation", "test"):
            for row in splits[split]:
                row.state["text"] += " VALIDATION_TEST_SENTINEL"
                row.expected["output"] = "b" if kind == "choice" else 1
        compiler.compile(source, **splits)
    finally:
        backend.close()
    assert seen[0] == seen[1]


def test_decision_fitting_opt_in_default_preserves_selection():
    program, rows, results = fixture("choice")
    fitted, report = fit_policies(program, rows, results, max_error=1)
    assert report["output"]["decision_fit"]["status"] == "disabled"
    assert fitted.policies["output"].choice_weights is None
    assert apply_policy(fitted, results[0]["answers"])["output"]["value"] == "a"


def test_cli_opt_in_reaches_flat_compiler(tmp_path, monkeypatch):
    import typewright.cli as cli
    original = cli.Compiler
    captured = []

    class Capture(original):
        def compile(self, source, **splits):
            captured.append(self.options.fit_decision_knobs)
            return super().compile(source, **splits)

    monkeypatch.setattr(cli, "Compiler", Capture)
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "examples/support_triage"
    args = ["compile", str(root / "usecase.yaml"), "--out", str(tmp_path / "compile"),
            "--fit-decision-knobs"]
    for split in ("train", "validation", "calibration", "test"):
        args += [f"--{split}", str(root / f"{split}.jsonl")]
    assert cli.main(args) == 0
    assert captured == [True]


def test_cli_rejects_hierarchy_fitting_before_calls(tmp_path, monkeypatch, capsys):
    import typewright.cli as cli
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "examples/hierarchy/support"

    def forbidden_backend(args):
        raise AssertionError("Unsupported fitting must fail before any backend is made")

    monkeypatch.setattr(cli, "make_backend", forbidden_backend)
    args = ["compile", str(root / "source.json"), "--out", str(tmp_path / "compile"),
            "--fit-decision-knobs"]
    for split in ("train", "validation", "calibration", "test"):
        args += [f"--{split}", str(root / f"{split}.jsonl")]
    assert cli.main(args) == 2
    assert "flat programs only" in capsys.readouterr().err
