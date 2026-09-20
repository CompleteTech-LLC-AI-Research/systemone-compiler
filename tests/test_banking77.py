import json
import random
import subprocess
import sys

import pytest

from s1compiler import banking77 as bench
from s1compiler.architect import template_program
from s1compiler.backends import ManagedBackend, MockBackend
from s1compiler.data import Example, assert_disjoint, dataset_hash
from s1compiler.errors import BackendError, CandidateError, ConfigurationError, DataError
from s1compiler.io import atomic_json, fingerprint, load_document


@pytest.fixture
def experiment():
    source = bench.baseline_source()
    splits = {name: [Example(id=f"{name}-{i}", state={"text": f"{name} request {i}"},
                            expected={"intent": "card_arrival" if i % 2 else "card_not_working"})
                     for i in range(4)] for name in bench.SPLITS}
    manifest = {"source_sha256": fingerprint(source.model_dump(mode="json")),
                "splits": {k: {"n": len(v), "sha256": dataset_hash(v)} for k, v in splits.items()}}
    return source, splits, manifest


def mock_factory(limit):
    return ManagedBackend(MockBackend(), max_calls=limit, cache=None)


def test_import_keeps_runtime_dependencies_lazy():
    result = subprocess.run([sys.executable, "-c", "import sys; import s1compiler; "
        "assert not any(x in sys.modules for x in ('dspy', 'gepa', 'typesafe_sdk'))"], capture_output=True)
    assert result.returncode == 0, result.stderr


def test_pinned_csv_rejects_corruption():
    with pytest.raises(DataError, match="checksum"):
        bench.parse_csv(b"text,category\nchanged,card_arrival\n", "train")


def test_stratified_split_reproducible_and_disjoint(experiment):
    source, splits, _ = experiment
    rows = [Example(id=str(i), state={"text": f"unique {i}"},
                    expected={"intent": "card_arrival" if i < 10 else "card_not_working"}) for i in range(20)]
    first = bench.partition(rows, 7)
    assert first == bench.partition(list(reversed(rows)), 7)
    assert first != bench.partition(rows, 17)
    assert [len(first[k]) for k in first] == [12, 4, 4]
    assert_disjoint(dict(first, test=splits["test"]), source)


def test_manifest_detects_dataset_and_source_changes(experiment, tmp_path):
    source, splits, manifest = experiment
    atomic_json(tmp_path / "usecase.json", source.model_dump(mode="json"))
    atomic_json(tmp_path / "dataset.json", manifest)
    for name, rows in splits.items():
        (tmp_path / f"{name}.jsonl").write_text("\n".join(r.model_dump_json() for r in rows), encoding="utf-8")
    assert bench.load_dataset(tmp_path)[2] == manifest
    path = tmp_path / "train.jsonl"
    path.write_text(path.read_text().replace("train request 0", "tampered input"), encoding="utf-8")
    with pytest.raises(DataError, match="manifest mismatch"):
        bench.load_dataset(tmp_path)


def test_no_feedback_control_always_starts_from_baseline(experiment):
    source, splits, _ = experiment
    base = template_program(source)
    calls = []
    class Teacher:
        def propose_components(self, candidate, feedback, components):
            calls.append((candidate, feedback))
            edited = dict(candidate)
            edited["intent/instructions"] = json.dumps("Classify the request in `text`.")
            return edited
    with_backend = mock_factory(100)
    bench.select_arm("D", base, splits, with_backend, Teacher(), seed=7,
                     search_calls=20, proposals=3, max_chars=24000)
    assert len(calls) == 3
    for candidate, feedback in calls:
        assert candidate == bench.components_from_program(base)
        for item in feedback["training_examples"]:
            assert item["Inputs"]["text"].startswith("train request")
            assert set(item["Feedback"]) == {"expected"}
            assert "Generated Outputs" not in item


def test_one_pass_does_not_select_on_validation(experiment):
    source, splits, _ = experiment
    class NoEvaluation:
        def evaluate(self, *args):
            pytest.fail("One-pass arm must not select against validation")
    result, _ = bench.select_arm("B", template_program(source), splits, NoEvaluation(),
        bench.IdentityTeacher(), seed=7, search_calls=20, proposals=3, max_chars=24000)
    assert result.questions == template_program(source).questions


def test_character_limit_and_structure_guard(experiment):
    source, _, _ = experiment
    base = template_program(source)
    teacher = bench.BoundedTeacher(bench.IdentityTeacher(), base, 1)
    parts = bench.components_from_program(base)
    with pytest.raises(CandidateError, match="ceiling"):
        teacher.propose_components(parts, {}, list(parts))
    with pytest.raises(CandidateError):
        bench.program_from_components(base, dict(parts, extra='"injected"'))


def test_provider_failure_is_not_a_rejected_candidate(experiment):
    source, splits, _ = experiment
    class BrokenTeacher:
        def propose_components(self, *args):
            raise BackendError("provider down")
    with pytest.raises(BackendError, match="provider down"):
        bench.select_arm("B", template_program(source), splits, mock_factory(10), BrokenTeacher(),
                         seed=7, search_calls=20, proposals=3, max_chars=24000)


def test_budget_plan_and_duplicate_seeds(experiment):
    _, splits, _ = experiment
    plan = bench.plan(splits, [7, 17, 29], 20, 3, 24000)
    assert plan["programs"] == 10
    assert plan["selection_request_ceiling"] == 160
    assert plan["test_request_ceiling"] == 40
    assert plan["teacher_signature_ceiling"] == 21
    with pytest.raises(ConfigurationError):
        bench.plan(splits, [7, 7], 20, 3, 24000)


def test_paired_bootstrap_direction_and_identity():
    result = bench.paired_accuracy([True] * 8, [False] * 8)
    assert result["delta"] == 1 and result["ci95"] == [1, 1]
    same = bench.paired_accuracy([True, False], [True, False])
    assert same["delta"] == 0 and same["ci95"] == [0, 0]
    with pytest.raises(ValueError):
        bench.paired_accuracy([True], [])


@pytest.fixture
def frozen(experiment, tmp_path, monkeypatch):
    source, splits, manifest = experiment
    original = bench.select_arm
    def fake_search(arm, base, splits, backend, teacher, **kwargs):
        if arm == "C":
            return base, [{"engine": "explicit_test_double"}]
        return original(arm, base, splits, backend, teacher, **kwargs)
    monkeypatch.setattr(bench, "select_arm", fake_search)
    calls = []
    class GuardBackend(MockBackend):
        def evaluate(self, program, state):
            calls.append(state["text"])
            assert not state["text"].startswith("test ")
            return super().evaluate(program, state)
    directory = tmp_path / "selected"
    digest = bench.select_experiment(source, splits, manifest, directory, seeds=[7], search_calls=20,
        proposals=3, max_chars=24000, backend_factory=lambda n: ManagedBackend(GuardBackend(), max_calls=n),
        teacher_factory=lambda n: bench.IdentityTeacher(), max_output_tokens=8192)
    assert calls and all(not text.startswith("test ") for text in calls)
    assert fingerprint(load_document(directory / "frozen.json")) == digest
    return directory, digest


def test_frozen_workflow_and_test_once(experiment, frozen, tmp_path):
    source, splits, manifest = experiment
    directory, digest = frozen
    report = bench.test_experiment(source, splits, manifest, directory, tmp_path / "results",
        reviewed=None, live=False, backend_factory=mock_factory)
    assert report["synthetic"] and report["manifest_sha256"] == digest
    assert set(report["arms"]) == {"A", "B-7", "C-7", "D-7"}
    assert all(arm["accounting"]["requests_attempted"] == 4 for arm in report["arms"].values())
    assert len(report["comparisons"]) == 3
    with pytest.raises(FileExistsError):
        bench.test_experiment(source, splits, manifest, directory, tmp_path / "repeat",
            reviewed=None, live=False, backend_factory=mock_factory)


def test_review_and_mode_checks_before_calls(experiment, frozen, tmp_path):
    source, splits, manifest = experiment
    directory, digest = frozen
    def forbidden(n):
        pytest.fail("Must fail before provider construction")
    with pytest.raises(ConfigurationError, match="Review"):
        bench.test_experiment(source, splits, manifest, directory, tmp_path / "results",
            reviewed=None, live=True, backend_factory=forbidden)
    with pytest.raises(DataError, match="mode"):
        bench.test_experiment(source, splits, manifest, directory, tmp_path / "results",
            reviewed=digest, live=True, backend_factory=forbidden)


def test_all_artifacts_verified_before_test_calls(experiment, frozen, tmp_path):
    source, splits, manifest = experiment
    directory, _ = frozen
    program = bench.Program.load(directory / "D-7.s1.json")
    program.questions["intent"].instructions = "Changed after freeze"
    program.save(directory / "D-7.s1.json")
    with pytest.raises(DataError, match="modified"):
        bench.test_experiment(source, splits, manifest, directory, tmp_path / "results",
            reviewed=None, live=False, backend_factory=lambda n: pytest.fail("Must validate all arms first"))
    assert not (directory / "test-started.json").exists()


@pytest.mark.optional
def test_real_gepa_benchmark_stops_within_both_budgets(experiment, capsys):
    pytest.importorskip("gepa")
    source, splits, _ = experiment
    backend = mock_factory(24)
    teacher = bench.IdentityTeacher()
    program, history = bench.select_arm("C", template_program(source), splits, backend, teacher,
                                       seed=7, search_calls=24, proposals=1, max_chars=24000)
    assert teacher.calls == 1
    assert backend.budget.used <= 24
    assert history[0]["module_selector"] == "all"
    assert program.questions == template_program(source).questions
    assert "Proposed new text" not in capsys.readouterr().out


def test_training_feedback_deduplicated_without_validation(experiment):
    source, splits, _ = experiment
    seen = []
    class Teacher(bench.IdentityTeacher):
        def propose_components(self, candidate, feedback, components):
            seen.append(feedback)
            return super().propose_components(candidate, feedback, components)
    parts = bench.components_from_program(template_program(source))
    trace = [{"Inputs": splits["train"][0].state, "Feedback": {"expected": splits["train"][0].expected}}]
    teacher = bench.BoundedTeacher(Teacher(), template_program(source), 24000)
    teacher.propose_components(parts, {key: trace for key in parts}, list(parts))
    assert seen == [{"training_examples": trace}]


def test_independent_rewrite_is_seeded(experiment):
    source, splits, _ = experiment
    class Teacher(bench.IdentityTeacher):
        def propose_components(self, candidate, feedback, components):
            self.feedback = feedback
            return super().propose_components(candidate, feedback, components)
    teacher = Teacher()
    base = template_program(source)
    bench.independent_rewrite(base, splits["train"], teacher, random.Random(7))
    first = teacher.feedback
    bench.independent_rewrite(base, splits["train"], teacher, random.Random(7))
    assert teacher.feedback == first


def test_teacher_never_receives_undeclared_fields(experiment):
    source, splits, _ = experiment
    for row in splits["train"]:
        row.state["private_note"] = "must not leave the process"
    class Teacher(bench.IdentityTeacher):
        def propose_components(self, candidate, feedback, components):
            assert "private_note" not in json.dumps(feedback)
            return super().propose_components(candidate, feedback, components)
    bench.independent_rewrite(template_program(source), splits["train"], Teacher(), random.Random(7))


def test_duplicate_inputs_fail_before_provider_construction(experiment, tmp_path):
    source, splits, manifest = experiment
    splits["test"][0].state = dict(splits["train"][0].state)
    with pytest.raises(DataError, match="Duplicate model-visible"):
        bench.select_experiment(source, splits, manifest, tmp_path / "frozen", seeds=[7], search_calls=20,
            proposals=3, max_chars=24000, backend_factory=lambda n: pytest.fail("No paid calls on bad data"),
            teacher_factory=lambda n: pytest.fail("No teacher on bad data"), max_output_tokens=8192)


def test_live_flags_checked_before_provider_construction(experiment, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "load_dataset", lambda path: experiment)
    monkeypatch.setattr(bench, "make_backend", lambda *args: pytest.fail("No consent, no provider"))
    for extra in ([], ["--allow-paid"], ["--allow-paid", "--acknowledge-budget-limits"]):
        with pytest.raises(SystemExit):
            bench.main(["select", "--data", str(tmp_path), "--out", str(tmp_path / "run"),
                        "--backend", "typesafe", *extra])
