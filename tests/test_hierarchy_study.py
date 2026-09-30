"""Offline preregistration gates for the independent hierarchy study."""
from __future__ import annotations

from pathlib import Path
import json

import pytest

from s1compiler.backends import ManagedBackend, MockBackend
from s1compiler.errors import ConfigurationError, DataError
from s1compiler.hierarchy_study import (load_frozen, load_protocol, main, register,
                                        report_study, select_and_freeze, test_frozen as run_frozen_test)
import s1compiler.hierarchy_study as study


ROOT = Path(__file__).resolve().parents[1] / "examples" / "hierarchy" / "support"
SPLITS = {name: ROOT / f"{name}.jsonl" for name in ("train", "validation", "calibration", "test")}


def test_mock_registration_pins_sources_splits_and_graph_aware_calls(tmp_path):
    path = tmp_path / "protocol.json"
    protocol = register(ROOT / "source.json", ROOT / "source.json",
                        ROOT / "flat_baseline.s1.json", SPLITS, path,
                        study_id="synthetic_support_h14")
    assert load_protocol(path) == protocol
    assert protocol["status"] == "registered_unexecuted"
    assert protocol["authorization"]["paid_calls_approved"] is False
    assert protocol["analysis"]["confirmatory_family"] == [
        "hierarchy_selected-flat_authored", "hierarchy_selected-hierarchy_authored"]
    assert protocol["budgets"]["graph_leaf_calls_worst_case_per_root"] == 6
    arms = protocol["budgets"]["by_arm"]
    assert arms["flat_authored"]["test_ceiling"] == 6
    assert arms["hierarchy_authored"]["test_ceiling"] == 36
    assert arms["hierarchy_selected"]["selection_calibration_ceiling"] > (
        arms["hierarchy_authored"]["selection_calibration_ceiling"])
    with pytest.raises(ConfigurationError, match="fresh"):
        register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
                 SPLITS, path, study_id="synthetic_support_h14")


def test_registration_rejects_split_leakage_and_source_drift_before_writing(tmp_path):
    bad = dict(SPLITS, test=SPLITS["train"])
    output = tmp_path / "invalid.json"
    with pytest.raises(DataError):
        register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
                 bad, output, study_id="synthetic_support_bad")
    assert not output.exists()

    copied = tmp_path / "train.jsonl"
    copied.write_bytes(SPLITS["train"].read_bytes())
    protocol = register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
                        dict(SPLITS, train=copied), tmp_path / "registered.json",
                        study_id="synthetic_support_drift")
    assert protocol["datasets"]["train"]["n"] == 6
    copied.write_text(copied.read_text() + "\n")
    with pytest.raises(DataError, match="bytes changed"):
        load_protocol(tmp_path / "registered.json")


def test_live_registration_requires_independent_labels_and_billing_caps(tmp_path):
    with pytest.raises(ConfigurationError, match="independent labels"):
        register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
                 SPLITS, tmp_path / "live.json", study_id="live_support_h14", mode="typesafe",
                 selected_method="dspy_gepa", structural_rounds=1, max_metric_calls=8,
                 teacher_max_calls=3, teacher_model="example/model",
                 data_attestation={"label_origin": "synthetic_fixture"})
    assert not (tmp_path / "live.json").exists()


def test_registration_rejects_undersized_gepa_budget_before_protocol(tmp_path):
    output = tmp_path / "undersized.json"
    with pytest.raises(ConfigurationError, match="initial validation"):
        register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
                 SPLITS, output, study_id="undersized_gepa_h14", mode="typesafe",
                 selected_method="dspy_gepa", structural_rounds=1, max_metric_calls=1,
                 teacher_max_calls=3, teacher_model="test-only/model")
    assert not output.exists()


@pytest.mark.parametrize("split", SPLITS)
def test_live_registration_rejects_cross_schema_prior_holdout_text(tmp_path, split):
    # This metadata is a test double; the checked-in labels remain synthetic.
    from s1compiler.io import fingerprint
    first = json.loads(SPLITS[split].read_text(encoding="utf-8").splitlines()[0])
    message = first["state"]["message"]
    prior_state = {"text": message}  # Old study has a different public input field.
    assert fingerprint(prior_state) != fingerprint(first["state"])
    attestation = {"label_origin": "independent_human_reviewed",
                   "test_independence_evidence": "test-only-review-record",
                   "reviewer": "test-only-reviewer",
                   "prior_test_input_sha256s": [fingerprint(prior_state)],
                   "prior_test_text_normalization": "casefold_whitespace_v1",
                   "prior_test_text_sha256s": [fingerprint(" ".join(message.casefold().split()))]}
    output = tmp_path / "unregistered.json"
    with pytest.raises(DataError, match="text overlaps"):
        register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
                 SPLITS, output, study_id="test_only_cross_schema_overlap", mode="typesafe",
                 selected_method="dspy_gepa", structural_rounds=1, max_metric_calls=16,
                 teacher_max_calls=3, teacher_model="test-only/model", data_attestation=attestation)
    assert not output.exists()


@pytest.mark.parametrize("split", SPLITS)
def test_live_registration_rejects_exact_prior_holdout_in_any_split(tmp_path, split):
    # Test-only exclusion metadata; the source labels remain synthetic.
    from s1compiler.io import fingerprint
    row = json.loads(SPLITS[split].read_text(encoding="utf-8").splitlines()[0])
    attestation = {"label_origin": "independent_human_reviewed",
                   "test_independence_evidence": "test-only-review-record",
                   "reviewer": "test-only-reviewer",
                   "prior_test_input_sha256s": [fingerprint(row["state"])],
                   "prior_test_text_normalization": "casefold_whitespace_v1",
                   "prior_test_text_sha256s": [fingerprint("test-only unrelated")]}
    output = tmp_path / "unregistered.json"
    with pytest.raises(DataError, match="input overlaps"):
        register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
                 SPLITS, output, study_id="test_only_exact_overlap", mode="typesafe",
                 selected_method="dspy_gepa", structural_rounds=1, max_metric_calls=16,
                 teacher_max_calls=3, teacher_model="test-only/model", data_attestation=attestation)
    assert not output.exists()


def test_live_registration_requires_normalized_text_exclusions(tmp_path):
    from s1compiler.io import fingerprint
    attestation = {"label_origin": "independent_human_reviewed",
                   "test_independence_evidence": "test-only-review-record",
                   "reviewer": "test-only-reviewer",
                   "prior_test_input_sha256s": [fingerprint({"text": "test-only unrelated"})],
                   "prior_test_text_normalization": "casefold_whitespace_v1"}
    output = tmp_path / "unregistered.json"
    with pytest.raises(ConfigurationError, match="normalized-text holdout exclusions"):
        register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
                 SPLITS, output, study_id="test_only_missing_text_exclusions", mode="typesafe",
                 selected_method="dspy_gepa", structural_rounds=1, max_metric_calls=16,
                 teacher_max_calls=3, teacher_model="test-only/model",
                 data_attestation=attestation)
    assert not output.exists()


def test_normalized_text_fingerprints_cover_nested_declared_values():
    from s1compiler.io import fingerprint
    found = study._normalized_text_fingerprints({
        "messages": ["  REFUND\tRequest  ", {"body": "Nested  TEXT"}],
        "count": 3, "blank": "  "})
    assert found == {fingerprint("refund request"), fingerprint("nested text")}


def test_live_registration_redacts_both_prior_exclusion_lists(tmp_path):
    # Exercise registration only with synthetic test doubles; no provider is built.
    from s1compiler.io import fingerprint
    attestation = {"label_origin": "independent_human_reviewed",
                   "test_independence_evidence": "test-only-review-record",
                   "reviewer": "test-only-reviewer",
                   "prior_test_input_sha256s": [fingerprint({"text": "test-only unrelated"})],
                   "prior_test_text_normalization": "casefold_whitespace_v1",
                   "prior_test_text_sha256s": [fingerprint("test-only unrelated")]}
    output = tmp_path / "registered.json"
    protocol = register(ROOT / "source.json", ROOT / "source.json",
                        ROOT / "flat_baseline.s1.json", SPLITS, output,
                        study_id="test_only_live_registration", mode="typesafe",
                        selected_method="dspy_gepa", structural_rounds=1,
                        max_metric_calls=16, teacher_max_calls=3,
                        teacher_model="test-only/model", data_attestation=attestation,
                        provider_call_price_cap_usd=1, teacher_call_price_cap_usd=1,
                        external_billing_cap_usd=1_000_000)  # Test-only cap; zero calls.
    stored = protocol["data_attestation"]
    assert stored["prior_test_input_sha256s"] is None
    assert stored["prior_test_text_sha256s"] is None
    assert stored["prior_test_exclusion_count"] == 1
    assert stored["prior_test_text_exclusion_count"] == 1
    assert load_protocol(output) == protocol


def test_mock_selection_calibration_freeze_never_runs_test(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    protocol = register(ROOT / "source.json", ROOT / "source.json",
                        ROOT / "flat_baseline.s1.json", SPLITS, protocol_path,
                        study_id="synthetic_support_freeze")
    frozen_dir = tmp_path / "selection"
    frozen = select_and_freeze(protocol_path, frozen_dir)
    assert frozen["status"] == "frozen_unreviewed"
    assert frozen["test_executed"] is False
    assert frozen["live_test_state"] == "unexecuted_requires_exact_frozen_review"
    assert set(frozen["arms"]) == {"flat_authored", "hierarchy_authored", "hierarchy_selected"}
    assert all(arm["accounting"]["requests_attempted"] <=
               protocol["budgets"]["by_arm"][name]["selection_calibration_ceiling"]
               for name, arm in frozen["arms"].items())
    assert frozen["review_gates"]["graph_digests"] == {
        name: arm["content_sha256"] for name, arm in frozen["arms"].items()}
    loaded, programs = load_frozen(frozen_dir)
    assert loaded == frozen
    assert len(programs) == 3
    assert not (frozen_dir / "test-started.json").exists()


def test_validation_selects_predeclared_candidate_without_test_feedback(tmp_path):
    source = json.loads((ROOT / "source.json").read_text())
    help_stage = next(stage for stage in source["graph"]["stages"] if stage["id"] == "technical_help")
    help_stage["program"]["questions"]["resolution"]["criteria"]["general"] = (
        "setup help instructions general response")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(source))
    protocol_path = tmp_path / "protocol.json"
    register(ROOT / "source.json", candidate, ROOT / "flat_baseline.s1.json", SPLITS,
             protocol_path, study_id="synthetic_candidate_h14")

    class NoTest(MockBackend):
        def evaluate(self, program, state):
            assert "test fixture" not in str(state)
            return super().evaluate(program, state)

    frozen = select_and_freeze(protocol_path, tmp_path / "frozen",
                               backend_factory=lambda _arm, _phase, ceiling: ManagedBackend(
                                   NoTest(), max_calls=ceiling))
    assert frozen["arms"]["hierarchy_selected"]["selected_source"] == "candidate"
    assert frozen["arms"]["hierarchy_selected"]["candidate_validation"]["objective"] > (
        frozen["arms"]["hierarchy_authored"]["validation"]["objective"])
    changes = frozen["arms"]["hierarchy_selected"]["semantic_review"][
        "predeclared_candidate_vs_authored"]["changed_child_prompts"]
    assert changes
    assert frozen["test_executed"] is False


def test_offline_test_resume_reuses_complete_records_and_strict_replay(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
             SPLITS, protocol_path, study_id="synthetic_resume_h14",
             bootstrap_replicates=100, randomization_replicates=100)
    frozen_dir = tmp_path / "frozen"
    select_and_freeze(protocol_path, frozen_dir)
    execution = tmp_path / "execution"
    observed = []

    class Counting(MockBackend):
        def evaluate(self, program, state):
            observed.append(program.name)
            return super().evaluate(program, state)

    def factory(_arm, _phase, ceiling):
        return ManagedBackend(Counting(), max_calls=ceiling)
    original = study._execute_record
    count = 0

    def interrupt_after_one(*args, **kwargs):
        nonlocal count
        record = original(*args, **kwargs)
        count += 1
        if count == 1:
            raise RuntimeError("simulated process interruption after durable row")
        return record

    with monkeypatch.context() as patch:
        patch.setattr(study, "_execute_record", interrupt_after_one)
        with pytest.raises(RuntimeError, match="simulated process"):
            run_frozen_test(frozen_dir, execution, backend_factory=factory)
    assert len(list((execution / "records").glob("*/*.json"))) >= 1
    report = run_frozen_test(frozen_dir, execution, resume=True, backend_factory=factory)
    assert report["status"] == "synthetic"
    assert report["n_root"] == 6
    assert report["n_groups"] == 6
    assert set(report["comparisons"]) == {"flat_authored", "hierarchy_authored"}
    assert all("holm_adjusted_p" in comparison for comparison in report["comparisons"].values())
    assert sum(report["arms"][arm]["native_attempts"] for arm in report["arms"]) == len(observed)
    assert report_study(frozen_dir, execution)["status"] == "synthetic"
    assert report["ablations"]["matched_supervision"] is True
    assert report["arms"]["hierarchy_authored"]["metrics"]["routes"]["path_counts"]

    record = next(path for path in (execution / "records").glob("*/*.json")
                  if not path.name.endswith(".started.json"))
    altered = json.loads(record.read_text())
    altered["content"]["synthetic"] = False
    record.write_text(json.dumps(altered))
    with pytest.raises(DataError):
        report_study(frozen_dir, execution)


def test_interrupted_graph_stage_reconciles_reserved_attempt(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
             SPLITS, protocol_path, study_id="synthetic_stage_crash_h14",
             bootstrap_replicates=100, randomization_replicates=100)
    frozen_dir = tmp_path / "frozen"
    select_and_freeze(protocol_path, frozen_dir)
    execution = tmp_path / "execution"
    observed = []

    class Counting(MockBackend):
        def evaluate(self, program, state):
            observed.append(program.name)
            return super().evaluate(program, state)

    def factory(_arm, _phase, ceiling):
        return ManagedBackend(Counting(), max_calls=ceiling)
    from s1compiler.hierarchy_evidence import HierarchyEvidence
    original = HierarchyEvidence._append
    interrupted = False

    def interrupt_once(self, payload):
        nonlocal interrupted
        if payload["kind"] == "stage" and not interrupted:
            interrupted = True
            raise RuntimeError("simulated stage receipt interruption")
        return original(self, payload)

    with monkeypatch.context() as patch:
        patch.setattr(HierarchyEvidence, "_append", interrupt_once)
        with pytest.raises(RuntimeError, match="stage receipt interruption"):
            run_frozen_test(frozen_dir, execution, backend_factory=factory)
    assert interrupted
    report = run_frozen_test(frozen_dir, execution, resume=True, backend_factory=factory)
    assert report["status"] == "synthetic"
    assert sum(report["arms"][arm]["native_attempts"] for arm in report["arms"]) == len(observed)
    assert any(report["arms"][arm]["native_attempts"] > 6
               for arm in ("hierarchy_authored", "hierarchy_selected"))


def test_live_test_requires_exact_review_digest_before_backend(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
             SPLITS, protocol_path, study_id="synthetic_guard_h14")
    frozen_dir = tmp_path / "frozen"
    select_and_freeze(protocol_path, frozen_dir)
    from s1compiler.io import fingerprint
    frozen, _ = load_frozen(frozen_dir)
    frozen["protocol"]["mode"] = "typesafe"
    with pytest.raises(ConfigurationError, match="approval"):
        study._require_live_test_approval(frozen, allow_paid=False,
                                          reviewed_frozen_sha256=None,
                                          semantic_review_approved=False)
    with pytest.raises(ConfigurationError, match="approval"):
        study._require_live_test_approval(frozen, allow_paid=True,
                                          reviewed_frozen_sha256="0" * 64,
                                          semantic_review_approved=True)
    study._require_live_test_approval(frozen, allow_paid=True,
                                      reviewed_frozen_sha256=fingerprint(frozen),
                                      semantic_review_approved=True)


def test_study_cli_no_key_registration_through_offline_report(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    protocol = tmp_path / "protocol.json"
    frozen = tmp_path / "frozen"
    execution = tmp_path / "execution"
    args = ["register", "--study-id", "synthetic_cli_h14", "--source", str(ROOT / "source.json"),
            "--candidate", str(ROOT / "source.json"), "--flat", str(ROOT / "flat_baseline.s1.json"),
            "--out", str(protocol), "--bootstrap-replicates", "100",
            "--randomization-replicates", "100"]
    for name, path in SPLITS.items():
        args.extend((f"--{name}", str(path)))
    assert main(args) == 0
    assert main(["select", "--protocol", str(protocol), "--out", str(frozen)]) == 0
    assert main(["test", "--frozen", str(frozen), "--out", str(execution)]) == 0
    assert main(["report", "--frozen", str(frozen), "--execution", str(execution)]) == 0
    assert json.loads((execution / "report.json").read_text())["synthetic"] is True
