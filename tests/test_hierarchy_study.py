"""Offline preregistration gates for the independent hierarchy study."""
from __future__ import annotations

from pathlib import Path
import json

import pytest

from typewright.backends import ManagedBackend, MockBackend
from typewright.errors import ConfigurationError, DataError
from typewright.hierarchy_study import (load_frozen, load_protocol, main, register,
                                        report_study, select_and_freeze, test_frozen as run_frozen_test)
import typewright.hierarchy_study as study


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
    from typewright.io import fingerprint
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
    from typewright.io import fingerprint
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
    from typewright.io import fingerprint
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
    from typewright.io import fingerprint
    found = study._normalized_text_fingerprints({
        "messages": ["  REFUND\tRequest  ", {"body": "Nested  TEXT"}],
        "count": 3, "blank": "  "})
    assert found == {fingerprint("refund request"), fingerprint("nested text")}


def test_live_registration_redacts_both_prior_exclusion_lists(tmp_path):
    # Exercise registration only with synthetic test doubles; no provider is built.
    from typewright.io import fingerprint
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
    from typewright.hierarchy_evidence import HierarchyEvidence
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
    from typewright.io import fingerprint
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


def _live_test_double_protocol(tmp_path, study_id="test_only_live_manifest"):
    # Registration with synthetic test-double attestation; zero provider or teacher calls.
    from typewright.io import fingerprint
    attestation = {"label_origin": "independent_human_reviewed",
                   "test_independence_evidence": "test-only-review-record",
                   "reviewer": "test-only-reviewer",
                   "prior_test_input_sha256s": [fingerprint({"text": "test-only unrelated"})],
                   "prior_test_text_normalization": "casefold_whitespace_v1",
                   "prior_test_text_sha256s": [fingerprint("test-only unrelated")]}
    path = tmp_path / f"{study_id}.json"
    protocol = register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
                        SPLITS, path, study_id=study_id, mode="typesafe", selected_method="dspy_gepa",
                        structural_rounds=1, max_metric_calls=16, teacher_max_calls=3,
                        teacher_model="test-only/model", data_attestation=attestation,
                        provider_call_price_cap_usd=1, teacher_call_price_cap_usd=1,
                        external_billing_cap_usd=1_000_000)  # Test-only cap; zero calls.
    return path, protocol


def _completed_review(proposal):
    from typewright.io import fingerprint
    return {"format": study.LIVE_MANIFEST_REVIEW_FORMAT, "manifest_sha256": fingerprint(proposal),
            "reviewer": "test-only-independent-reviewer", "reviewed_at": "2026-09-30T00:00:00+00:00",
            "attestations": {name: {"attested": True, "evidence": "test-only record"}
                             for name in study.LIVE_REVIEW_ATTESTATIONS}}


def test_live_manifest_proposal_pins_every_run_parameter_without_calls(tmp_path, monkeypatch):
    from typewright.io import fingerprint
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("S1_TEACHER_API_KEY", raising=False)
    monkeypatch.delenv("S1_TEACHER_API_BASE", raising=False)
    protocol_path, protocol = _live_test_double_protocol(tmp_path)
    proposal = study.propose_live_manifest(protocol_path, tmp_path / "manifest.json")
    assert proposal["status"] == "proposed_unexecuted"
    assert proposal["unspecified_parameters"] == []
    assert proposal["protocol_sha256"] == fingerprint(protocol)
    assert proposal["model"] == json.loads((ROOT / "source.json").read_text())["source"]["model"]
    assert proposal["native_backend"]["identity"] == "typesafe-sdk/0.7.0"
    assert proposal["native_backend"]["sdk_retries"] == 0
    assert proposal["native_backend"]["credential"]["recorded_in_manifest"] is False
    assert proposal["teacher"]["model"] == "test-only/model"
    assert proposal["teacher"]["api_base"] == "provider_default"
    assert proposal["teacher"]["credential"]["recorded_in_manifest"] is False
    assert "S1_TEACHER_API_KEY" not in json.dumps({k: v for k, v in proposal["teacher"].items()
                                                    if k != "credential"})
    assert proposal["teacher"]["provider_request_ceiling"] == (
        protocol["budgets"]["teacher_provider_request_ceiling"])
    assert proposal["selection"]["min_calibration_samples"] == study.LIVE_MIN_CALIBRATION_SAMPLES
    assert proposal["data_sharing_scope"] == "train-only teacher examples/traces"
    assert proposal["data_attestation"]["prior_test_exclusion_count"] == 1
    assert "prior_test_input_sha256s" not in proposal["data_attestation"]
    assert fingerprint(protocol) in proposal["commands"]["select"]
    assert "--reviewed-manifest" in proposal["commands"]["select"]
    assert set(proposal["review_template"]["attestations"]) == set(study.LIVE_REVIEW_ATTESTATIONS)
    again = study.propose_live_manifest(protocol_path, tmp_path / "manifest-again.json")
    assert fingerprint(again) == fingerprint(proposal)
    assert study._envelope_read(tmp_path / "manifest.json") == proposal
    with pytest.raises(ConfigurationError, match="fresh"):
        study.propose_live_manifest(protocol_path, tmp_path / "manifest.json")

    mock_path = tmp_path / "mock.json"
    register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
             SPLITS, mock_path, study_id="synthetic_no_manifest")
    with pytest.raises(ConfigurationError, match="mock studies"):
        study.propose_live_manifest(mock_path, tmp_path / "mock-manifest.json")


def test_live_manifest_review_binds_exact_digest_and_every_attestation(tmp_path):
    from typewright.io import fingerprint
    protocol_path, _ = _live_test_double_protocol(tmp_path)
    proposal = study.propose_live_manifest(protocol_path, tmp_path / "manifest.json")
    manifest = tmp_path / "manifest.json"

    def attempt(review, name):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(review))
        return study.review_live_manifest(manifest, path, tmp_path / f"{name}-reviewed.json")

    with pytest.raises(ConfigurationError, match="different manifest digest"):
        attempt(dict(proposal["review_template"]), "unfilled")
    wrong_digest = _completed_review(proposal) | {"manifest_sha256": "0" * 64}
    with pytest.raises(ConfigurationError, match="different manifest digest"):
        attempt(wrong_digest, "wrong-digest")
    partial = _completed_review(proposal)
    partial["attestations"]["external_billing_cap_enforced"] = {"attested": False, "evidence": "pending"}
    with pytest.raises(ConfigurationError, match="unattested items: external_billing_cap_enforced"):
        attempt(partial, "partial")
    placeholder = _completed_review(proposal)
    placeholder["attestations"]["paid_selection_calls_approved"]["evidence"] = "<record or reason>"
    with pytest.raises(ConfigurationError, match="unattested items: paid_selection_calls_approved"):
        attempt(placeholder, "placeholder")
    extra = _completed_review(proposal)
    extra["attestations"]["deployment_approved"] = {"attested": True, "evidence": "never"}
    with pytest.raises(ConfigurationError, match="exactly the required attestations"):
        attempt(extra, "extra")
    bad_time = _completed_review(proposal) | {"reviewed_at": "yesterday"}
    with pytest.raises(ConfigurationError, match="ISO 8601"):
        attempt(bad_time, "bad-time")
    assert not list(tmp_path.glob("*-reviewed.json"))

    reviewed = attempt(_completed_review(proposal), "complete")
    assert reviewed["status"] == "reviewed_unexecuted"
    assert reviewed["manifest_sha256"] == fingerprint(proposal)
    assert reviewed["selection_executed"] is False and reviewed["test_executed"] is False
    assert reviewed["deployment_approved"] is False
    assert study._envelope_read(tmp_path / "complete-reviewed.json") == reviewed


def test_live_selection_requires_matching_reviewed_manifest_before_any_provider(tmp_path, monkeypatch):
    from typewright.backends import TypeSafeBackend
    from typewright.io import fingerprint
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("S1_TEACHER_API_KEY", raising=False)
    monkeypatch.setenv("S1_TEACHER_API_BASE", "https://test-only.invalid/v1")
    protocol_path, protocol = _live_test_double_protocol(tmp_path)
    approvals = {"allow_paid": True, "share_feedback": True,
                 "approved_protocol_sha256": fingerprint(protocol)}
    with pytest.raises(ConfigurationError, match="reviewed live manifest"):
        select_and_freeze(protocol_path, tmp_path / "no-manifest", teacher=object(), **approvals)
    assert not (tmp_path / "no-manifest").exists()

    proposal = study.propose_live_manifest(protocol_path, tmp_path / "manifest.json")
    (tmp_path / "review.json").write_text(json.dumps(_completed_review(proposal)))
    reviewed_path = tmp_path / "reviewed.json"
    study.review_live_manifest(tmp_path / "manifest.json", tmp_path / "review.json", reviewed_path)
    with pytest.raises(ConfigurationError, match="Reviewed live manifest differs"):
        select_and_freeze(protocol_path, tmp_path / "unreviewed", teacher=object(),
                          reviewed_manifest=tmp_path / "manifest.json", **approvals)
    with monkeypatch.context() as patch:
        patch.setattr(TypeSafeBackend, "identity", "typesafe-sdk/0.0.0-test-only")
        with pytest.raises(ConfigurationError, match="differs from the current protocol, software"):
            select_and_freeze(protocol_path, tmp_path / "drifted", teacher=object(),
                              reviewed_manifest=reviewed_path, **approvals)
    assert not (tmp_path / "drifted").exists()
    assert study._envelope_read(tmp_path / "manifest.json")["teacher"]["api_base"] == (
        "https://test-only.invalid/v1")
    with monkeypatch.context() as patch:
        patch.setenv("S1_TEACHER_API_BASE", "https://other-endpoint.invalid/v1")
        with pytest.raises(ConfigurationError, match="teacher endpoint"):
            select_and_freeze(protocol_path, tmp_path / "endpoint-drift", teacher=object(),
                              reviewed_manifest=reviewed_path, **approvals)
    assert not (tmp_path / "endpoint-drift").exists()

    # With a matching reviewed manifest the next stop is the local credential check, never a call.
    with pytest.raises(ConfigurationError, match="TYPESAFE_API_KEY"):
        select_and_freeze(protocol_path, tmp_path / "credential", teacher=object(),
                          reviewed_manifest=reviewed_path, **approvals)
    failure = json.loads((tmp_path / "credential" / "selection-failure.json").read_text())
    assert failure["completed_arms"] == [] and failure["synthetic"] is False

    mock_path = tmp_path / "mock.json"
    register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
             SPLITS, mock_path, study_id="synthetic_mock_manifest")
    with pytest.raises(ConfigurationError, match="Mock studies"):
        select_and_freeze(mock_path, tmp_path / "mock-frozen", reviewed_manifest=reviewed_path)
    assert not (tmp_path / "mock-frozen").exists()


def test_post_freeze_live_manifest_carries_reviewed_digest_and_unapproved_test(tmp_path):
    from typewright.io import fingerprint
    protocol_path, protocol = _live_test_double_protocol(tmp_path)
    proposal = study.propose_live_manifest(protocol_path, tmp_path / "manifest.json")
    (tmp_path / "review.json").write_text(json.dumps(_completed_review(proposal)))
    reviewed = study.review_live_manifest(tmp_path / "manifest.json", tmp_path / "review.json",
                                          tmp_path / "reviewed.json")
    arms = {arm: {"content_sha256": fingerprint(arm), "provenance_sha256": fingerprint(arm + "p")}
            for arm in study.ARMS}
    frozen = {"protocol_sha256": fingerprint(protocol), "arms": arms,
              "review_gates": {"semantic_review": {arm: {"status": "test-only"} for arm in arms},
                               "data_sharing_scope": "train-only teacher examples/traces"}}
    manifest = study._live_manifest_after_freeze(frozen, reviewed)
    assert manifest["reviewed_manifest_sha256"] == fingerprint(reviewed)
    assert manifest["frozen_sha256"] == fingerprint(frozen)
    assert manifest["frozen_artifacts"] == arms
    assert manifest["backend"] == "typesafe-sdk/0.7.0"
    assert manifest["approval"] == {"paid_selection_calls": True, "teacher_example_sharing": True,
                                    "external_billing_cap_verified": True,
                                    "paid_test_calls": False, "semantic_review": False}
    assert manifest["test_executed"] is False and manifest["deployment_approved"] is False


def test_study_cli_manifest_and_review_run_without_keys(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("S1_TEACHER_API_KEY", raising=False)
    protocol_path, _ = _live_test_double_protocol(tmp_path)
    manifest = tmp_path / "manifest.json"
    assert main(["manifest", "--protocol", str(protocol_path), "--out", str(manifest)]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["status"] == "proposed_unexecuted"
    assert printed["unspecified_parameters"] == []
    proposal = study._envelope_read(manifest)
    assert printed["manifest_sha256"] == study.fingerprint(proposal)
    template = tmp_path / "template-review.json"
    template.write_text(json.dumps(proposal["review_template"]))
    assert main(["review", "--manifest", str(manifest), "--review", str(template),
                 "--out", str(tmp_path / "rejected.json")]) == 2
    assert not (tmp_path / "rejected.json").exists()
    completed = tmp_path / "review.json"
    completed.write_text(json.dumps(_completed_review(proposal)))
    assert main(["review", "--manifest", str(manifest), "--review", str(completed),
                 "--out", str(tmp_path / "reviewed.json")]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["status"] == "reviewed_unexecuted"
    assert printed["selection_executed"] is False and printed["test_executed"] is False
    assert main(["select", "--protocol", str(protocol_path), "--out", str(tmp_path / "frozen"),
                 "--allow-paid", "--share-feedback", "--approved-protocol-sha256", "0" * 64,
                 "--reviewed-manifest", str(tmp_path / "reviewed.json")]) == 2
    assert not (tmp_path / "frozen").exists()


@pytest.mark.parametrize("value", ["https://user:secret@test-only.invalid/v1", "https://test-only.invalid/v1?key=secret",
                                   "https://test-only.invalid/v1#secret", "ftp://test-only.invalid/v1",
                                   "https:///v1", "https://test-only.invalid:notaport/v1"])
def test_teacher_endpoint_refuses_urls_that_could_carry_credentials(monkeypatch, value):
    monkeypatch.setenv("S1_TEACHER_API_BASE", value)
    with pytest.raises(ConfigurationError, match="S1_TEACHER_API_BASE"):
        study._teacher_endpoint()


def test_teacher_endpoint_records_plain_urls_and_defaults(monkeypatch):
    monkeypatch.delenv("S1_TEACHER_API_BASE", raising=False)
    assert study._teacher_endpoint() == "provider_default"
    monkeypatch.setenv("S1_TEACHER_API_BASE", "https://test-only.invalid:8443/v1")
    assert study._teacher_endpoint() == "https://test-only.invalid:8443/v1"


def test_manifest_proposal_never_records_a_credentialed_endpoint(tmp_path, monkeypatch):
    protocol_path, _ = _live_test_double_protocol(tmp_path)
    monkeypatch.setenv("S1_TEACHER_API_BASE", "https://user:test-only-secret@test-only.invalid/v1")
    output = tmp_path / "credentialed-manifest.json"
    with pytest.raises(ConfigurationError, match="userinfo"):
        study.propose_live_manifest(protocol_path, output)
    assert not output.exists()


class _FailingLive:
    """Live-looking backend whose first request fails; counts requests it received."""
    identity = "typesafe-sdk/0.7.0"
    synthetic = False

    def __init__(self):
        self.requests = 0

    def evaluate(self, program, state):
        from typewright.errors import BackendError
        self.requests += 1
        raise BackendError("Test-only provider failure")

    def close(self):
        pass


def _live_selection_setup(tmp_path, monkeypatch):
    from typewright.io import fingerprint
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("S1_TEACHER_API_KEY", raising=False)
    monkeypatch.delenv("S1_TEACHER_API_BASE", raising=False)
    protocol_path, protocol = _live_test_double_protocol(tmp_path)
    proposal = study.propose_live_manifest(protocol_path, tmp_path / "manifest.json")
    (tmp_path / "review.json").write_text(json.dumps(_completed_review(proposal)))
    reviewed = tmp_path / "reviewed.json"
    study.review_live_manifest(tmp_path / "manifest.json", tmp_path / "review.json", reviewed)
    approvals = {"allow_paid": True, "share_feedback": True,
                 "approved_protocol_sha256": fingerprint(protocol), "reviewed_manifest": reviewed}
    return protocol_path, approvals


def test_failed_live_selection_records_accounting_and_blocks_silent_reselection(tmp_path, monkeypatch):
    from typewright.io import fingerprint
    protocol_path, approvals = _live_selection_setup(tmp_path, monkeypatch)
    live = _FailingLive()

    def factory(_arm, _phase, ceiling):
        return ManagedBackend(live, max_calls=ceiling)

    with pytest.raises(Exception, match="Test-only provider failure"):
        select_and_freeze(protocol_path, tmp_path / "first", backend_factory=factory,
                          teacher=object(), **approvals)
    assert live.requests == 1
    failure = json.loads((tmp_path / "first" / "selection-failure.json").read_text())
    assert failure["arm"] == "flat_authored"
    assert failure["accounting"]["flat_authored"]["requests_attempted"] == 1
    ledger_path = study._selection_ledger_path(protocol_path)
    ledger = study._envelope_read(ledger_path)
    assert [attempt["status"] for attempt in ledger["attempts"]] == ["failed"]
    assert ledger["attempts"][0]["accounting"]["flat_authored"]["requests_attempted"] == 1

    # A second attempt is refused before any provider is built and no directory is created.
    live.requests = 0
    with pytest.raises(ConfigurationError, match="already has 1 live selection attempt"):
        select_and_freeze(protocol_path, tmp_path / "second", backend_factory=factory,
                          teacher=object(), **approvals)
    with pytest.raises(ConfigurationError, match="already has 1 live selection attempt"):
        select_and_freeze(protocol_path, tmp_path / "second", backend_factory=factory, teacher=object(),
                          acknowledge_prior_selection_sha256="0" * 64, **approvals)
    assert live.requests == 0 and not (tmp_path / "second").exists()

    # An explicit, digest-bound acknowledgement allows it and appends to the same ledger.
    with pytest.raises(Exception, match="Test-only provider failure"):
        select_and_freeze(protocol_path, tmp_path / "third", backend_factory=factory, teacher=object(),
                          acknowledge_prior_selection_sha256=fingerprint(ledger), **approvals)
    assert live.requests == 1
    assert len(study._envelope_read(ledger_path)["attempts"]) == 2


def test_selection_ledger_is_bound_to_its_protocol(tmp_path, monkeypatch):
    protocol_path, approvals = _live_selection_setup(tmp_path, monkeypatch)
    other_path, _ = _live_test_double_protocol(tmp_path, study_id="test_only_other_protocol")
    study._envelope_write(study._selection_ledger_path(protocol_path), {
        "format": study.SELECTION_LEDGER_FORMAT, "protocol_sha256": "0" * 64, "attempts": []})
    with pytest.raises(DataError, match="differs from this protocol"):
        select_and_freeze(protocol_path, tmp_path / "out", teacher=object(), **approvals)
    assert other_path.exists() and not (tmp_path / "out").exists()


def test_mock_selection_needs_no_ledger_and_can_repeat(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
             SPLITS, protocol_path, study_id="synthetic_repeat_h14")
    select_and_freeze(protocol_path, tmp_path / "first")
    select_and_freeze(protocol_path, tmp_path / "second")
    assert not study._selection_ledger_path(protocol_path).exists()


def _rows(spec):
    """spec: list of (id, group). Returns row stand-ins with only the fields the statistics read."""
    from types import SimpleNamespace
    return [SimpleNamespace(id=row_id, group=group) for row_id, group in spec]


def test_quantile_interpolates_linearly_and_handles_one_value():
    assert study._quantile([4.0], 0.5) == 4.0
    assert study._quantile([0.0, 10.0], 0.25) == 2.5
    assert study._quantile([3.0, 1.0, 2.0, 4.0], 0.0) == 1.0
    assert study._quantile([3.0, 1.0, 2.0, 4.0], 1.0) == 4.0
    assert study._quantile([3.0, 1.0, 2.0, 4.0], 0.5) == 2.5


def test_paired_clusters_weights_by_root_not_by_group():
    # Group A has three roots each +1; group B has one root at -1. Root mean is 0.5, group mean is 0.
    rows = _rows([("a1", "A"), ("a2", "A"), ("a3", "A"), ("b1", "B")])
    left = {"a1": 1.0, "a2": 1.0, "a3": 1.0, "b1": 0.0}
    right = {"a1": 0.0, "a2": 0.0, "a3": 0.0, "b1": 1.0}
    result = study._paired_clusters(rows, left, right, seed=1, bootstrap=200, permutations=200)
    assert result["mean_root_utility_delta"] == pytest.approx(0.5)
    assert result["n_groups"] == 2 and result["n_roots"] == 4
    lower, upper = result["cluster_bootstrap_ci_95"]
    assert -1.0 <= lower <= upper <= 1.0


def test_paired_clusters_zero_difference_has_p_one_and_degenerate_interval():
    rows = _rows([(f"r{i}", f"g{i}") for i in range(6)])
    same = {row.id: 0.5 for row in rows}
    result = study._paired_clusters(rows, same, dict(same), seed=3, bootstrap=100, permutations=500)
    assert result["mean_root_utility_delta"] == 0.0
    assert result["two_sided_group_sign_flip_p"] == 1.0
    assert result["cluster_bootstrap_ci_95"] == [0.0, 0.0]


def test_paired_clusters_sign_flip_p_matches_the_exact_value_for_a_unanimous_effect():
    # Eight groups all favour the left arm, so the exact two-sided p is 2 / 2**8 = 0.0078125.
    rows = _rows([(f"r{i}", f"g{i}") for i in range(8)])
    left = {row.id: 1.0 for row in rows}
    right = {row.id: 0.0 for row in rows}
    result = study._paired_clusters(rows, left, right, seed=11, bootstrap=100, permutations=20000)
    assert result["mean_root_utility_delta"] == 1.0
    assert result["two_sided_group_sign_flip_p"] == pytest.approx(2 / 256, abs=0.004)
    assert result["cluster_bootstrap_ci_95"] == [1.0, 1.0]


def test_paired_clusters_is_deterministic_and_bonferroni_interval_is_wider():
    rows = _rows([(f"r{i}", f"g{i % 5}") for i in range(20)])
    left = {row.id: float(index % 3) for index, row in enumerate(rows)}
    right = {row.id: float((index * 7) % 4) for index, row in enumerate(rows)}
    first = study._paired_clusters(rows, left, right, seed=5, bootstrap=400, permutations=400)
    again = study._paired_clusters(rows, left, right, seed=5, bootstrap=400, permutations=400)
    other = study._paired_clusters(rows, left, right, seed=6, bootstrap=400, permutations=400)
    assert first == again
    assert first["cluster_bootstrap_ci_95"] != other["cluster_bootstrap_ci_95"] or (
        first["two_sided_group_sign_flip_p"] != other["two_sided_group_sign_flip_p"])
    lo, hi = first["cluster_bootstrap_ci_95"]
    blo, bhi = first["cluster_bootstrap_bonferroni_ci_95"]
    assert blo <= lo <= hi <= bhi
    assert 0.0 < first["two_sided_group_sign_flip_p"] <= 1.0


def test_paired_clusters_refuses_a_single_group():
    rows = _rows([("a", "G"), ("b", "G")])
    values = {"a": 1.0, "b": 0.0}
    with pytest.raises(DataError, match="two independent root groups"):
        study._paired_clusters(rows, values, {"a": 0.0, "b": 0.0}, seed=1, bootstrap=100, permutations=100)


def test_roots_without_a_group_cluster_by_their_own_id():
    rows = _rows([("a", None), ("b", None), ("c", None)])
    result = study._paired_clusters(rows, {"a": 1.0, "b": 1.0, "c": 1.0}, {"a": 0.0, "b": 0.0, "c": 0.0},
                                    seed=2, bootstrap=100, permutations=100)
    assert result["n_groups"] == 3 and result["n_roots"] == 3


def test_holm_adjusts_the_two_primary_comparisons_monotonically():
    from typewright.research_stats import holm
    assert holm({"a": 0.01, "b": 0.04}) == pytest.approx({"a": 0.02, "b": 0.04})
    # The smaller raw p cannot end up with a larger adjusted p than the larger raw p.
    adjusted = holm({"a": 0.03, "b": 0.02})
    assert adjusted["b"] == pytest.approx(0.04) and adjusted["a"] == pytest.approx(0.04)
    assert holm({"a": 0.9, "b": 0.8}) == pytest.approx({"a": 1.0, "b": 1.0})
    with pytest.raises(ValueError):
        holm({"a": 1.5})


def _frozen_mock_study(tmp_path, study_id):
    protocol_path = tmp_path / "protocol.json"
    register(ROOT / "source.json", ROOT / "source.json", ROOT / "flat_baseline.s1.json",
             SPLITS, protocol_path, study_id=study_id, bootstrap_replicates=100,
             randomization_replicates=100)
    frozen_dir = tmp_path / "frozen"
    select_and_freeze(protocol_path, frozen_dir)
    return frozen_dir


def _counting_factory():
    return lambda _arm, _phase, ceiling: ManagedBackend(MockBackend(), max_calls=ceiling)


class _CrashingRuntime:
    """Stands in for Runtime on the flat arm; optionally dispatches before crashing."""
    dispatch = False

    def __init__(self, program, backend, **_kwargs):
        self.program, self.backend = program, backend

    def run(self, state):
        if type(self).dispatch:
            self.backend.evaluate(self.program, state)  # reserves budget, then the process "dies"
        raise RuntimeError("simulated crash")


def _first_flat_record(execution):
    start = study._envelope_read(execution / "test-started.json")
    item = next(entry for entry in start["schedule"] if entry["arm"] == "flat_authored")
    return item, execution / "records" / "flat_authored" / f"{item['key']}.json"


def test_flat_marker_is_removed_when_no_request_could_have_been_sent(tmp_path, monkeypatch):
    frozen_dir = _frozen_mock_study(tmp_path, "synthetic_marker_clean")
    execution = tmp_path / "execution"
    _CrashingRuntime.dispatch = False
    with monkeypatch.context() as patch:
        patch.setattr(study, "Runtime", _CrashingRuntime)
        with pytest.raises(RuntimeError, match="simulated crash"):
            run_frozen_test(frozen_dir, execution, backend_factory=_counting_factory())
    _, record = _first_flat_record(execution)
    assert not record.with_suffix(".started.json").exists()
    report = run_frozen_test(frozen_dir, execution, resume=True, backend_factory=_counting_factory())
    assert report["status"] == "synthetic" and report["reconciled_attempts"] == []


def test_dispatched_flat_marker_blocks_resume_until_an_explicit_recorded_reconcile(tmp_path, monkeypatch):
    frozen_dir = _frozen_mock_study(tmp_path, "synthetic_marker_reconcile")
    execution = tmp_path / "execution"
    _CrashingRuntime.dispatch = True
    with monkeypatch.context() as patch:
        patch.setattr(study, "Runtime", _CrashingRuntime)
        with pytest.raises(RuntimeError, match="simulated crash"):
            run_frozen_test(frozen_dir, execution, backend_factory=_counting_factory())
    item, record = _first_flat_record(execution)
    marker = record.with_suffix(".started.json")
    assert marker.exists()
    with pytest.raises(DataError, match="automatic retry is unsafe"):
        run_frozen_test(frozen_dir, execution, resume=True, backend_factory=_counting_factory())

    with pytest.raises(ConfigurationError, match="reviewer and the evidence"):
        study.reconcile_flat_attempt(frozen_dir, execution, root_id=item["id"], reviewer=" ", attestation="x")
    with pytest.raises(ConfigurationError, match="reviewer and the evidence"):
        study.reconcile_flat_attempt(frozen_dir, execution, root_id=item["id"], reviewer="r", attestation="")
    with pytest.raises(DataError, match="Unknown held-out root"):
        study.reconcile_flat_attempt(frozen_dir, execution, root_id="nope", reviewer="r", attestation="x")
    assert marker.exists()

    claim = study.reconcile_flat_attempt(frozen_dir, execution, root_id=item["id"],
                                         reviewer="test-only-reviewer",
                                         attestation="test-only: process killed before dispatch")
    assert claim["reviewer"] == "test-only-reviewer" and claim["sequence"] == 1
    assert not marker.exists()
    stored = sorted(record.parent.glob(f"{item['key']}.reconciled-*.json"))
    assert len(stored) == 1 and study._envelope_read(stored[0])["claim"].startswith("no provider request")
    with pytest.raises(DataError, match="no in-flight marker"):
        study.reconcile_flat_attempt(frozen_dir, execution, root_id=item["id"], reviewer="r", attestation="x")

    report = run_frozen_test(frozen_dir, execution, resume=True, backend_factory=_counting_factory())
    assert report["status"] == "synthetic"
    assert [entry["reviewer"] for entry in report["reconciled_attempts"]] == ["test-only-reviewer"]
    with pytest.raises(DataError, match="already has a result"):
        study.reconcile_flat_attempt(frozen_dir, execution, root_id=item["id"], reviewer="r", attestation="x")


def test_reconcile_cli_refuses_without_evidence(tmp_path, capsys):
    frozen_dir = _frozen_mock_study(tmp_path, "synthetic_marker_cli")
    execution = tmp_path / "execution"
    run_frozen_test(frozen_dir, execution, backend_factory=_counting_factory())
    item, record = _first_flat_record(execution)
    assert main(["reconcile", "--frozen", str(frozen_dir), "--execution", str(execution),
                 "--root-id", item["id"], "--reviewer", "test-only", "--evidence", "x"]) == 2
    assert "already has a result" in capsys.readouterr().err


def test_live_mode_selection_freeze_and_manifest_with_explicit_offline_doubles(tmp_path, monkeypatch):
    """Exercise live-mode software gates; all data, reviews, and responses are test doubles."""
    pytest.importorskip("gepa")
    from typewright.io import fingerprint
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    protocol_path, approvals = _live_selection_setup(tmp_path, monkeypatch)
    dispatches, teacher_payloads = [], []
    test_states = [json.loads(line)["state"] for line in SPLITS["test"].read_text().splitlines()]

    class ForbiddenRealProvider:
        identity = "typesafe-sdk/0.7.0"

        def __init__(self, *, allow_paid=False, timeout=60.0):
            raise AssertionError("Real provider construction is forbidden in this offline test")

    monkeypatch.setattr(study, "TypeSafeBackend", ForbiddenRealProvider)

    class NativeInterfaceDouble(MockBackend):
        # False exists solely to enter live-mode gates; these lexical responses are not Jev evidence.
        synthetic = False
        identity = "typesafe-sdk/0.7.0"

        def evaluate(self, program, state):
            assert state not in test_states
            dispatches.append(program.name)
            response = super().evaluate(program, state)
            response.synthetic = False  # Test-only live-interface simulation, never measured evidence.
            return response

    def factory(arm, phase, ceiling):
        assert phase == "selection"
        return ManagedBackend(NativeInterfaceDouble(), max_calls=ceiling)

    class TeacherDouble:
        model = "test-only/model"

        def propose_hierarchy(self, source, current, feedback):
            teacher_payloads.append(feedback)
            assert len(teacher_payloads) <= 3
            assert source.source == current.source
            return current

        def propose_components(self, candidate, reflective_dataset, components):
            teacher_payloads.append(reflective_dataset)
            assert len(teacher_payloads) <= 3
            return {name: candidate[name] for name in components}

        def accounting(self):
            return {"model": self.model, "signature_calls": len(teacher_payloads),
                    "provider_requests_attempted": 0}

    directory = tmp_path / "test-double-frozen"
    frozen = select_and_freeze(protocol_path, directory, backend_factory=factory,
                               teacher=TeacherDouble(), **approvals)
    verified, artifacts = load_frozen(directory)
    assert verified == frozen
    assert set(artifacts) == set(study.ARMS)
    assert frozen["status"] == "frozen_unreviewed"
    assert frozen["test_executed"] is False
    assert frozen["deployment_approved"] is False
    assert teacher_payloads
    train_ids = {json.loads(line)["id"] for line in SPLITS["train"].read_text().splitlines()}
    assert set(teacher_payloads[0]["root_quality"]) == train_ids
    for split in ("validation", "calibration", "test"):
        for line in SPLITS[split].read_text().splitlines():
            assert json.loads(line)["state"]["message"] not in json.dumps(teacher_payloads)
    reviewed = study._envelope_read(approvals["reviewed_manifest"])
    manifest = study._envelope_read(directory / "live-manifest.json")
    assert manifest["frozen_sha256"] == fingerprint(frozen)
    assert manifest["reviewed_manifest_sha256"] == fingerprint(reviewed)
    assert manifest["frozen_artifacts"] == {
        arm: {"content_sha256": entry["content_sha256"], "provenance_sha256": entry["provenance_sha256"]}
        for arm, entry in frozen["arms"].items()}
    assert manifest["semantic_review"] == frozen["review_gates"]["semantic_review"]
    assert manifest["test_executed"] is False
    assert manifest["approval"]["paid_test_calls"] is False
    assert manifest["approval"]["semantic_review"] is False
    ledger = study._envelope_read(study._selection_ledger_path(protocol_path))
    assert len(ledger["attempts"]) == 1
    attempt = ledger["attempts"][0]
    assert attempt["status"] == "frozen"
    assert sum(entry["requests_attempted"] for entry in attempt["accounting"].values()) == len(dispatches)
    assert attempt["teacher_accounting"]["provider_requests_attempted"] == 0
    count = len(dispatches)
    with pytest.raises(ConfigurationError):
        run_frozen_test(directory, tmp_path / "unapproved-test", backend_factory=factory)
    assert len(dispatches) == count
    assert not (tmp_path / "unapproved-test").exists()


def test_reconcile_rejects_a_tampered_schedule_key_before_building_any_path(tmp_path):
    frozen_dir = _frozen_mock_study(tmp_path, "synthetic_marker_tamper")
    execution = tmp_path / "execution"
    run_frozen_test(frozen_dir, execution, backend_factory=_counting_factory())
    item, _ = _first_flat_record(execution)
    started_path = execution / "test-started.json"
    started = study._envelope_read(started_path)
    for entry in started["schedule"]:
        if entry["arm"] == "flat_authored" and entry["id"] == item["id"]:
            entry["key"] = "../../outside"
    study._envelope_write(started_path, started)
    with pytest.raises(DataError, match="record key is not the expected digest"):
        study.reconcile_flat_attempt(frozen_dir, execution, root_id=item["id"], reviewer="r", attestation="x")
    assert not (tmp_path / "outside").exists()
