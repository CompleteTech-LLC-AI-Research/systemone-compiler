import json

import pytest

from typewright import research, research_data as data, research_stats as stats
from typewright.architect import template_program
from typewright.backends import ManagedBackend, MockBackend
from typewright.banking77 import IdentityTeacher
from typewright.data import Example, assert_disjoint, dataset_hash
from typewright.errors import BackendError, ConfigurationError, DataError
from typewright.io import atomic_json, fingerprint, load_document
from typewright.runtime import Runtime


@pytest.fixture
def np():
    return pytest.importorskip("numpy")


def source_for_test(task):
    return data.source_for(task, ["joy", "anger", "neutral"] if task == "goemotions" else ["first", "second", "oos"])


def test_prior_holdout_export_is_test_only_and_preserves_registered_inputs(datasets, tmp_path, np):
    from typewright.holdout_exclusions import export
    from typewright.models import project_state

    protocol_path = tmp_path / "protocol.json"
    research.register(datasets, protocol_path, backend="mock")
    before = {path: path.read_bytes() for path in datasets.rglob("*") if path.is_file()}
    result = export(protocol_path, tmp_path / "exports", data_root=datasets)
    inputs = load_document(tmp_path / "exports/prior-test-input-exclusions.json")
    texts = load_document(tmp_path / "exports/prior-test-text-exclusions.json")
    assert result["test_rows"] == 60
    assert result["tasks"] == 5
    for task in data.TASKS:
        source, splits, _, _ = data.load_dataset(datasets / task)
        test_states = [project_state(source.state, row.state) for row in splits["test"]]
        assert inputs["tasks"][task]["projected_input_sha256s"] == sorted(map(fingerprint, test_states))
        assert inputs["tasks"][task]["declared_input_fields"] == sorted(source.state)
        assert texts["tasks"][task]["normalized_text_sha256s"] == sorted({
            fingerprint(f"{task} test {field} example {i}")
            for field in source.state for i in range(12)})
        train_hashes = {fingerprint(project_state(source.state, row.state)) for row in splits["train"]}
        assert not train_hashes.intersection(inputs["tasks"][task]["projected_input_sha256s"])
    assert {path: path.read_bytes() for path in before} == before
    assert inputs["human_independence_review_required"] is True
    serialized = json.dumps([inputs, texts])
    assert '"expected"' not in serialized
    assert '"predictions"' not in serialized
    assert '"results"' not in serialized
    assert "boolq test passage example" not in serialized
    with pytest.raises(ConfigurationError, match="fresh holdout exclusion"):
        export(protocol_path, tmp_path / "exports")


@pytest.mark.parametrize("tamper", ["protocol", "split", "registered_manifest"])
def test_prior_holdout_export_rejects_changed_evidence_before_output(datasets, tmp_path, tamper, np):
    from typewright.holdout_exclusions import export

    protocol_path = tmp_path / "protocol.json"
    research.register(datasets, protocol_path, backend="mock")
    envelope = load_document(protocol_path)
    if tamper == "protocol":
        envelope["content"]["backend"] = "typesafe"
        atomic_json(protocol_path, envelope)
    elif tamper == "registered_manifest":
        envelope["content"]["datasets"]["boolq"]["manifest_sha256"] = "0" * 64
        envelope["sha256"] = fingerprint(envelope["content"])
        atomic_json(protocol_path, envelope)
    else:
        path = datasets / "boolq/test.jsonl"
        path.write_text(path.read_text().replace("example 0", "modified example 0"), encoding="utf-8")
    with pytest.raises(DataError):
        export(protocol_path, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def gold_for(task, i):
    if task == "boolq":
        return {"answer": bool(i % 2)}
    if task == "sst5":
        return {"sentiment": i % 5}
    if task == "clinc150":
        return {"intent": ["first", "second", "oos"][i % 3]}
    if task == "goemotions":
        return {k: (i+j) % 3 == 0 for j, k in enumerate(["joy", "anger", "neutral"])}
    return {"relation": ["entailment", "contradiction", "neutral"][i % 3]}


@pytest.fixture
def datasets(tmp_path):
    root = tmp_path / "data"
    for task in data.TASKS:
        source = source_for_test(task)
        path = root / task
        path.mkdir(parents=True)
        splits, metadata = {}, {}
        for split in data.SPLITS:
            rows = []
            for i in range(12):
                state = {field: f"{task} {split} {field} example {i}" for field in source.state}
                row = Example(id=f"{task}-{split}-{i}", state=state, expected=gold_for(task, i),
                              group=f"{task}-{split}-group-{i//2}")
                rows.append(row)
                metadata[row.id] = {"upstream_split": split, "stratum": f"R{i%3+1}" if task == "anli" else "all"}
            splits[split] = rows
            (path / f"{split}.jsonl").write_text("".join(r.model_dump_json()+"\n" for r in rows), encoding="utf-8")
        manifest = {"format": "research-dataset/v1", "task": task, "variant": "synthetic-unit-test-fixture",
            "source_sha256": fingerprint(source.model_dump(mode="json")), "metadata_sha256": fingerprint(metadata),
            "audit_sha256": fingerprint({}), "exclusions_sha256": fingerprint({"rows": []}),
            "removed_holdout_rows": 0, "card": data.CARDS[task],
            "splits": {k: {"n": len(v), "sha256": dataset_hash(v)} for k, v in splits.items()}}
        for name, value in [("dataset", manifest), ("metadata", metadata), ("raw-audit", {}),
                            ("exclusions", {"rows": []}), ("usecase", source.model_dump(mode="json"))]:
            atomic_json(path / f"{name}.json", value)
    return root


def evidence(task, rows, source):
    backend = ManagedBackend(MockBackend(), max_calls=len(rows))
    program = template_program(source)
    output = []
    for i, row in enumerate(rows):
        result = Runtime(program, backend).run(row.state)
        output.append(dict(result, id=row.id, group=row.group, gold=row.expected,
                           stratum=f"R{i%3+1}" if task == "anli" else "all"))
    return output


def test_sst_root_only_and_literal_tokens():
    label, sentence = data.tree_sentence("(4 (2 It) (3 (2 is) (4 excellent)))")
    assert label == 4 and sentence == "It is excellent"
    with pytest.raises(DataError):
        data.tree_sentence("__import__('os').system('bad')")
    with pytest.raises(DataError):
        data.tree_sentence("(4 (3 malformed)")


def test_download_hash_is_checked_on_cached_bytes(tmp_path):
    (tmp_path / "sst5").write_bytes(b"bad")
    with pytest.raises(DataError, match="byte verification"):
        data.download("sst5", tmp_path)


def test_derived_cleaning_records_conflicts_and_preserves_test_priority():
    source = data.source_for("boolq")
    def row(uid, question, gold, group):
        return Example(id=uid, state={"question": question, "passage": "evidence"},
                       expected={"answer": gold}, group=group)
    records = [
        ("train", row("a", "same", True, "g"), "all"),
        ("test", row("b", "same", True, "g"), "all"),
        ("train", row("c", "another", False, "g"), "all"),
        ("train", row("d", "conflict", False, "h"), "all"),
        ("test", row("e", "conflict", True, "h"), "all"),
        ("train", row("f", "clean", True, "i"), "all")]
    audit = data.audit_records(records, source)
    assert len(audit["duplicate_inputs"]) == 2
    assert len(audit["conflicting_inputs"]) == 1
    assert audit["cross_holdout_groups"] == ["g", "h"]
    kept, exclusions = data.clean_records(records, source)
    assert {r.id for _, r, _ in kept} == {"b", "f"}
    assert {r["reason"] for r in exclusions} == {"duplicate_exact_input", "conflicting_exact_input", "group_overlaps_holdout"}


def test_strict_prepare_refuses_leakage_without_manifest(tmp_path, monkeypatch):
    source = data.source_for("sst5")
    row = Example(id="1", state={"text": "same"}, expected={"sentiment": 1}, group="same")
    second = row.model_copy(update={"id": "2"})
    monkeypatch.setattr(data, "download", lambda *args: b"fixture")
    monkeypatch.setattr(data, "parse_sources", lambda *args: (source, [("train", row, "all"), ("test", second, "all")]))
    with pytest.raises(DataError, match="leakage"):
        data.prepare("sst5", tmp_path / "strict", tmp_path / "cache")
    assert (tmp_path / "strict/raw-audit.json").exists()
    assert not (tmp_path / "strict/dataset.json").exists()


def test_group_assignment_never_crosses_splits():
    source = data.source_for("sst5")
    records = [("train", Example(id=str(i), state={"text": str(i)}, expected={"sentiment": 1}, group=str(i//2)), "all")
               for i in range(100)]
    records.append(("test", Example(id="holdout", state={"text": "holdout"}, expected={"sentiment": 1}, group="holdout"), "all"))
    splits = data.split_groups(records, 7)
    assert_disjoint(splits, source)
    assert splits == data.split_groups(list(reversed(records)), 7)


@pytest.mark.parametrize("task", data.TASKS)
def test_all_task_metrics_are_finite_and_typed(task, datasets, np):
    source, splits, _, _ = data.load_dataset(datasets / task)
    result = stats.task_metrics(task, evidence(task, splits["test"], source), source)
    assert result["n"] == 12 and result["n_clusters"] == 6
    assert np.isfinite(result["primary"])
    if task == "goemotions":
        assert result["joint_probability"] is None
        assert set(result["per_label"]) == {"joy", "anger", "neutral"}
    elif task == "sst5":
        assert result["mae"] == result["primary"]
    elif task == "clinc150":
        assert result["out_of_scope"]["n"] == 4


def test_binary_rank_ties_and_degenerate_labels(np):
    assert stats.binary_ranking([True, False], [.5, .5]) == {"auroc": .5, "average_precision": .5}
    assert stats.binary_ranking([True, False], [.9, .1]) == {"auroc": 1., "average_precision": 1.}
    assert stats.binary_ranking([True, True], [.9, .1])["auroc"] is None


def test_wilson_retains_uncertainty_for_zero_errors():
    interval = stats.wilson(0, 10)
    assert interval[0] == pytest.approx(0)
    assert interval[1] > .25
    assert stats.wilson(0, 0) is None


def test_score_mae_is_expected_score_not_rounded_level(datasets, np):
    source, splits, _, _ = data.load_dataset(datasets / "sst5")
    rows = evidence("sst5", splits["test"][:1], source)
    rows[0]["gold"]["sentiment"] = 0
    rows[0]["decisions"]["sentiment"].update(value=.6, probabilities={"0": .4, "1": .6, "2": 0., "3": 0., "4": 0.})
    result = stats.task_metrics("sst5", rows, source)
    assert result["mae"] == pytest.approx(.6)
    assert result["accuracy"] == 0


def test_primary_binary_threshold_is_not_calibrated_gate(datasets, np):
    source, splits, _, _ = data.load_dataset(datasets / "boolq")
    rows = evidence("boolq", splits["test"][:1], source)
    rows[0]["gold"]["answer"] = True
    rows[0]["decisions"]["answer"].update(p_true=.6, value=False, review_required=True)
    assert stats.task_metrics("boolq", rows, source)["accuracy"] == 1


def test_clustered_pair_conditions_on_seeds_and_aligns_ids(datasets, np):
    source, splits, _, _ = data.load_dataset(datasets / "boolq")
    rows = evidence("boolq", splits["test"], source)
    result = stats.clustered_pair("boolq", [rows, rows, rows], [rows, rows, rows], source,
                                  bootstrap=100, permutations=100)
    assert result["n_clusters"] == 6 and result["n_examples"] == 12
    assert result["optimization_seeds"] == 3
    assert result["improvement"] == 0 and result["two_sided_cluster_randomization_p"] == 1
    assert result["bootstrap_degenerate"]
    with pytest.raises(DataError, match="identical ordered"):
        stats.clustered_pair("boolq", [rows], [list(reversed(rows))], source, bootstrap=100, permutations=100)


def test_paired_improvement_sign_for_score(datasets, np):
    source, splits, _, _ = data.load_dataset(datasets / "sst5")
    rows = evidence("sst5", splits["test"], source)
    import copy
    better, worse = copy.deepcopy(rows), copy.deepcopy(rows)
    for a, b in zip(better, worse):
        a["decisions"]["sentiment"]["value"] = a["gold"]["sentiment"]
        b["decisions"]["sentiment"]["value"] = b["gold"]["sentiment"]+1
    result = stats.clustered_pair("sst5", [better], [worse], source, bootstrap=100, permutations=100)
    assert result["raw_delta_left_minus_right"] == -1
    assert result["improvement"] == 1


def test_holm_preserves_missing_planned_comparisons():
    corrected = stats.holm({"a": .01, "b": .03, "missing": 1.})
    assert corrected == pytest.approx({"a": .03, "b": .06, "missing": 1.})


def test_cluster_randomization_matches_small_exact_null(datasets, np):
    import copy
    source, splits, _, _ = data.load_dataset(datasets / "boolq")
    left = evidence("boolq", splits["test"][:8], source)
    for row in left:
        row["gold"]["answer"] = True
        row["decisions"]["answer"]["p_true"] = .9
    right = copy.deepcopy(left)
    for row in right:
        row["decisions"]["answer"]["p_true"] = .1
    result = stats.clustered_pair("boolq", [left], [right], source, bootstrap=100, permutations=5000)
    # Four whole groups => exact two-sided extreme probability 2 / 2**4 = .125.
    # Incorrect independent-row swapping would instead give 2 / 2**8 = .0078125.
    assert result["n_clusters"] == 4
    assert result["two_sided_cluster_randomization_p"] == pytest.approx(.125, abs=.025)


def test_multilabel_macro_is_not_accuracy_or_micro_f1(np):
    # Label 1: TP=2, FP=0, FN=0 => F1=1. Label 2: TP=0, FP=1, FN=1 => F1=0.
    value = stats.primary_from_stats("goemotions", np.array([2., 0., 0., 1., 0., 1.]))
    assert value == .5


def test_task_contracts_exclude_annotation_hints():
    assert set(data.source_for("anli").state) == {"premise", "hypothesis"}
    assert set(data.source_for("boolq").state) == {"question", "passage"}
    assert data.source_for("sst5").decisions["sentiment"].type == "score"


def test_live_registration_rejects_synthetic_labels(datasets, tmp_path, np):
    from typewright.errors import ConfigurationError
    with pytest.raises(ConfigurationError, match="upstream human labels"):
        research.register(datasets, tmp_path / "live.json", backend="typesafe")


@pytest.fixture
def protocol(datasets, tmp_path, np):
    pytest.importorskip("matplotlib")
    return research.register(datasets, tmp_path / "protocol.json", seeds=[7], search_calls=60,
                             teacher_calls=1, bootstrap=100, permutations=100)


@pytest.fixture
def selected(protocol, tmp_path, monkeypatch):
    def fake_select(arm, base, splits, backend, teacher, **kwargs):
        assert all(row.id.split("-")[1] == "train" for row in splits["train"])
        return base, [{"engine": "explicit_test_double"}]
    monkeypatch.setattr(research, "select_arm", fake_select)
    directory = tmp_path / "selected"
    research.select(protocol, directory, backend_factory=lambda n: ManagedBackend(MockBackend(), max_calls=n),
                    teacher_factory=lambda n: IdentityTeacher())
    return directory


def test_registration_freezes_all_metrics_and_family(protocol, datasets):
    assert len(protocol["analysis"]["primary_family"]) == 10
    assert protocol["datasets"]["sst5"]["direction"] == "lower"
    assert protocol["datasets"]["goemotions"]["primary"] == "macro_f1"
    assert protocol["backend"] == "mock"
    path = datasets / "boolq/metadata.json"
    altered = load_document(path)
    altered[next(iter(altered))]["stratum"] = "tampered"
    atomic_json(path, altered)
    with pytest.raises(DataError):
        research.validate_protocol(protocol)


def test_full_offline_suite_replay_and_test_once(selected, tmp_path, np):
    results = tmp_path / "results"
    report = research.test_suite(selected, results)
    assert report["synthetic"] and report["full_primary_family_executed"]
    assert len(report["primary_tests"]) == 10
    assert all(test["holm_adjusted_p"] == 1 for test in report["primary_tests"].values())
    replay = research.report(results)
    assert replay == report
    assert (results / "report.md").exists()
    assert (results / "primary-table.json").exists()
    assert (results / "primary-effects.pdf").exists()
    assert "SYNTHETIC" in (results / "primary-effects.svg").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        research.test_suite(selected, tmp_path / "retry")


def test_provider_failure_leaves_evidence_and_no_success_report(selected, tmp_path):
    class Broken(MockBackend):
        def evaluate(self, program, state):
            raise BackendError("fixture outage")
    directory = tmp_path / "failed"
    with pytest.raises(BackendError):
        research.test_suite(selected, directory, backend_factory=lambda n: ManagedBackend(Broken(), max_calls=n))
    execution = research.envelope_read(directory / "execution.json")
    assert execution["status"] == "failed"
    assert execution["failure"]["exception_type"] == "BackendError"
    assert not (directory / "report.json").exists()
    with pytest.raises(DataError, match="failed"):
        research.report(directory)


def test_corrupted_final_artifact_fails_before_any_test(selected, tmp_path):
    path = selected / "anli-D-7.s1.json"
    program = research.Program.load(path)
    program.questions["relation"].instructions = "changed"
    program.save(path)
    with pytest.raises(DataError, match="changed after freeze"):
        research.test_suite(selected, tmp_path / "results",
                            backend_factory=lambda n: pytest.fail("No provider until every artifact is checked"))
    assert not (selected / "test-started.json").exists()


def test_replay_rejects_tampered_evidence(selected, tmp_path, np):
    directory = tmp_path / "results"
    research.test_suite(selected, directory)
    path = directory / "boolq-A.evidence.jsonl"
    path.write_text(path.read_text()+"{}\n")
    with pytest.raises(DataError, match="checksum"):
        research.report(directory)


def test_data_relocation_keeps_registered_hashes(protocol, datasets, tmp_path):
    import shutil
    moved = tmp_path / "relocated"
    shutil.copytree(datasets, moved)
    for value in protocol["datasets"].values():
        value["path"] = "nonexistent-original-location"
    assert set(research.validate_protocol(protocol, moved)) == set(data.TASKS)


def test_selection_has_no_test_requests(protocol, tmp_path, monkeypatch):
    def fake_select(arm, base, splits, backend, teacher, **kwargs):
        # Search may only expose training and validation to evaluate.
        for row in splits["train"]:
            Runtime(base, backend).run(row.state)
        return base, []
    monkeypatch.setattr(research, "select_arm", fake_select)
    calls = []
    class Guard(MockBackend):
        def evaluate(self, program, state):
            assert " test " not in json.dumps(state)
            calls.append(state)
            return super().evaluate(program, state)
    research.select(protocol, tmp_path / "selected", backend_factory=lambda n: ManagedBackend(Guard(), max_calls=n),
                    teacher_factory=lambda n: IdentityTeacher())
    assert calls


def test_live_selection_still_requires_separate_sharing(protocol, tmp_path, monkeypatch):
    protocol["backend"] = "typesafe"
    path = tmp_path / "live.json"
    research.envelope_write(path, protocol)
    monkeypatch.setattr(research, "select", lambda *args: pytest.fail("No sharing consent"))
    with pytest.raises(SystemExit):
        research.main(["select", "--protocol", str(path), "--out", str(tmp_path / "run"),
                       "--allow-paid", "--acknowledge-budget-limits"])


@pytest.mark.optional
@pytest.mark.parametrize("task", data.TASKS)
def test_real_gepa_search_accepts_each_typed_task(task, datasets, np):
    pytest.importorskip("gepa")
    source, splits, _, _ = data.load_dataset(datasets / task)
    backend = ManagedBackend(MockBackend(), max_calls=60)
    program, history = research.select_arm("C", template_program(source), splits, backend, IdentityTeacher(),
        seed=7, search_calls=60, proposals=1, max_chars=24000)
    assert program.decisions == source.decisions
    assert history[0]["engine"] == "gepa"
    assert backend.budget.used <= 60


def test_prior_holdout_export_cli_prints_only_the_message_for_a_chained_data_error(
        tmp_path, monkeypatch, capsys):
    import typewright.holdout_exclusions as module
    secret = "test-only-secret-customer-text"

    def failing(*_args, **_kwargs):
        raise DataError("Dataset row is invalid.") from ValueError(f"input_value={secret}")

    monkeypatch.setattr(module, "export", failing)
    status = module.main(["--protocol", str(tmp_path / "p.json"), "--out", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert status == 2
    assert "Dataset row is invalid." in captured.err
    assert secret not in captured.out + captured.err

    def os_failure(*_args, **_kwargs):
        raise OSError(f"cannot open {secret}")

    monkeypatch.setattr(module, "export", os_failure)
    assert module.main(["--protocol", str(tmp_path / "p.json"), "--out", str(tmp_path / "out")]) == 2
    assert secret not in capsys.readouterr().err


def test_prior_holdout_export_removes_a_partial_output_so_the_run_can_be_retried(
        datasets, tmp_path, monkeypatch, np):
    import typewright.holdout_exclusions as module
    protocol_path = tmp_path / "protocol.json"
    research.register(datasets, protocol_path, backend="mock")
    real = module.atomic_json
    calls = []

    def fail_on_second_write(path, value):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("test-only disk full")
        return real(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_json", fail_on_second_write)
        with pytest.raises(OSError, match="disk full"):
            module.export(protocol_path, tmp_path / "exports", data_root=datasets)
    assert not (tmp_path / "exports").exists()
    result = module.export(protocol_path, tmp_path / "exports", data_root=datasets)
    assert result["tasks"] == 5
    assert (tmp_path / "exports/prior-test-text-exclusions.json").is_file()


@pytest.mark.parametrize("variant", ["exact_extra_field", "case_whitespace"])
def test_export_consumer_roundtrip_rejects_live_overlap(datasets, tmp_path, np, variant):
    from pathlib import Path

    from typewright.hierarchy import HierarchySource
    from typewright.hierarchy_study import register
    from typewright.holdout_exclusions import export, read_holdout_exclusions
    from typewright.models import StateField

    protocol_path = tmp_path / "prior.json"
    research.register(datasets, protocol_path, backend="mock")
    export(protocol_path, tmp_path / "export", data_root=datasets)
    attestation = read_holdout_exclusions(tmp_path / "export/prior-test-input-exclusions.json",
                                          tmp_path / "export/prior-test-text-exclusions.json", "sst5")
    root = Path(__file__).resolve().parents[1] / "examples/hierarchy/support"
    source = HierarchySource.load(root / "source.json")
    # Preserve the hierarchy fixture contract; add fields visible in new roots.
    source.source.state["text"] = StateField(type="string", description="Test-only prior field")
    source.graph.inputs["text"] = source.source.state["text"].model_copy(deep=True)
    source_path, flat_path = tmp_path / "source.json", tmp_path / "flat.json"
    atomic_json(source_path, source.model_dump(mode="json"))
    template_program(source.source).save(flat_path)
    split_paths = {}
    prior = data.load_dataset(datasets / "sst5")[1]["test"][0].state["text"]
    for split in data.SPLITS:
        rows = [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
        for i, row in enumerate(rows):
            row["state"]["text"] = f"unique new {split} {i}"
        if split == "train":
            rows[0]["state"]["text"] = prior if variant == "exact_extra_field" else "  " + prior.upper().replace(" ", "\t  ") + "  "
        split_paths[split] = tmp_path / f"{split}.jsonl"
        split_paths[split].write_text("".join(json.dumps(row) + "\n" for row in rows))
    attestation.update(label_origin="independent_human_reviewed", reviewer="test-only-reviewer",
                       test_independence_evidence="test-only synthetic review fixture")
    output = tmp_path / "live.json"
    with pytest.raises(DataError, match="input overlaps" if variant == "exact_extra_field" else "text overlaps"):
        register(source_path, source_path, flat_path, split_paths, output,
                 study_id="test_only_export_roundtrip", mode="typesafe", selected_method="dspy_gepa",
                 structural_rounds=1, max_metric_calls=16, teacher_max_calls=3,
                 teacher_model="test-only/model", data_attestation=attestation)
    assert not output.exists()


@pytest.mark.parametrize("fault", ["fields_missing", "protocol_mismatch", "normalization", "bad_hash"])
def test_export_consumer_rejects_malformed_pairs_without_inference(tmp_path, fault):
    from typewright.holdout_exclusions import read_holdout_exclusions

    entry = {"n": 1, "manifest_sha256": "a" * 64, "declared_input_fields": ["text"],
             "projected_input_sha256s": ["b" * 64]}
    inputs = {"format": "systemone-prior-flat-test-input-exclusions/v1",
              "source_protocol_sha256": "c" * 64, "tasks": {"sst5": entry}}
    texts = {"format": "systemone-prior-flat-test-text-exclusions/v1",
             "source_protocol_sha256": "c" * 64, "normalization": "casefold_whitespace_v1",
             "tasks": {"sst5": {"n": 1, "manifest_sha256": "a" * 64,
                                  "normalized_text_sha256s": ["d" * 64]}}}
    if fault == "fields_missing":
        entry.pop("declared_input_fields")
    elif fault == "protocol_mismatch":
        texts["source_protocol_sha256"] = "e" * 64
    elif fault == "normalization":
        texts["normalization"] = "none"
    else:
        entry["projected_input_sha256s"] = ["bad"]
    atomic_json(tmp_path / "inputs.json", inputs)
    atomic_json(tmp_path / "texts.json", texts)
    with pytest.raises(DataError, match="Malformed or mismatched"):
        read_holdout_exclusions(tmp_path / "inputs.json", tmp_path / "texts.json", "sst5")
