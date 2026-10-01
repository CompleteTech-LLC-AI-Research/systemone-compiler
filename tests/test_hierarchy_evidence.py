"""Durable graph replay and crash recovery, using only synthetic backends."""
import copy
import json
from pathlib import Path

import pytest

from s1compiler.backends import ManagedBackend, MockBackend
from s1compiler.errors import BackendError, ConfigurationError, DataError
from s1compiler.hierarchy import HierarchySource, RootRef, lower_hierarchy
from s1compiler.hierarchy_evidence import HierarchyEvidence, _input_references
from s1compiler.hierarchy_runtime import GraphRetryPolicy, HierarchyRuntime


FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "hierarchy_contract"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))


class Recording(MockBackend):
    def __init__(self):
        self.calls = []

    def evaluate(self, program, state):
        self.calls.append(program.name)
        return super().evaluate(program, state)


def runtime(name="chain", backend=None):
    artifact = lower_hierarchy(HierarchySource.load(FIXTURES / f"{name}.json"))
    return HierarchyRuntime(artifact, ManagedBackend(backend or Recording()))


@pytest.mark.parametrize("name", ["chain", "conditional", "diamond", "nested"])
def test_offline_replay_recomputes_routes_with_zero_backend_calls(tmp_path, name):
    state = CASES[f"{name}.json"]["state"]
    directory = tmp_path / "evidence"
    first_backend = Recording()
    first = runtime(name, first_backend).run(state, evidence_dir=directory)
    assert first["status"] == "completed"
    assert first["evidence"]["mode"] == "retained_raw_response"
    assert len(first_backend.calls) == len(first["executed"])
    second_backend = Recording()
    replay = runtime(name, second_backend).run(state, evidence_dir=directory, evidence_mode="replay")
    assert replay == first
    assert second_backend.calls == []
    before = (directory / "events.jsonl").read_bytes()
    resumed_backend = Recording()
    resumed = runtime(name, resumed_backend).run(state, evidence_dir=directory, evidence_mode="resume")
    assert resumed == first
    assert resumed_backend.calls == []
    assert (directory / "events.jsonl").read_bytes() == before
    if name == "conditional":
        assert replay["stages"]["technical"]["status"] == "skipped"
        assert replay["decisions"]["resolution"]["distribution_scope"] == "branch_conditional"


def test_rounded_score_raw_response_survives_strict_replay(tmp_path):
    class Rounded(Recording):
        def evaluate(self, program, state):
            response = super().evaluate(program, state)
            for answer in response.answers.values():
                if answer["type"] == "score":
                    answer["probabilities"] = {"0": 0.33, "1": 0.33, "2": 0.33}
                    answer["score"] = 1.0
            return response

    state = CASES["chain.json"]["state"]
    directory = tmp_path / "rounded"
    first = runtime(backend=Rounded()).run(state, evidence_dir=directory)
    assert first["status"] == "completed"
    second = runtime(backend=Recording()).run(state, evidence_dir=directory, evidence_mode="replay")
    assert second == first
    events = (directory / "events.jsonl").read_text(encoding="utf-8")
    assert '"0":0.33' in events


def test_crash_before_stage_append_charges_lost_response_and_resume_only_missing(tmp_path, monkeypatch):
    directory = tmp_path / "crash"
    state = CASES["chain.json"]["state"]
    first_backend = Recording()
    original = HierarchyEvidence._append

    def fail_before_append(self, payload):
        if payload["kind"] == "stage":
            raise RuntimeError("simulated process death before durable response append")
        return original(self, payload)

    with monkeypatch.context() as context:
        context.setattr(HierarchyEvidence, "_append", fail_before_append)
        with pytest.raises(RuntimeError, match="process death"):
            runtime(backend=first_backend).run(state, evidence_dir=directory)
    assert first_backend.calls == ["signal"]
    checkpoint = json.loads((directory / "attempts.json").read_text(encoding="utf-8"))["ledger"]
    assert checkpoint["used"] == 1
    assert checkpoint["receipts"][0]["status"] == "validated"
    resumed_backend = Recording()
    result = runtime(backend=resumed_backend).run(state, evidence_dir=directory, evidence_mode="resume")
    assert result["status"] == "completed"
    assert result["accounting"]["requests_attempted"] == 3
    assert result["accounting"]["requests_attempted_this_session"] == 2
    assert result["accounting"]["usage_unknown_calls"] == 3
    assert result["accounting"]["reported_input_tokens"] is None
    assert resumed_backend.calls == ["signal", "priority"]
    assert result["evidence"]["run_id"] == json.loads((directory / "manifest.json").read_text())["content"]["run_id"]


def test_crash_after_durable_reservation_before_dispatch_remains_charged(tmp_path, monkeypatch):
    directory = tmp_path / "reservation"
    state = CASES["chain.json"]["state"]
    first_backend = Recording()
    original = HierarchyEvidence.save_ledger

    def crash_after_reservation(self, snapshot):
        original(self, snapshot)
        if snapshot["used"] == 1:
            raise RuntimeError("simulated death after reservation")

    with monkeypatch.context() as context:
        context.setattr(HierarchyEvidence, "save_ledger", crash_after_reservation)
        with pytest.raises(RuntimeError, match="after reservation"):
            runtime(backend=first_backend).run(state, evidence_dir=directory)
    assert first_backend.calls == []
    checkpoint = json.loads((directory / "attempts.json").read_text(encoding="utf-8"))["ledger"]
    assert checkpoint["used"] == 1
    assert checkpoint["receipts"][0]["status"] == "in_flight"
    resumed_backend = Recording()
    result = runtime(backend=resumed_backend).run(state, evidence_dir=directory, evidence_mode="resume")
    assert result["status"] == "completed"
    assert result["accounting"]["requests_attempted"] == 3
    assert result["accounting"]["attempt_ledger"]["receipts"][0]["status"] == "uncertain"
    assert result["accounting"]["reported_output_tokens"] is None
    assert resumed_backend.calls == ["signal", "priority"]


def test_resume_manifest_only_initialization_without_dispatch(tmp_path, monkeypatch):
    directory = tmp_path / "initialization"
    state = CASES["chain.json"]["state"]
    first_backend = Recording()

    def crash_before_ledger(self, snapshot):
        raise RuntimeError("simulated initialization crash")

    with monkeypatch.context() as context:
        context.setattr(HierarchyEvidence, "save_ledger", crash_before_ledger)
        with pytest.raises(RuntimeError, match="initialization crash"):
            runtime(backend=first_backend).run(state, evidence_dir=directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["content"]
    assert first_backend.calls == []
    assert not (directory / "attempts.json").exists()
    resumed_backend = Recording()
    result = runtime(backend=resumed_backend).run(state, evidence_dir=directory, evidence_mode="resume")
    assert result["status"] == "completed"
    assert result["evidence"]["run_id"] == manifest["run_id"]
    assert result["accounting"]["requests_attempted"] == 2


def test_crash_after_stage_append_preserves_bytes_and_replays_completed_stage(tmp_path, monkeypatch):
    directory = tmp_path / "crash"
    state = CASES["chain.json"]["state"]
    original = HierarchyEvidence._append
    first_backend = Recording()

    def fail_after_append(self, payload):
        original(self, payload)
        if payload["kind"] == "stage":
            raise RuntimeError("simulated process death after durable append")

    with monkeypatch.context() as context:
        context.setattr(HierarchyEvidence, "_append", fail_after_append)
        with pytest.raises(RuntimeError, match="after durable"):
            runtime(backend=first_backend).run(state, evidence_dir=directory)
    before = (directory / "events.jsonl").read_bytes()
    with (directory / "events.jsonl").open("ab") as stream:
        stream.write(b'{"seq":2')  # Torn trailing append is discarded; completed lines survive.
    resumed_backend = Recording()
    result = runtime(backend=resumed_backend).run(state, evidence_dir=directory, evidence_mode="resume")
    after = (directory / "events.jsonl").read_bytes()
    assert result["status"] == "completed"
    assert result["accounting"]["requests_attempted"] == 2
    assert result["accounting"]["evidence_replayed_stages"] == 1
    assert resumed_backend.calls == ["priority"]
    assert after.startswith(before)
    assert after.count(before) == 1


def test_failed_run_history_is_retained_when_missing_work_resumes(tmp_path):
    class Unavailable(Recording):
        def evaluate(self, program, state):
            self.calls.append(program.name)
            raise BackendError("synthetic provider outage")

    directory = tmp_path / "failed"
    state = CASES["chain.json"]["state"]
    first = runtime(backend=Unavailable()).run(state, evidence_dir=directory)
    assert first["status"] == "failed"
    assert first["accounting"]["requests_attempted"] == 1
    resumed_backend = Recording()
    resumed = runtime(backend=resumed_backend).run(state, evidence_dir=directory, evidence_mode="resume")
    assert resumed["status"] == "completed"
    assert resumed["accounting"]["requests_attempted"] == 3
    assert resumed["evidence"]["run_id"] == first["evidence"]["run_id"]
    kinds = [json.loads(line)["payload"]["kind"]
             for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert kinds == ["failure", "stage", "stage", "final"]
    assert resumed_backend.calls == ["signal", "priority"]


def test_replay_rejects_wrong_state_graph_and_tampered_raw_response(tmp_path):
    state = CASES["chain.json"]["state"]
    directory = tmp_path / "evidence"
    runtime().run(state, evidence_dir=directory)
    with pytest.raises(DataError, match="plan, data"):
        runtime().run({"message": "different"}, evidence_dir=directory, evidence_mode="replay")
    altered = runtime()
    altered.artifact.nodes[0].program.questions["urgent"].instructions = "changed"
    with pytest.raises((DataError, ConfigurationError)):
        altered.run(state, evidence_dir=directory, evidence_mode="replay")
    policy_changed = HierarchyRuntime(runtime().artifact, ManagedBackend(Recording()),
                                      retry_policy=GraphRetryPolicy(max_transient_retries=1))
    with pytest.raises(DataError, match="plan, data"):
        policy_changed.run(state, evidence_dir=directory, evidence_mode="resume")
    budget_changed = HierarchyRuntime(runtime().artifact, ManagedBackend(Recording(), max_calls=1))
    with pytest.raises(DataError, match="owner budget"):
        budget_changed.run(state, evidence_dir=directory, evidence_mode="resume")
    events_path = directory / "events.jsonl"
    events = events_path.read_text(encoding="utf-8")
    events_path.write_text(events.replace('"noul":', '"noul_tampered":', 1), encoding="utf-8")
    with pytest.raises(DataError, match="event chain"):
        runtime().run(state, evidence_dir=directory, evidence_mode="replay")


def test_replay_rejects_insufficient_raw_validation_evidence_even_with_recomputed_checksums(tmp_path):
    from s1compiler.io import canonical, fingerprint

    state = CASES["chain.json"]["state"]
    directory = tmp_path / "evidence"
    runtime().run(state, evidence_dir=directory)
    path = directory / "events.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    del events[0]["payload"]["response"]["answers"]["urgent"]["noul"]
    previous = "0" * 64
    for event in events:
        event["prev_sha256"] = previous
        event["sha256"] = fingerprint({key: event[key] for key in ("seq", "prev_sha256", "payload")})
        previous = event["sha256"]
    path.write_text("".join(canonical(event) + "\n" for event in events), encoding="utf-8")
    with pytest.raises(DataError, match="replay differs"):
        runtime().run(state, evidence_dir=directory, evidence_mode="replay")


def test_checkpoint_checksum_rejects_modified_reservations(tmp_path):
    directory = tmp_path / "evidence"
    state = CASES["chain.json"]["state"]
    runtime().run(state, evidence_dir=directory)
    path = directory / "attempts.json"
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    checkpoint["ledger"]["used"] = 0
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(DataError, match="checkpoint checksum"):
        runtime().run(state, evidence_dir=directory, evidence_mode="replay")


def test_replay_missing_evidence_and_default_run_reports_unavailable(tmp_path):
    state = CASES["chain.json"]["state"]
    ordinary = runtime().run(state)
    assert ordinary["evidence"] == {"mode": "none", "resume_available": False}
    with pytest.raises(ConfigurationError, match="requires an evidence directory"):
        runtime().run(state, evidence_mode="replay")
    with pytest.raises(ConfigurationError, match="does not exist"):
        runtime().run(state, evidence_dir=tmp_path / "missing", evidence_mode="replay")
    directory = tmp_path / "partial"
    result = runtime().run(state, evidence_dir=directory, cancel_requested=lambda: True)
    assert result["status"] == "cancelled"
    with pytest.raises(DataError, match="complete final"):
        runtime().run(state, evidence_dir=directory, evidence_mode="replay")


def test_evidence_redacts_input_and_secrets_but_records_request_fingerprint(tmp_path):
    directory = tmp_path / "evidence"
    source = copy.deepcopy(CASES["chain.json"]["state"])
    source["secret"] = "test-only-secret-value"
    runtime().run(source, evidence_dir=directory)
    text = "".join(path.read_text(encoding="utf-8") for path in directory.glob("*.json*"))
    assert "test-only-secret-value" not in text
    assert "TYPESAFE_API_KEY" not in text
    assert "urgent outage" not in text
    assert '"input_sha256"' in text
    node = runtime().artifact.nodes[0]
    node.inputs["message"] = RootRef(root="message", default="test-only-secret-default")
    assert "test-only-secret-default" not in json.dumps(_input_references(node))


def test_exclusive_owner_rejects_duplicate_controller(tmp_path):
    state = CASES["chain.json"]["state"]
    directory = tmp_path / "evidence"
    initial = runtime()
    initial.run(state, evidence_dir=directory)
    artifact = initial.artifact
    backend = ManagedBackend(Recording())
    from s1compiler.io import fingerprint
    from dataclasses import asdict
    with HierarchyEvidence(directory, artifact=artifact, backend=backend,
                           input_sha256=fingerprint(state), max_graph_attempts=64,
                           node_limits={}, policy=asdict(GraphRetryPolicy()), mode="replay"):
        with pytest.raises(ConfigurationError, match="live owner"):
            runtime().run(state, evidence_dir=directory, evidence_mode="replay")


def test_implementation_hash_ignores_line_endings_but_not_code(tmp_path, monkeypatch):
    import s1compiler.hierarchy_evidence as module
    source_dir = Path(module.__file__).parent
    names = ("hierarchy_evidence.py", "hierarchy_runtime.py", "hierarchy.py", "hierarchy_validation.py",
             "runtime.py", "backends.py", "models.py")

    def copy_tree(target, newline):
        target.mkdir()
        for name in names:
            text = (source_dir / name).read_bytes().replace(b"\r\n", b"\n")
            (target / name).write_bytes(text.replace(b"\n", newline))
        return target / "hierarchy_evidence.py"

    lf = copy_tree(tmp_path / "lf", b"\n")
    crlf = copy_tree(tmp_path / "crlf", b"\r\n")
    monkeypatch.setattr(module, "__file__", str(lf))
    unix_hash = module.implementation_hash()
    monkeypatch.setattr(module, "__file__", str(crlf))
    assert module.implementation_hash() == unix_hash

    edited = copy_tree(tmp_path / "edited", b"\n")
    runtime = edited.with_name("runtime.py")
    runtime.write_bytes(runtime.read_bytes() + b"# changed behaviour\n")
    monkeypatch.setattr(module, "__file__", str(edited))
    assert module.implementation_hash() != unix_hash


@pytest.mark.parametrize("mode", ["resume", "replay"])
def test_oversized_event_fails_without_repair_or_dispatch(tmp_path, mode):
    from s1compiler.hierarchy_evidence import MAX_LINE_BYTES

    state = CASES["chain.json"]["state"]
    directory = tmp_path / "oversized"
    runtime().run(state, evidence_dir=directory)
    path = directory / "events.jsonl"
    before = path.read_bytes() + b"x" * (MAX_LINE_BYTES + 1)
    path.write_bytes(before)
    backend = Recording()
    with pytest.raises(DataError, match="event is oversized"):
        runtime(backend=backend).run(state, evidence_dir=directory, evidence_mode=mode)
    assert backend.calls == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("limits", [
    {"max_graph_attempts": 2},
    {"node_attempt_limits": {"signal": 1}},
])
def test_resume_rejects_changed_graph_or_node_limits_before_mutation(tmp_path, limits):
    state = CASES["chain.json"]["state"]
    directory = tmp_path / "limits"
    initial = runtime()
    initial.run(state, evidence_dir=directory)
    before = {name: (directory / name).read_bytes()
              for name in ("manifest.json", "attempts.json", "events.jsonl")}
    backend = Recording()
    changed = HierarchyRuntime(initial.artifact, ManagedBackend(backend), **limits)
    with pytest.raises(DataError, match="plan, data"):
        changed.run(state, evidence_dir=directory, evidence_mode="resume")
    assert backend.calls == []
    assert {name: (directory / name).read_bytes() for name in before} == before


@pytest.mark.parametrize("mode", ["resume", "replay"])
def test_event_sequence_rejected_even_with_valid_recomputed_hash_chain(tmp_path, mode):
    from s1compiler.io import canonical, fingerprint

    state = CASES["chain.json"]["state"]
    directory = tmp_path / "sequence"
    runtime().run(state, evidence_dir=directory)
    path = directory / "events.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events[0]["seq"], events[1]["seq"] = events[1]["seq"], events[0]["seq"]
    previous = "0" * 64
    for event in events:
        event["prev_sha256"] = previous
        event["sha256"] = fingerprint({key: event[key] for key in ("seq", "prev_sha256", "payload")})
        previous = event["sha256"]
    path.write_text("".join(canonical(event) + "\n" for event in events), encoding="utf-8")
    before = path.read_bytes()
    backend = Recording()
    with pytest.raises(DataError, match="event chain"):
        runtime(backend=backend).run(state, evidence_dir=directory, evidence_mode=mode)
    assert backend.calls == []
    assert path.read_bytes() == before


def test_replay_rejects_torn_tail_without_truncating_completed_evidence(tmp_path):
    state = CASES["chain.json"]["state"]
    directory = tmp_path / "torn-replay"
    runtime().run(state, evidence_dir=directory)
    path = directory / "events.jsonl"
    before = path.read_bytes() + b'{"seq":4'
    path.write_bytes(before)
    backend = Recording()
    with pytest.raises(DataError, match="incomplete trailing event"):
        runtime(backend=backend).run(state, evidence_dir=directory, evidence_mode="replay")
    assert backend.calls == []
    assert path.read_bytes() == before


def _completed_evidence(tmp_path):
    state = CASES["chain.json"]["state"]
    directory = tmp_path / "evidence"
    first = runtime().run(state, evidence_dir=directory)
    assert first["status"] == "completed"
    return state, directory, first


def test_resume_repairs_a_torn_trailing_event_and_replay_refuses_it(tmp_path):
    state, directory, first = _completed_evidence(tmp_path)
    events = directory / "events.jsonl"
    intact = events.read_bytes()
    events.write_bytes(intact + b'{"seq":99,"prev_sha256":"')  # process died mid-append, no newline
    with pytest.raises(DataError, match="incomplete trailing event"):
        runtime().run(state, evidence_dir=directory, evidence_mode="replay")
    assert events.read_bytes() != intact  # replay never repairs
    backend = Recording()
    resumed = runtime(backend=backend).run(state, evidence_dir=directory, evidence_mode="resume")
    assert resumed == first
    assert backend.calls == []
    assert events.read_bytes() == intact


def test_swapped_or_dropped_events_break_the_chain(tmp_path):
    state, directory, _ = _completed_evidence(tmp_path)
    events = directory / "events.jsonl"
    lines = events.read_bytes().splitlines(keepends=True)
    assert len(lines) >= 3
    swapped = [lines[1], lines[0], *lines[2:]]
    events.write_bytes(b"".join(swapped))
    with pytest.raises(DataError, match="event chain"):
        runtime().run(state, evidence_dir=directory, evidence_mode="replay")
    events.write_bytes(b"".join([lines[0], *lines[2:]]))  # one event removed from the middle
    with pytest.raises(DataError, match="event chain"):
        runtime().run(state, evidence_dir=directory, evidence_mode="replay")
