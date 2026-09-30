"""Opt-in, local and sensitive hierarchy response evidence.

The OS file lock prevents concurrent controllers. Checksums detect accidental
changes; they do not authenticate a maliciously edited evidence directory.
"""
from __future__ import annotations

import copy
import hashlib
from importlib import metadata
import os
from pathlib import Path
import sys
import uuid
from typing import Any

from .backends import AttemptLedger, ManagedBackend, Response
from .errors import ConfigurationError, DataError
from .hierarchy import RootRef
from .io import canonical, fingerprint, json_loads


FORMAT = "systemone-hierarchy-evidence/v1"
LEDGER_FORMAT = "systemone-hierarchy-ledger-checkpoint/v1"
MAX_LINE_BYTES = 8_000_000


def package_version() -> str:
    try:
        return metadata.version("systemone-compiler")
    except metadata.PackageNotFoundError:
        return "source-uninstalled"


def _input_references(node) -> dict[str, dict[str, str]]:
    """Keep provenance names, never optional literal default values."""
    return {port: ({"root": ref.root} if isinstance(ref, RootRef) else
                   {"stage": ref.stage, "decision": ref.decision, "field": ref.field})
            for port, ref in node.inputs.items()}


def implementation_hash() -> str:
    root = Path(__file__).parent
    modules = ("hierarchy_evidence.py", "hierarchy_runtime.py", "hierarchy.py",
               "hierarchy_validation.py", "runtime.py", "backends.py", "models.py")
    return fingerprint({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in modules})


def _durable_json(path: Path, value: Any) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        _sync_dir(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def _sync_dir(directory: Path) -> None:
    if sys.platform == "win32":
        return  # os.replace is atomic locally; Windows has no portable directory fsync.
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _lock(file):
    if sys.platform == "win32":
        import msvcrt
        file.seek(0)
        try:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise ConfigurationError("Hierarchy evidence already has a live owner.") from exc
    else:
        import fcntl
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ConfigurationError("Hierarchy evidence already has a live owner.") from exc


def _unlock(file):
    if sys.platform == "win32":
        import msvcrt
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


class HierarchyEvidence:
    """One durable example, owned for the entire execution or replay."""

    def __init__(self, directory: str | Path, *, artifact, backend: ManagedBackend,
                 input_sha256: str, max_graph_attempts: int, node_limits: dict[str, int],
                 policy: dict[str, Any], mode: str,
                 lineage_identity: dict[str, Any] | None = None):
        if mode not in {"create", "resume", "replay"}:
            raise ConfigurationError("Unknown hierarchy evidence mode.")
        self.directory, self.mode = Path(directory), mode
        self.artifact, self.backend = artifact, backend
        self.input_sha256 = input_sha256
        self.lineage_identity = copy.deepcopy(lineage_identity)
        self.max_graph_attempts, self.node_limits, self.policy = max_graph_attempts, node_limits, policy
        self.lock_file = None
        self.ledger: AttemptLedger | None = None
        self.stages: dict[str, dict[str, Any]] = {}
        self.final: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []
        self.replayed_stages = 0

    def __enter__(self):
        if self.mode == "create":
            self.directory.mkdir(parents=True, exist_ok=False)
            try:
                os.chmod(self.directory, 0o700)
            except OSError:
                pass
        elif not self.directory.is_dir():
            raise ConfigurationError("Hierarchy evidence directory does not exist.")
        self.lock_file = (self.directory / ".owner.lock").open("a+b")
        try:
            _lock(self.lock_file)
            self._open_locked()
        except BaseException:
            self.lock_file.close()
            self.lock_file = None
            raise
        return self

    def __exit__(self, *_):
        if self.lock_file is not None:
            _unlock(self.lock_file)
            self.lock_file.close()
            self.lock_file = None

    def _identity(self, owner_start_used: int, run_id: str) -> dict[str, Any]:
        return {"format": FORMAT, "run_id": run_id, "graph_sha256": self.artifact.content_hash,
                "input_sha256": self.input_sha256, "model": self.artifact.source.model,
                "lineage_identity": self.lineage_identity,
                "backend": self.backend.identity, "synthetic": self.backend.synthetic,
                "package_version": package_version(), "implementation_sha256": implementation_hash(),
                "max_graph_attempts": self.max_graph_attempts, "node_limits": self.node_limits,
                "retry_policy": self.policy, "owner_budget_maximum": self.backend.budget.maximum,
                "owner_start_used": owner_start_used}

    def _open_locked(self):
        manifest_path = self.directory / "manifest.json"
        ledger_path = self.directory / "attempts.json"
        if self.mode == "create":
            self.manifest = self._identity(self.backend.budget.used, uuid.uuid4().hex)
            _durable_json(manifest_path, {"content": self.manifest,
                                          "sha256": fingerprint(self.manifest)})
            self.ledger = AttemptLedger(graph_sha256=self.artifact.content_hash,
                                        maximum=self.max_graph_attempts,
                                        nodes={node.id for node in self.artifact.nodes},
                                        node_limits=self.node_limits, policy=self.policy)
            self.save_ledger(self.ledger.snapshot())
            (self.directory / "events.jsonl").touch(exist_ok=False)
            _sync_dir(self.directory)
        else:
            envelope = self._read_json(manifest_path)
            if set(envelope) != {"content", "sha256"} or (
                envelope["sha256"] != fingerprint(envelope["content"])
            ):
                raise DataError("Hierarchy evidence manifest checksum is invalid.")
            self.manifest = envelope["content"]
            if not isinstance(self.manifest, dict) or type(self.manifest.get("owner_start_used")) is not int or (
                not isinstance(self.manifest.get("run_id"), str)
            ):
                raise DataError("Hierarchy evidence manifest is malformed.")
            expected = self._identity(self.manifest["owner_start_used"], self.manifest["run_id"])
            if self.manifest != expected:
                raise DataError("Hierarchy evidence plan, data, model, owner budget, or code changed.")
            if self.mode == "resume" and not ledger_path.exists() and not (
                self.directory / "events.jsonl"
            ).exists():
                # Initialization stopped after the manifest but before any
                # ledger existed, so dispatch was not yet possible.
                self.ledger = AttemptLedger(graph_sha256=self.artifact.content_hash,
                                            maximum=self.max_graph_attempts,
                                            nodes={node.id for node in self.artifact.nodes},
                                            node_limits=self.node_limits, policy=self.policy)
                self.save_ledger(self.ledger.snapshot())
                (self.directory / "events.jsonl").touch(exist_ok=False)
                _sync_dir(self.directory)
            ledger_envelope = self._read_json(ledger_path)
            if set(ledger_envelope) != {"format", "ledger", "sha256"} or (
                ledger_envelope["format"] != LEDGER_FORMAT or
                ledger_envelope["sha256"] != fingerprint(ledger_envelope["ledger"])
            ):
                raise DataError("Hierarchy attempt checkpoint checksum or format differs.")
            self.ledger = AttemptLedger.restore(ledger_envelope["ledger"],
                                                graph_sha256=self.artifact.content_hash,
                                                maximum=self.max_graph_attempts,
                                                nodes={node.id for node in self.artifact.nodes},
                                                node_limits=self.node_limits, policy=self.policy)
            self._read_events()
            if self.mode == "resume":
                expected_used = self.manifest["owner_start_used"] + self.ledger.snapshot()["used"]
                if self.backend.budget.used not in {0, expected_used}:
                    raise ConfigurationError("Owner budget cannot be reconciled with durable reservations.")
                self.backend.budget.used = expected_used
                self.save_ledger(self.ledger.snapshot())
        if self.mode != "replay":
            self.ledger.on_change = self.save_ledger

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file() or path.stat().st_size > MAX_LINE_BYTES:
            raise DataError("Hierarchy evidence file is missing or oversized.")
        value = json_loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise DataError("Hierarchy evidence file must contain an object.")
        return value

    def save_ledger(self, snapshot: dict[str, Any]) -> None:
        _durable_json(self.directory / "attempts.json",
                      {"format": LEDGER_FORMAT, "ledger": snapshot, "sha256": fingerprint(snapshot)})

    def _read_events(self) -> None:
        path = self.directory / "events.jsonl"
        if not path.is_file():
            raise DataError("Hierarchy evidence event stream is missing.")
        previous = "0" * 64
        last_attempt_count = 0
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_LINE_BYTES:
                    raise DataError("Hierarchy evidence event is oversized.")
                if not line.endswith(b"\n"):
                    if self.mode == "replay":
                        raise DataError("Hierarchy evidence has an incomplete trailing event.")
                    with path.open("r+b") as repair:
                        repair.truncate(offset)
                        repair.flush()
                        os.fsync(repair.fileno())
                    break
                event = json_loads(line.decode("utf-8"))
                self._validate_event(event, len(self.records) + 1, previous, last_attempt_count)
                self.records.append(event)
                previous = event["sha256"]
                if event["payload"]["kind"] in {"stage", "failure"}:
                    last_attempt_count = event["payload"]["attempts_used"]

    def _validate_event(self, event: dict[str, Any], index: int, previous: str,
                        last_attempt_count: int) -> None:
        if not isinstance(event, dict) or set(event) != {"seq", "prev_sha256", "payload", "sha256"} or (
            type(event["seq"]) is not int or event["seq"] != index or
            event["prev_sha256"] != previous or
            event["sha256"] != fingerprint({key: event[key] for key in ("seq", "prev_sha256", "payload")})
        ):
            raise DataError("Hierarchy evidence event chain is invalid.")
        payload = event["payload"]
        if not isinstance(payload, dict) or payload.get("format") != FORMAT:
            raise DataError("Hierarchy evidence payload is invalid.")
        kind = payload.get("kind")
        if kind == "stage":
            required = {"format", "kind", "run_id", "node_id", "parent", "graph_sha256",
                        "program_sha256", "state_sha256", "model", "backend", "synthetic",
                        "status", "response", "answers", "decisions", "inputs", "attempts_used",
                        "attempt_receipt"}
            node_id = payload.get("node_id")
            node = next((item for item in self.artifact.nodes if item.id == node_id), None)
            if set(payload) != required or not isinstance(node_id, str) or node_id in self.stages or (
                node is None or self.final or
                payload.get("status") != "completed" or
                payload.get("parent") != (node_id.rpartition("/")[0] or None) or
                payload.get("graph_sha256") != self.artifact.content_hash or
                payload.get("program_sha256") != node.program.content_hash or
                payload.get("model") != self.artifact.source.model or
                payload.get("backend") != self.backend.identity or
                payload.get("synthetic") != self.backend.synthetic or
                not isinstance(payload.get("answers"), dict) or
                not isinstance(payload.get("decisions"), dict) or
                payload.get("inputs") != _input_references(node)
            ):
                raise DataError("Hierarchy evidence has duplicate or unknown stage data.")
            receipt = payload["attempt_receipt"]
            reservations = self.ledger.snapshot()["receipts"]
            response_data = payload["response"]
            if not isinstance(response_data, dict) or response_data.get("cached") is not (receipt is None):
                raise DataError("Hierarchy stage cache and attempt evidence disagree.")
            if receipt is not None and (
                type(receipt) is not int or not 1 <= receipt <= len(reservations) or
                reservations[receipt - 1]["node_id"] != node_id or
                reservations[receipt - 1]["status"] != "validated"
            ):
                raise DataError("Hierarchy stage attempt receipt is invalid.")
            self.stages[node_id] = payload
        elif kind == "final":
            if set(payload) != {"format", "kind", "run_id", "result"} or self.final is not None or (
                not isinstance(payload.get("result"), dict) or
                payload["result"].get("graph_sha256") != self.artifact.content_hash or
                payload["result"].get("status") not in {"completed", "review_required"} or
                not isinstance(payload["result"].get("executed"), list) or
                set(payload["result"]["executed"]) != set(self.stages) or
                not isinstance(payload["result"].get("accounting"), dict) or
                payload["result"]["accounting"].get("attempt_ledger") != self.ledger.snapshot() or
                payload["result"]["accounting"].get("requests_attempted") != self.ledger.snapshot()["used"]
            ):
                raise DataError("Hierarchy evidence has duplicate final results.")
            self.final = payload
        elif kind != "failure" or self.final is not None or (
            set(payload) != {"format", "kind", "run_id", "status", "error", "attempts_used"} or
            payload.get("status") not in {"failed", "cancelled"}
        ):
            raise DataError("Hierarchy evidence has an invalid event kind or order.")
        if payload.get("run_id") != self.manifest["run_id"]:
            raise DataError("Hierarchy evidence run identity differs.")
        if kind in {"stage", "failure"}:
            count = payload.get("attempts_used")
            if type(count) is not int or not last_attempt_count <= count <= self.ledger.snapshot()["used"]:
                raise DataError("Hierarchy evidence attempt count is inconsistent.")

    def _append(self, payload: dict[str, Any]) -> None:
        if self.mode == "replay" or self.final is not None:
            raise ConfigurationError("Hierarchy evidence is read-only or finalized.")
        seq = len(self.records) + 1
        previous = self.records[-1]["sha256"] if self.records else "0" * 64
        event = {"seq": seq, "prev_sha256": previous, "payload": payload}
        event["sha256"] = fingerprint(event)
        data = (canonical(event) + "\n").encode("utf-8")
        if len(data) > MAX_LINE_BYTES:
            raise DataError("Hierarchy evidence event is oversized.")
        with (self.directory / "events.jsonl").open("ab") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self.records.append(event)

    def stage_response(self, node_id: str, program, state: dict[str, Any]) -> Response | None:
        payload = self.stages.get(node_id)
        if payload is None:
            if self.mode == "replay" or self.final is not None:
                raise DataError("Offline replay lacks a completed stage response.")
            return None
        if payload.get("program_sha256") != program.content_hash or (
            payload.get("state_sha256") != fingerprint(state) or payload.get("model") != program.model or
            payload.get("backend") != self.backend.identity or payload.get("synthetic") != self.backend.synthetic
        ):
            raise DataError("Recorded stage response differs from the frozen request identity.")
        response_data = payload.get("response")
        if not isinstance(response_data, dict) or set(response_data) != {
            "answers", "model", "usage", "latency_ms", "cached", "synthetic"
        }:
            raise DataError("Recorded stage response is malformed.")
        self.replayed_stages += 1
        return Response(**copy.deepcopy(response_data))

    def append_stage(self, node, state: dict[str, Any], result: dict[str, Any], response: Response) -> None:
        if node.id in self.stages:
            expected = self.stages[node.id]
            if expected.get("decisions") != result["decisions"] or expected.get("answers") != result["answers"]:
                raise DataError("Offline stage decisions differ from recorded evidence.")
            return
        payload = {"format": FORMAT, "kind": "stage", "run_id": self.manifest["run_id"],
                   "node_id": node.id, "parent": node.id.rpartition("/")[0] or None,
                   "graph_sha256": self.artifact.content_hash,
                   "program_sha256": node.program.content_hash,
                   "state_sha256": fingerprint(state), "model": node.program.model,
                   "backend": self.backend.identity, "synthetic": self.backend.synthetic,
                   "status": "completed", "response": response.to_dict(),
                   "attempt_receipt": response.attempt_receipt,
                   "answers": result["answers"], "decisions": result["decisions"],
                   "inputs": _input_references(node),
                   "attempts_used": self.ledger.snapshot()["used"]}
        self._append(payload)
        self.stages[node.id] = payload

    def reconcile_accounting(self, result: dict[str, Any]) -> None:
        """Report whole-example attempts; unknown prior usage never becomes zero."""
        receipts = self.ledger.snapshot()["receipts"]
        recorded = {item["attempt_receipt"]: item for item in self.stages.values()
                    if item["attempt_receipt"] is not None}
        missing = [item for item in receipts if item["id"] not in recorded]
        unknown = len(missing)
        input_tokens = output_tokens = 0
        for item in recorded.values():
            usage = item["response"]["usage"]
            if not isinstance(usage, dict) or usage.get("input_tokens") is None or (
                usage.get("output_tokens") is None
            ):
                unknown += 1
            else:
                input_tokens += usage["input_tokens"]
                output_tokens += usage["output_tokens"]
        accounting = result["accounting"]
        accounting["requests_attempted_this_session"] = accounting["requests_attempted"]
        accounting["requests_attempted"] = len(receipts)
        accounting["retries"] = len(receipts) - len({item["node_id"] for item in receipts})
        accounting["leaf_evaluations"] = len(self.stages) + len({item["node_id"] for item in missing
                                                                   if item["node_id"] not in self.stages})
        accounting["usage_unknown_calls"] = unknown
        accounting["reported_input_tokens"] = None if unknown else input_tokens
        accounting["reported_output_tokens"] = None if unknown else output_tokens
        accounting["evidence_replayed_stages"] = self.replayed_stages

    def append_result(self, result: dict[str, Any]) -> None:
        if set(self.stages) != set(result["executed"]):
            raise DataError("Completed graph stages and durable stage evidence differ.")
        payload = {"format": FORMAT, "kind": "final", "run_id": self.manifest["run_id"],
                   "result": copy.deepcopy(result)}
        self._append(payload)
        self.final = payload

    def append_failure(self, result: dict[str, Any]) -> None:
        payload = {"format": FORMAT, "kind": "failure", "run_id": self.manifest["run_id"],
                   "status": result["status"], "error": result.get("error"),
                   "attempts_used": self.ledger.snapshot()["used"]}
        self._append(payload)
