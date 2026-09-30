"""Preregistered flat-versus-hierarchy study; no paid calls at import time.

Registration is read-only except for its new protocol file. Selection and held-out
execution are separate phases so test labels cannot enter candidate proposals.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any

from .backends import ManagedBackend, MockBackend, Response, TypeSafeBackend
from .data import dataset_hash
from .errors import ConfigurationError, DataError, S1Error
from .hierarchy import HierarchyArtifact, HierarchySource, lower_hierarchy
from .hierarchy_architect import semantic_review_manifest
from .hierarchy_compiler import HierarchyCompileOptions, HierarchyCompiler
from .hierarchy_data import HierarchySplitGuard, read_hierarchy_jsonl
from .hierarchy_metrics import (_flat_quality, evaluate_hierarchy, report_from_hierarchy_results)
from .hierarchy_runtime import HierarchyRuntime
from .hierarchy_validation import validate_hierarchy_compile_inputs
from .io import atomic_json, fingerprint, load_document
from .metrics import evaluate, report_from_results
from .models import Program, assert_contract, project_state
from .policy import fit_policies
from .runtime import Runtime, apply_policy, normalize_answers


FORMAT = "systemone-hierarchy-study-protocol/v1"
SPLITS = ("train", "validation", "calibration", "test")
ARMS = ("flat_authored", "hierarchy_authored", "hierarchy_selected")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _envelope_write(path: Path, content: dict[str, Any]) -> None:
    atomic_json(path, {"content": content, "sha256": fingerprint(content)})


def _envelope_read(path: Path) -> dict[str, Any]:
    envelope = load_document(path)
    if not isinstance(envelope, dict) or set(envelope) != {"content", "sha256"} or (
        not isinstance(envelope["content"], dict) or
        envelope["sha256"] != fingerprint(envelope["content"])
    ):
        raise DataError("Hierarchy study envelope checksum or shape differs.")
    return envelope["content"]


def _inputs(source_path: Path, candidate_path: Path, flat_path: Path, split_paths: dict[str, Path]):
    source = HierarchySource.load(source_path)
    candidate = HierarchySource.load(candidate_path)
    flat = Program.load(flat_path)
    if source.source != candidate.source or source.limits != candidate.limits:
        raise DataError("Candidate hierarchy changes the fixed root contract, model, or graph limits.")
    assert_contract(flat, source.source)
    if set(split_paths) != set(SPLITS):
        raise DataError("A hierarchy study requires train, validation, calibration, and test paths.")
    draft = lower_hierarchy(source)
    candidate_draft = lower_hierarchy(candidate)
    splits = {name: read_hierarchy_jsonl(split_paths[name], draft) for name in SPLITS}
    validate_hierarchy_compile_inputs(source, splits)
    validate_hierarchy_compile_inputs(candidate, splits)
    return source, candidate, flat, splits, draft, candidate_draft


def estimate_requests(source: HierarchySource, splits: dict[str, list], *,
                      candidate: HierarchySource | None = None,
                      selected_method: str, structural_rounds: int = 0,
                      max_metric_calls: int = 0, teacher_max_calls: int = 0,
                      teacher_provider_calls_per_signature: int = 2) -> dict[str, Any]:
    """Conservative call ceilings for zero-retry graph arms and one flat arm.

    These count native request attempts, not graph executions or tokens. Any
    different retry, search, or teacher plan needs a new protocol identity.
    """
    if selected_method not in {"authored_validation", "dspy_gepa"} or any(
        name not in splits or not splits[name] for name in SPLITS
    ):
        raise ConfigurationError("Invalid preregistered arms or empty split.")
    if (type(structural_rounds) is not int or not 0 <= structural_rounds <= 10 or
        type(max_metric_calls) is not int or max_metric_calls < 0 or
        type(teacher_max_calls) is not int or teacher_max_calls < 0 or
        type(teacher_provider_calls_per_signature) is not int or
        teacher_provider_calls_per_signature < 1):
        raise ConfigurationError("Invalid graph or teacher budget.")
    if selected_method == "authored_validation" and (structural_rounds or max_metric_calls or teacher_max_calls):
        raise ConfigurationError("Authored validation selection must have zero teacher and optimizer calls.")
    if selected_method == "dspy_gepa" and (structural_rounds < 1 or max_metric_calls < 1 or teacher_max_calls < 1):
        raise ConfigurationError("DSPy/GEPA selection requires bounded rounds, metrics, and teacher calls.")
    leaf_calls = max(len(lower_hierarchy(source).nodes),
                     len(lower_hierarchy(candidate).nodes) if candidate is not None else 0)
    if selected_method == "dspy_gepa":
        # A bounded teacher proposal may expand to the declared native-call cap.
        leaf_calls = source.limits.max_native_calls_per_example
    if leaf_calls > source.limits.max_native_calls_per_example:
        raise ConfigurationError("Expanded graph exceeds its per-example native attempt limit.")
    n = {name: len(splits[name]) for name in SPLITS}
    flat_selection = n["validation"] + n["calibration"]
    authored_selection = leaf_calls * (n["train"] + n["validation"] + 2 * n["calibration"]) + n["validation"]
    selected_selection = authored_selection
    if selected_method == "authored_validation":
        # Two preregistered graph sources are compared on validation, then only
        # the chosen graph is calibrated. Selection of the candidate itself
        # evaluates train and validation, not calibration or test.
        selected_selection += leaf_calls * (n["train"] + n["validation"])
    else:
        selected_selection += leaf_calls * (structural_rounds * (n["validation"] + n["train"]) +
                                             max_metric_calls + n["validation"] + n["train"])
    by_arm = {
        "flat_authored": {"selection_calibration_ceiling": flat_selection,
                          "test_ceiling": n["test"]},
        "hierarchy_authored": {"selection_calibration_ceiling": authored_selection,
                               "test_ceiling": leaf_calls * n["test"]},
        "hierarchy_selected": {"selection_calibration_ceiling": selected_selection,
                               "test_ceiling": leaf_calls * n["test"]},
    }
    for arm in by_arm.values():
        arm["total_ceiling"] = arm["selection_calibration_ceiling"] + arm["test_ceiling"]
    common_selection = max(arm["selection_calibration_ceiling"] for arm in by_arm.values())
    common_test = max(arm["test_ceiling"] for arm in by_arm.values())
    for arm in by_arm.values():
        arm["allocated_selection_ceiling"] = common_selection
        arm["allocated_test_ceiling"] = common_test
    return {"graph_leaf_calls_worst_case_per_root": leaf_calls, "retry_policy": "zero_retries",
            "by_arm": by_arm,
            "native_attempt_ceiling_total": len(ARMS) * (common_selection + common_test),
            "common_selection_ceiling_per_arm": common_selection,
            "common_test_ceiling_per_arm": common_test,
            "teacher_signature_ceiling": teacher_max_calls,
            "teacher_provider_request_ceiling": teacher_max_calls * teacher_provider_calls_per_signature,
            "limitations": "Equal hard owner ceilings are allocated per arm, while estimated required "
                           "calls differ by topology and search. Realized calls, provider tokens, and "
                           "billed dollars are not inferred from these ceilings."}


def register(source_path: str | Path, candidate_path: str | Path, flat_path: str | Path,
             split_paths: dict[str, str | Path], out: str | Path, *, study_id: str,
             selected_method: str = "authored_validation", mode: str = "mock",
             seed: int = 7, analysis_seed: int = 90210, structural_rounds: int = 0,
             max_metric_calls: int = 0, teacher_max_calls: int = 0,
             teacher_model: str | None = None, teacher_max_tokens: int = 4096,
             bootstrap_replicates: int = 1000, randomization_replicates: int = 2000,
             data_attestation: dict[str, Any] | None = None,
             provider_call_price_cap_usd: float | None = None,
             teacher_call_price_cap_usd: float | None = None,
             external_billing_cap_usd: float | None = None) -> dict[str, Any]:
    """Freeze a new study plan before selection or test; never construct a provider."""
    output = Path(out)
    if output.exists() or not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", study_id):
        raise ConfigurationError("Use a fresh protocol path and a unique lowercase study ID.")
    if mode not in {"mock", "typesafe"} or type(seed) is not int or type(analysis_seed) is not int:
        raise ConfigurationError("Invalid study backend or seed.")
    if (type(bootstrap_replicates) is not int or bootstrap_replicates < 100 or
        type(randomization_replicates) is not int or randomization_replicates < 100):
        raise ConfigurationError("Paired cluster analysis needs at least 100 preregistered resamples.")
    paths = {"source": Path(source_path).resolve(), "candidate": Path(candidate_path).resolve(),
             "flat": Path(flat_path).resolve()}
    splits_at = {name: Path(path).resolve() for name, path in split_paths.items()}
    source, candidate, flat, splits, draft, candidate_draft = _inputs(
        paths["source"], paths["candidate"], paths["flat"], splits_at)
    if len({row.group or row.id for row in splits["test"]}) < 2:
        raise DataError("Confirmatory paired inference needs at least two independent test groups.")
    estimate = estimate_requests(source, splits, candidate=candidate, selected_method=selected_method,
                                 structural_rounds=structural_rounds, max_metric_calls=max_metric_calls,
                                 teacher_max_calls=teacher_max_calls)
    attestation = data_attestation or {}
    if mode == "typesafe":
        exclusions = attestation.get("prior_test_input_sha256s")
        if (attestation.get("label_origin") != "independent_human_reviewed" or
            not attestation.get("test_independence_evidence") or
            not attestation.get("reviewer") or
            not isinstance(exclusions, list) or not exclusions or
            any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in exclusions) or
            selected_method != "dspy_gepa" or not teacher_model):
            raise ConfigurationError("Live registration needs reviewed independent labels, teacher, "
                                     "and explicit test-independence evidence.")
        test_inputs = {fingerprint(project_state(source.source.state, row.state)) for row in splits["test"]}
        if test_inputs & set(exclusions):
            raise DataError("Live hierarchy test input overlaps an excluded prior-study holdout.")
        attestation = {**attestation, "prior_test_input_sha256s": None,
                       "prior_test_exclusion_digest": fingerprint(sorted(set(exclusions))),
                       "prior_test_exclusion_count": len(set(exclusions))}
        prices = (provider_call_price_cap_usd, teacher_call_price_cap_usd, external_billing_cap_usd)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in prices):
            raise ConfigurationError("Live registration needs positive per-call price caps and an "
                                     "externally enforced billing cap.")
        computed_cap = (estimate["native_attempt_ceiling_total"] * provider_call_price_cap_usd +
                        estimate["teacher_provider_request_ceiling"] * teacher_call_price_cap_usd)
        if computed_cap > external_billing_cap_usd:
            raise ConfigurationError("Declared billing cap is below worst-case call-price estimate.")
    else:
        computed_cap = 0.0
        if teacher_model is not None:
            raise ConfigurationError("Mock registration cannot configure a paid teacher model.")
    if teacher_max_tokens < 1:
        raise ConfigurationError("Teacher output-token limit must be positive.")
    protocol = {"format": FORMAT, "study_id": study_id,
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "mode": mode, "status": "registered_unexecuted", "deployment_approved": False,
                "source": {key: {"path": str(path), "file_sha256": _file_sha256(path)}
                           for key, path in paths.items()},
                "source_contract_sha256": fingerprint(source.source.model_dump(mode="json")),
                "authored_graph_sha256": draft.content_hash,
                "candidate_graph_sha256": candidate_draft.content_hash,
                "flat_program_sha256": flat.content_hash,
                "datasets": {name: {"path": str(splits_at[name]), "file_sha256": _file_sha256(splits_at[name]),
                                    "content_sha256": dataset_hash(splits[name]), "n": len(splits[name]),
                                    "n_groups": len({row.group or row.id for row in splits[name]})}
                             for name in SPLITS},
                "data_attestation": attestation,
                "analysis": {"seed": analysis_seed, "selection_seed": seed,
                             "bootstrap_replicates": bootstrap_replicates,
                             "randomization_replicates": randomization_replicates,
                             "primary": "mean paired root utility; review/incomplete=0, exact Choice/Noul "
                                        "and normalized Score error",
                             "confirmatory_family": ["hierarchy_selected-flat_authored",
                                                      "hierarchy_selected-hierarchy_authored"],
                             "multiplicity": "Holm over both prespecified primary comparisons; "
                                             "group-paired randomization and cluster bootstrap",
                             "secondary": ["route errors", "coverage", "native attempts", "latency",
                                           "reported tokens", "unknown usage", "review rate"]},
                "arms": {"flat_authored": "fixed template flat baseline",
                         "hierarchy_authored": "authored graph, no optimizer",
                         "hierarchy_selected": ("validation-selected authored candidate" if selected_method ==
                                                "authored_validation" else "DSPy structure plus GEPA wording")},
                "selected_method": selected_method,
                "optimization": {"structural_rounds": structural_rounds,
                                 "max_metric_calls": max_metric_calls,
                                 "teacher_max_calls": teacher_max_calls,
                                 "teacher_model": teacher_model,
                                 "teacher_max_tokens": teacher_max_tokens},
                "budgets": estimate | {"price_assumptions": {
                    "provider_call_price_cap_usd": provider_call_price_cap_usd,
                    "teacher_call_price_cap_usd": teacher_call_price_cap_usd,
                    "external_billing_cap_usd": external_billing_cap_usd,
                    "estimated_worst_case_usd": computed_cap,
                    "price_caps_require_independent_verification": mode == "typesafe"}},
                "authorization": {"paid_calls_approved": False, "teacher_example_sharing_approved": False,
                                  "frozen_semantic_review_approved": False},
                "limitations": ["Synthetic labels and mock output cannot establish Jev gains.",
                                "Independence and price attestation require human verification.",
                                "Equal hard arm ceilings are not equal search intensity, realized calls, tokens, or spend.",
                                "No test row is available to selection or teacher feedback."]}
    _envelope_write(output, protocol)
    return protocol


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = _envelope_read(Path(path))
    if protocol.get("format") != FORMAT:
        raise DataError("Unsupported hierarchy study protocol version.")
    paths = {key: Path(value["path"]) for key, value in protocol["source"].items()}
    split_paths = {key: Path(value["path"]) for key, value in protocol["datasets"].items()}
    if any(_file_sha256(paths[key]) != protocol["source"][key]["file_sha256"] for key in paths) or any(
        _file_sha256(split_paths[key]) != protocol["datasets"][key]["file_sha256"] for key in SPLITS
    ):
        raise DataError("Preregistered source or dataset bytes changed.")
    source, candidate, flat, splits, draft, candidate_draft = _inputs(
        paths["source"], paths["candidate"], paths["flat"], split_paths)
    if (fingerprint(source.source.model_dump(mode="json")) != protocol["source_contract_sha256"] or
        draft.content_hash != protocol["authored_graph_sha256"] or
        candidate_draft.content_hash != protocol["candidate_graph_sha256"] or
        flat.content_hash != protocol["flat_program_sha256"] or
        any(dataset_hash(splits[name]) != protocol["datasets"][name]["content_sha256"]
            for name in SPLITS)):
        raise DataError("Preregistered contract, graph, or split content changed.")
    return protocol


def _study_backend(protocol: dict[str, Any], arm: str, *, phase: str,
                   allow_paid: bool, backend_factory=None) -> ManagedBackend:
    ceiling = protocol["budgets"]["by_arm"][arm][
        "allocated_selection_ceiling" if phase == "selection" else "allocated_test_ceiling"]
    if backend_factory is not None:
        backend = backend_factory(arm, phase, ceiling)
    elif protocol["mode"] == "mock":
        backend = ManagedBackend(MockBackend(), max_calls=ceiling, cache=None)
    else:
        backend = ManagedBackend(TypeSafeBackend(allow_paid=allow_paid), max_calls=ceiling, cache=None)
    if backend.synthetic != (protocol["mode"] == "mock") or backend.budget.maximum != ceiling:
        raise ConfigurationError("Study backend identity or request ceiling differs from the protocol.")
    return backend


def select_and_freeze(protocol_path: str | Path, out: str | Path, *,
                      allow_paid: bool = False, share_feedback: bool = False,
                      approved_protocol_sha256: str | None = None,
                      backend_factory=None, teacher=None) -> dict[str, Any]:
    """Select on validation, fit on calibration, then seal all arms before test."""
    protocol = load_protocol(protocol_path)
    if protocol["mode"] == "typesafe" and (
        not allow_paid or not share_feedback or approved_protocol_sha256 != fingerprint(protocol)
    ):
        raise ConfigurationError("Live selection requires separate paid, teacher-sharing, and exact "
                                 "preregistered-protocol approvals.")
    if protocol["selected_method"] == "dspy_gepa" and teacher is None:
        if protocol["mode"] != "typesafe":
            raise ConfigurationError("Mock GEPA study needs an explicit synthetic teacher test double.")
        from .architect import DSPyTeacher
        teacher = DSPyTeacher(protocol["optimization"]["teacher_model"], allow_paid=allow_paid,
                              share_feedback=share_feedback,
                              max_calls=protocol["optimization"]["teacher_max_calls"],
                              max_tokens=protocol["optimization"]["teacher_max_tokens"])
    out = Path(out)
    if out.exists():
        raise ConfigurationError("Choose a new selection directory; frozen study state is immutable.")
    paths = {key: Path(value["path"]) for key, value in protocol["source"].items()}
    split_paths = {name: Path(value["path"]) for name, value in protocol["datasets"].items()}
    source, candidate, flat, splits, _, candidate_draft = _inputs(
        paths["source"], paths["candidate"], paths["flat"], split_paths)
    out.mkdir(parents=True, exist_ok=False)
    _envelope_write(out / "selection-started.json", {
        "protocol_sha256": fingerprint(protocol), "status": "started",
        "synthetic": protocol["mode"] == "mock", "test_executed": False})
    arms: dict[str, Any] = {}
    current = None
    try:
        current = "flat_authored"
        flat_backend = _study_backend(protocol, current, phase="selection",
                                      allow_paid=allow_paid, backend_factory=backend_factory)
        try:
            flat_validation, _ = evaluate(flat, splits["validation"], flat_backend)
            _, calibration_results = evaluate(flat, splits["calibration"], flat_backend)
            fitted, calibration_fit = fit_policies(
                flat, splits["calibration"], calibration_results, max_error=0.05,
                min_samples=1 if protocol["mode"] == "mock" else 10)
            fitted.provenance = {"status": "synthetic" if flat_backend.synthetic else "measured",
                                 "deployment_approved": False, "study_id": protocol["study_id"],
                                 "protocol_sha256": fingerprint(protocol), "arm": current,
                                 "source_sha256": protocol["source_contract_sha256"]}
            assert_contract(fitted, source.source)
            fitted.save(out / "flat_authored.s1.json")
            arms[current] = {"file": "flat_authored.s1.json", "content_sha256": fitted.content_hash,
                             "provenance_sha256": fingerprint(fitted.provenance),
                             "validation": flat_validation, "calibration_fit": calibration_fit,
                             "accounting": flat_backend.accounting(),
                             "semantic_review": {"status": "authored_contract_only"}}
        finally:
            flat_backend.close()

        current = "hierarchy_authored"
        authored_backend = _study_backend(protocol, current, phase="selection",
                                          allow_paid=allow_paid, backend_factory=backend_factory)
        try:
            authored_compiler = HierarchyCompiler(
                authored_backend, options=HierarchyCompileOptions(
                    min_calibration_samples=1 if protocol["mode"] == "mock" else 10))
            authored_session = authored_compiler.select(source, **splits)
            authored_compiler.calibrate(authored_session)
            authored_artifact = authored_compiler.freeze(authored_session)
            authored_artifact.save(out / "hierarchy_authored.s1.json")
            arms[current] = {"file": "hierarchy_authored.s1.json",
                             "content_sha256": authored_artifact.content_hash,
                             "provenance_sha256": authored_artifact.provenance_hash,
                             "validation": authored_session.validation_report,
                             "calibration_fit": authored_session.calibration_fit,
                             "accounting": authored_backend.accounting(),
                             "semantic_review": authored_session.semantic_review}
        finally:
            authored_backend.close()

        current = "hierarchy_selected"
        selected_backend = _study_backend(protocol, current, phase="selection",
                                          allow_paid=allow_paid, backend_factory=backend_factory)
        try:
            candidate_quality = None
            chosen = source
            if protocol["selected_method"] == "authored_validation":
                candidate_guard = HierarchySplitGuard(candidate_draft, splits)
                candidate_quality, _ = evaluate_hierarchy(
                    candidate_draft, splits["validation"], selected_backend,
                    guard=candidate_guard, split="validation")
                if candidate_quality["objective"] is None:
                    raise DataError("Failed candidate has no validation quality for selection.")
                if candidate_quality["objective"] > authored_session.validation_report["objective"]:
                    chosen = candidate
                options = HierarchyCompileOptions(
                    min_calibration_samples=1 if protocol["mode"] == "mock" else 10)
            else:
                options = HierarchyCompileOptions(
                    architect="dspy", optimizer="gepa",
                    structural_rounds=protocol["optimization"]["structural_rounds"],
                    max_metric_calls=protocol["optimization"]["max_metric_calls"],
                    seed=protocol["analysis"]["selection_seed"],
                    min_calibration_samples=1 if protocol["mode"] == "mock" else 10)
            selected_compiler = HierarchyCompiler(selected_backend, teacher=teacher, options=options)
            selected_session = selected_compiler.select(chosen, **splits)
            selected_compiler.calibrate(selected_session)
            selected_artifact = selected_compiler.freeze(selected_session)
            selected_artifact.save(out / "hierarchy_selected.s1.json")
            arms[current] = {"file": "hierarchy_selected.s1.json",
                             "content_sha256": selected_artifact.content_hash,
                             "provenance_sha256": selected_artifact.provenance_hash,
                             "validation": selected_session.validation_report,
                             "candidate_validation": candidate_quality,
                             "calibration_fit": selected_session.calibration_fit,
                             "accounting": selected_backend.accounting(),
                             "teacher_accounting": teacher.accounting() if teacher else None,
                             "selected_source": "candidate" if chosen is candidate else "authored",
                             "semantic_review": {
                                 "predeclared_candidate_vs_authored": semantic_review_manifest(source, chosen),
                                 "compiler": selected_session.semantic_review}}
        finally:
            selected_backend.close()
        frozen = {"format": "systemone-hierarchy-study-frozen/v1",
                  "status": "frozen_unreviewed", "protocol": protocol,
                  "protocol_sha256": fingerprint(protocol),
                  "arms": arms, "synthetic": protocol["mode"] == "mock",
                  "test_executed": False, "deployment_approved": False,
                  "review_gates": {"source_contract": protocol["source_contract_sha256"],
                                   "semantic_review": {arm: item["semantic_review"] for arm, item in arms.items()},
                                   "graph_digests": {arm: item["content_sha256"] for arm, item in arms.items()},
                                   "data_sharing_scope": "train-only teacher examples/traces" if teacher else "none",
                                   "paid_calls_authorized_for_this_run": bool(allow_paid),
                                   "teacher_sharing_authorized_for_this_run": bool(share_feedback)},
                  "limits": protocol["budgets"],
                  "live_test_state": "unexecuted_requires_exact_frozen_review"}
        _envelope_write(out / "frozen.json", frozen)
        if protocol["mode"] == "typesafe":
            _envelope_write(out / "live-manifest.json", {
                "format": "systemone-hierarchy-live-manifest/v1",
                "study_id": protocol["study_id"], "status": "unexecuted_requires_separate_approval",
                "model": source.source.model, "backend": "typesafe-sdk/0.7.0",
                "protocol_sha256": fingerprint(protocol), "frozen_sha256": fingerprint(frozen),
                "source_contract_sha256": protocol["source_contract_sha256"],
                "dataset_hashes": {name: entry["content_sha256"]
                                   for name, entry in protocol["datasets"].items()},
                "test_independence": protocol["data_attestation"],
                "frozen_artifacts": {arm: {"content_sha256": item["content_sha256"],
                                           "provenance_sha256": item["provenance_sha256"]}
                                     for arm, item in arms.items()},
                "semantic_review": frozen["review_gates"]["semantic_review"],
                "data_sharing_scope": frozen["review_gates"]["data_sharing_scope"],
                "teacher_model": protocol["optimization"]["teacher_model"],
                "teacher_max_tokens": protocol["optimization"]["teacher_max_tokens"],
                "call_and_spend_limits": protocol["budgets"],
                "approval": {"paid_test_calls": False, "semantic_review": False,
                             "external_billing_cap_verified": False},
                "deployment_approved": False})
        return frozen
    except BaseException as exc:
        atomic_json(out / "selection-failure.json", {"arm": current, "exception_type": type(exc).__name__,
                                                      "completed_arms": list(arms),
                                                      "synthetic": protocol["mode"] == "mock",
                                                      "test_executed": False})
        raise


def load_frozen(directory: str | Path) -> tuple[dict[str, Any], dict[str, Program | HierarchyArtifact]]:
    directory = Path(directory)
    frozen = _envelope_read(directory / "frozen.json")
    if frozen.get("format") != "systemone-hierarchy-study-frozen/v1" or (
        frozen.get("protocol_sha256") != fingerprint(frozen.get("protocol"))
    ):
        raise DataError("Frozen hierarchy study identity differs.")
    protocol = frozen["protocol"]
    if (frozen.get("status") != "frozen_unreviewed" or frozen.get("test_executed") is not False or
        frozen.get("deployment_approved") is not False or
        frozen.get("synthetic") != (protocol.get("mode") == "mock") or
        frozen.get("limits") != protocol.get("budgets") or
        frozen.get("review_gates", {}).get("graph_digests") != {
            arm: entry["content_sha256"] for arm, entry in frozen.get("arms", {}).items()
        }):
        raise DataError("Frozen study mode, review gates, or limits differ from registration.")
    # Verify the independently registered source and all four datasets again.
    paths = {key: Path(value["path"]) for key, value in protocol["source"].items()}
    split_paths = {name: Path(value["path"]) for name, value in protocol["datasets"].items()}
    if any(_file_sha256(paths[key]) != protocol["source"][key]["file_sha256"] for key in paths) or any(
        _file_sha256(split_paths[name]) != protocol["datasets"][name]["file_sha256"] for name in SPLITS
    ):
        raise DataError("Preregistered source or dataset bytes changed after freeze.")
    source, candidate, flat, splits, _, _ = _inputs(
        paths["source"], paths["candidate"], paths["flat"], split_paths)
    if (fingerprint(source.source.model_dump(mode="json")) != protocol["source_contract_sha256"] or
        any(dataset_hash(splits[name]) != protocol["datasets"][name]["content_sha256"] for name in SPLITS)):
        raise DataError("Frozen study data changed after registration.")
    programs: dict[str, Program | HierarchyArtifact] = {}
    if set(frozen["arms"]) != set(ARMS):
        raise DataError("Frozen study is missing an arm.")
    for arm, entry in frozen["arms"].items():
        if entry.get("file") != f"{arm}.s1.json":
            raise DataError("Frozen arm artifact path differs from the fixed study layout.")
        artifact = (Program.load(directory / entry["file"]) if arm == "flat_authored" else
                    HierarchyArtifact.load(directory / entry["file"]))
        digest = (fingerprint(artifact.provenance) if arm == "flat_authored" else artifact.provenance_hash)
        if artifact.content_hash != entry["content_sha256"] or digest != entry["provenance_sha256"]:
            raise DataError(f"Frozen {arm} artifact or provenance changed.")
        if arm == "flat_authored":
            assert_contract(artifact, source.source)
        elif artifact.source != source.source or artifact.source.model != source.source.model:
            raise DataError(f"Frozen {arm} root contract or model changed.")
        programs[arm] = artifact
    return frozen, programs


def _require_live_test_approval(frozen: dict[str, Any], *, allow_paid: bool,
                                reviewed_frozen_sha256: str | None,
                                semantic_review_approved: bool) -> None:
    if frozen["protocol"]["mode"] == "typesafe" and (
        not allow_paid or not semantic_review_approved or
        reviewed_frozen_sha256 != fingerprint(frozen)
    ):
        raise ConfigurationError("Live test needs paid approval and exact frozen semantic review digest.")


def _study_schedule(protocol: dict[str, Any], rows: list) -> list[dict[str, str]]:
    rng = random.Random(protocol["analysis"]["seed"])
    schedule = []
    for row in rows:
        arms = list(ARMS)
        rng.shuffle(arms)
        for arm in arms:
            schedule.append({"arm": arm, "id": row.id,
                             "key": fingerprint({"arm": arm, "id": row.id})})
    return schedule


def _record_path(directory: Path, item: dict[str, str]) -> Path:
    return directory / "records" / item["arm"] / f"{item['key']}.json"


def _record_identity(item: dict[str, str], row, frozen: dict[str, Any]) -> dict[str, Any]:
    return {"arm": item["arm"], "id": row.id, "group": row.group,
            "gold_sha256": fingerprint(row.expected),
            "frozen_sha256": fingerprint(frozen),
            "artifact_sha256": frozen["arms"][item["arm"]]["content_sha256"]}


def _read_record(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    record = _envelope_read(path)
    if record.get("identity") != identity or not isinstance(record.get("result"), dict) or (
        type(record.get("native_attempts")) is not int or record["native_attempts"] < 0
    ):
        raise DataError("Hierarchy study row identity or native ledger differs.")
    return record


def _execute_record(directory: Path, item: dict[str, str], row, frozen: dict[str, Any],
                    program: Program | HierarchyArtifact, backend: ManagedBackend,
                    prior_used: int, guard: HierarchySplitGuard | None) -> dict[str, Any]:
    """Persist a result or leave a durable uncertain attempt; never retry a flat unknown."""
    path = _record_path(directory, item)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = _record_identity(item, row, frozen)
    if path.exists():
        return _read_record(path, identity)
    if backend.budget.used != prior_used:
        raise DataError("Study owner budget differs from completed row receipts.")
    if item["arm"] == "flat_authored":
        marker = path.with_suffix(".started.json")
        if marker.exists():
            started = _envelope_read(marker)
            if started != {"identity": identity, "owner_start_used": prior_used}:
                raise DataError("Flat in-flight attempt identity changed.")
            raise DataError("Flat attempt may have reached the provider; automatic retry is unsafe.")
        _envelope_write(marker, {"identity": identity, "owner_start_used": prior_used})
        result = Runtime(program, backend, enforce_release=False).run(row.state)
        attempts = backend.budget.used - prior_used
        if attempts != 1:
            raise DataError("Flat row did not consume exactly one native attempt.")
    else:
        evidence_dir = directory / "evidence" / item["arm"] / item["key"]
        evidence_dir.parent.mkdir(parents=True, exist_ok=True)
        mode = "resume" if evidence_dir.exists() else "create"
        if mode == "resume":
            envelope = _envelope_read(evidence_dir / "manifest.json")
            if envelope.get("owner_start_used") != prior_used:
                raise DataError("Graph evidence owner budget start differs from prior rows.")
            backend.budget.used = 0  # HierarchyEvidence restores the exact persisted owner count.
        result = HierarchyRuntime(program, backend, enforce_release=False).run(
            row.state, lineage=guard.bind(program, "test", row),
            evidence_dir=evidence_dir, evidence_mode=mode)
        attempts = backend.budget.used - prior_used
        if result["status"] in {"failed", "cancelled"}:
            raise DataError(f"Graph root {row.id} {result['status']}; retain evidence for reconciliation.")
    record = {"identity": identity, "result": result, "native_attempts": attempts,
              "backend": backend.identity, "synthetic": backend.synthetic}
    _envelope_write(path, record)
    return record


def test_frozen(frozen_dir: str | Path, out: str | Path, *, resume: bool = False,
                allow_paid: bool = False, reviewed_frozen_sha256: str | None = None,
                semantic_review_approved: bool = False, backend_factory=None) -> dict[str, Any]:
    """Run or resume only missing held-out arm/root pairs under one owner lock."""
    frozen_dir, out = Path(frozen_dir), Path(out)
    frozen, programs = load_frozen(frozen_dir)
    protocol = frozen["protocol"]
    _require_live_test_approval(frozen, allow_paid=allow_paid,
                                reviewed_frozen_sha256=reviewed_frozen_sha256,
                                semantic_review_approved=semantic_review_approved)
    rows = read_hierarchy_jsonl(Path(protocol["datasets"]["test"]["path"]),
                                programs["hierarchy_authored"])
    split_paths = {name: Path(entry["path"]) for name, entry in protocol["datasets"].items()}
    splits = {name: read_hierarchy_jsonl(split_paths[name], programs["hierarchy_authored"])
              for name in SPLITS}
    guards = {arm: HierarchySplitGuard(programs[arm], splits) for arm in ARMS if arm != "flat_authored"}
    schedule = _study_schedule(protocol, rows)
    start = {"format": "systemone-hierarchy-study-test-start/v1",
             "frozen_sha256": fingerprint(frozen), "study_id": protocol["study_id"],
             "schedule": schedule, "synthetic": frozen["synthetic"],
             "test_rows_sha256": dataset_hash(rows)}
    if resume:
        if not out.is_dir() or _envelope_read(out / "test-started.json") != start:
            raise DataError("Resume target or frozen test schedule differs.")
    else:
        if out.exists():
            raise ConfigurationError("Choose a fresh held-out directory or explicitly resume the same one.")
        out.mkdir(parents=True, exist_ok=False)
        _envelope_write(out / "test-started.json", start)
    from .hierarchy_evidence import _lock, _unlock
    with (out / ".owner.lock").open("a+b") as owner:
        _lock(owner)
        backends = {}
        try:
            backends = {arm: _study_backend(protocol, arm, phase="test", allow_paid=allow_paid,
                                            backend_factory=backend_factory) for arm in ARMS}
            used = {arm: 0 for arm in ARMS}
            by_id = {row.id: row for row in rows}
            for item in schedule:
                arm = item["arm"]
                record = _execute_record(out, item, by_id[item["id"]], frozen,
                                         programs[arm], backends[arm], used[arm], guards.get(arm))
                used[arm] += record["native_attempts"]
                backends[arm].budget.used = used[arm]
            if any(used[arm] > protocol["budgets"]["by_arm"][arm]["allocated_test_ceiling"] for arm in ARMS):
                raise DataError("Held-out attempts exceeded a preregistered arm ceiling.")
            execution = {"format": "systemone-hierarchy-study-execution/v1",
                         "status": "synthetic" if frozen["synthetic"] else "measured",
                         "frozen_sha256": fingerprint(frozen), "schedule_sha256": fingerprint(schedule),
                         "complete_records": len(schedule), "expected_records": len(schedule),
                         "native_attempts_by_arm": used,
                         "native_attempt_ceiling_by_arm": {arm: protocol["budgets"]["by_arm"][arm]["allocated_test_ceiling"]
                                                           for arm in ARMS},
                         "synthetic": frozen["synthetic"], "deployment_approved": False}
            _envelope_write(out / "execution.json", execution)
        finally:
            for backend in backends.values():
                backend.close()
            _unlock(owner)
    return report_study(frozen_dir, out)


class _ReplayOnlyBackend:
    def __init__(self, identity: str, synthetic: bool):
        self.identity, self.synthetic = identity, synthetic

    def evaluate(self, *_args, **_kwargs):
        raise DataError("Offline study replay attempted a provider call.")

    def close(self):
        pass


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _paired_clusters(rows: list, left: dict[str, float], right: dict[str, float], *,
                     seed: int, bootstrap: int, permutations: int) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(row.group or row.id, []).append(left[row.id] - right[row.id])
    if len(groups) < 2:
        raise DataError("Paired cluster inference needs at least two independent root groups.")
    cluster = [(sum(values), len(values)) for _, values in sorted(groups.items())]
    observed = sum(total for total, _ in cluster) / sum(count for _, count in cluster)
    rng = random.Random(seed)
    boot = []
    for _ in range(bootstrap):
        selected = [cluster[rng.randrange(len(cluster))] for _ in cluster]
        boot.append(sum(total for total, _ in selected) / sum(count for _, count in selected))
    null = []
    denominator = sum(count for _, count in cluster)
    for _ in range(permutations):
        null.append(sum((1 if rng.getrandbits(1) else -1) * total for total, _ in cluster) / denominator)
    return {"mean_root_utility_delta": observed,
            "cluster_bootstrap_ci_95": [_quantile(boot, 0.025), _quantile(boot, 0.975)],
            "cluster_bootstrap_bonferroni_ci_95": [_quantile(boot, 0.0125), _quantile(boot, 0.9875)],
            "two_sided_group_sign_flip_p": (1 + sum(abs(value) >= abs(observed) - 1e-12 for value in null)) /
                                           (permutations + 1),
            "n_groups": len(groups), "n_roots": len(rows), "bootstrap_replicates": bootstrap,
            "randomization_replicates": permutations,
            "note": "Paired root utility; resample or sign-flip whole groups. No inference over new seeds."}


def report_study(frozen_dir: str | Path, execution_dir: str | Path) -> dict[str, Any]:
    """Recompute every metric from complete, checksum-bound, offline-replayed rows."""
    frozen_dir, execution_dir = Path(frozen_dir), Path(execution_dir)
    frozen, programs = load_frozen(frozen_dir)
    protocol = frozen["protocol"]
    rows = read_hierarchy_jsonl(Path(protocol["datasets"]["test"]["path"]),
                                programs["hierarchy_authored"])
    splits = {name: read_hierarchy_jsonl(Path(protocol["datasets"][name]["path"]),
                                         programs["hierarchy_authored"]) for name in SPLITS}
    schedule = _study_schedule(protocol, rows)
    expected_start = {"format": "systemone-hierarchy-study-test-start/v1",
                      "frozen_sha256": fingerprint(frozen), "study_id": protocol["study_id"],
                      "schedule": schedule, "synthetic": frozen["synthetic"],
                      "test_rows_sha256": dataset_hash(rows)}
    if _envelope_read(execution_dir / "test-started.json") != expected_start:
        raise DataError("Held-out schedule or frozen study changed.")
    execution = _envelope_read(execution_dir / "execution.json")
    if (execution.get("format") != "systemone-hierarchy-study-execution/v1" or
        execution.get("frozen_sha256") != fingerprint(frozen) or
        execution.get("schedule_sha256") != fingerprint(schedule) or
        execution.get("status") != ("synthetic" if frozen["synthetic"] else "measured") or
        execution.get("complete_records") != len(schedule) or
        execution.get("expected_records") != len(schedule)):
        raise DataError("Held-out execution is incomplete or has different provenance.")
    expected_files = {_record_path(execution_dir, item).resolve() for item in schedule}
    actual_files = {path.resolve() for path in (execution_dir / "records").glob("*/*.json")
                    if not path.name.endswith(".started.json")}
    if actual_files != expected_files:
        raise DataError("Held-out result set contains missing or extra arm/root records.")
    by_id = {row.id: row for row in rows}
    results: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    attempts = {arm: 0 for arm in ARMS}
    guards = {arm: HierarchySplitGuard(programs[arm], splits) for arm in ARMS if arm != "flat_authored"}
    for item in schedule:
        arm, row = item["arm"], by_id[item["id"]]
        record = _read_record(_record_path(execution_dir, item), _record_identity(item, row, frozen))
        program = programs[arm]
        if record["synthetic"] != frozen["synthetic"] or (
            record["backend"] != ("mock-lexical/v1" if frozen["synthetic"] else "typesafe-sdk/0.7.0")
        ):
            raise DataError("Held-out backend mode or identity differs.")
        result = record["result"]
        if arm == "flat_authored":
            if (result.get("program_sha256") != program.content_hash or
                result.get("model") != program.model or result.get("synthetic") != frozen["synthetic"] or
                record["native_attempts"] != 1):
                raise DataError("Flat held-out result identity or native attempt count differs.")
            normalize_answers(program, Response(result["answers"], result["model"],
                                                synthetic=frozen["synthetic"]))
            if apply_policy(program, result["answers"]) != result["decisions"]:
                raise DataError("Flat frozen decisions do not replay from typed answers.")
        else:
            replay_backend = ManagedBackend(_ReplayOnlyBackend(record["backend"], frozen["synthetic"]),
                                            max_calls=protocol["budgets"]["by_arm"][arm]["allocated_test_ceiling"])
            try:
                replay = HierarchyRuntime(program, replay_backend, enforce_release=False).run(
                    row.state, lineage=guards[arm].bind(program, "test", row),
                    evidence_dir=execution_dir / "evidence" / arm / item["key"], evidence_mode="replay")
            finally:
                replay_backend.close()
            if replay != result or result["status"] in {"failed", "cancelled"} or (
                record["native_attempts"] != result["accounting"]["requests_attempted"]
            ):
                raise DataError("Graph row differs from strict durable evidence replay.")
        results[arm][row.id] = result
        attempts[arm] += record["native_attempts"]
    if attempts != execution["native_attempts_by_arm"] or any(
        attempts[arm] > protocol["budgets"]["by_arm"][arm]["allocated_test_ceiling"] for arm in ARMS
    ):
        raise DataError("Native attempt ledger does not reconcile with the frozen test plan.")
    ordered = {arm: [results[arm][row.id] for row in rows] for arm in ARMS}
    flat_report = report_from_results(programs["flat_authored"], rows, ordered["flat_authored"])
    graph_reports = {arm: report_from_hierarchy_results(programs[arm], rows, ordered[arm])
                     for arm in ARMS if arm != "flat_authored"}
    if any(report["objective"] is None for report in graph_reports.values()):
        raise DataError("Failed graph root cannot enter a study quality report.")
    utility = {"flat_authored": {row.id: _flat_quality(protocol_source(programs), row, result)
                                 for row, result in zip(rows, ordered["flat_authored"])}}
    utility.update({arm: report["quality_by_root_id"] for arm, report in graph_reports.items()})
    analysis = protocol["analysis"]
    comparisons = {right: _paired_clusters(
        rows, utility["hierarchy_selected"], utility[right], seed=analysis["seed"],
        bootstrap=analysis["bootstrap_replicates"], permutations=analysis["randomization_replicates"])
        for right in ("flat_authored", "hierarchy_authored")}
    from .research_stats import holm
    adjusted = holm({key: item["two_sided_group_sign_flip_p"] for key, item in comparisons.items()})
    for key, value in adjusted.items():
        comparisons[key]["holm_adjusted_p"] = value
    exploratory_authored_vs_flat = _paired_clusters(
        rows, utility["hierarchy_authored"], utility["flat_authored"], seed=analysis["seed"],
        bootstrap=analysis["bootstrap_replicates"], permutations=analysis["randomization_replicates"])
    exploratory_authored_vs_flat["status"] = "exploratory_not_in_primary_family"
    arms = {"flat_authored": {"metrics": flat_report, "root_utility": utility["flat_authored"],
                              "native_attempts": attempts["flat_authored"],
                              "unknown_usage_calls": sum(
                                  result["usage"].get("input_tokens") is None or
                                  result["usage"].get("output_tokens") is None
                                  for result in ordered["flat_authored"]),
                              "teacher_ledger": frozen["arms"]["flat_authored"].get("teacher_accounting")}}
    for arm, metrics in graph_reports.items():
        arms[arm] = {"metrics": metrics, "root_utility": utility[arm],
                     "native_attempts": attempts[arm],
                     "unknown_usage_calls": metrics["usage"]["usage_unknown_calls"],
                     "teacher_ledger": frozen["arms"][arm].get("teacher_accounting")}
    report = {"format": "systemone-hierarchy-study-report/v1", "study_id": protocol["study_id"],
              "status": "synthetic" if frozen["synthetic"] else "measured",
              "synthetic": frozen["synthetic"], "deployment_approved": False,
              "protocol_sha256": fingerprint(protocol), "frozen_sha256": fingerprint(frozen),
              "n_root": len(rows), "n_groups": len({row.group or row.id for row in rows}),
              "primary": analysis["primary"], "arms": arms, "comparisons": comparisons,
              "ablations": {"authored_hierarchy_vs_flat": exploratory_authored_vs_flat,
                            "route_and_review": {arm: graph_reports[arm]["routes"]
                                                 for arm in graph_reports},
                            "native_call_delta_selected_minus_flat": (
                                attempts["hierarchy_selected"] - attempts["flat_authored"]),
                            "native_call_delta_selected_minus_authored": (
                                attempts["hierarchy_selected"] - attempts["hierarchy_authored"]),
                            "matched_supervision": True,
                            "unmatched_realized_calls": len(set(attempts.values())) > 1},
              "budgets": protocol["budgets"], "selection_ledgers": {
                  arm: entry["accounting"] for arm, entry in frozen["arms"].items()},
              "limitations": ["Synthetic runs cannot support a Jev gain claim." if frozen["synthetic"] else
                              "Measured effects require independent label, semantic, and billing review.",
                              "Equal hard ceilings do not imply matched search, realized calls, tokens, or spend.",
                              "Billed dollars remain unknown without external provider accounting."]}
    atomic_json(execution_dir / "report.json", report)
    (execution_dir / "report.md").write_text(
        "# Hierarchy comparison\n\n" +
        ("**SYNTHETIC SOFTWARE CHECK — NOT JEV QUALITY EVIDENCE.**\n\n" if frozen["synthetic"] else
         "**Measured frozen executions; no automatic gain or deployment claim.**\n\n") +
        f"Study `{protocol['study_id']}`; {len(rows)} paired roots in {report['n_groups']} groups. "
        "See `report.json` for route coverage, quality, costs, and multiplicity-adjusted comparisons.\n",
        encoding="utf-8")
    return report


def protocol_source(programs: dict[str, Program | HierarchyArtifact]):
    return programs["hierarchy_authored"].source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="s1-study", description="Preregister and replay hierarchy research.")
    sub = parser.add_subparsers(dest="command", required=True)
    registration = sub.add_parser("register", help="Pin a new study without model calls.")
    registration.add_argument("--study-id", required=True)
    registration.add_argument("--source", type=Path, required=True)
    registration.add_argument("--candidate", type=Path, required=True)
    registration.add_argument("--flat", type=Path, required=True)
    for split in SPLITS:
        registration.add_argument(f"--{split}", type=Path, required=True)
    registration.add_argument("--out", type=Path, required=True)
    registration.add_argument("--mode", choices=["mock", "typesafe"], default="mock")
    registration.add_argument("--selected-method", choices=["authored_validation", "dspy_gepa"],
                              default="authored_validation")
    registration.add_argument("--seed", type=int, default=7)
    registration.add_argument("--analysis-seed", type=int, default=90210)
    registration.add_argument("--bootstrap-replicates", type=int, default=1000)
    registration.add_argument("--randomization-replicates", type=int, default=2000)
    registration.add_argument("--structural-rounds", type=int, default=0)
    registration.add_argument("--max-metric-calls", type=int, default=0)
    registration.add_argument("--teacher-max-calls", type=int, default=0)
    registration.add_argument("--teacher-max-tokens", type=int, default=4096)
    registration.add_argument("--teacher-model")
    registration.add_argument("--data-attestation", type=Path)
    registration.add_argument("--provider-call-price-cap-usd", type=float)
    registration.add_argument("--teacher-call-price-cap-usd", type=float)
    registration.add_argument("--external-billing-cap-usd", type=float)
    selection = sub.add_parser("select", help="Select, calibrate, and freeze; no held-out calls.")
    selection.add_argument("--protocol", type=Path, required=True)
    selection.add_argument("--out", type=Path, required=True)
    selection.add_argument("--allow-paid", action="store_true")
    selection.add_argument("--share-feedback", action="store_true")
    selection.add_argument("--approved-protocol-sha256")
    test = sub.add_parser("test", help="Run only a frozen study's held-out roots.")
    test.add_argument("--frozen", type=Path, required=True)
    test.add_argument("--out", type=Path, required=True)
    test.add_argument("--resume", action="store_true")
    test.add_argument("--allow-paid", action="store_true")
    test.add_argument("--reviewed-frozen-sha256")
    test.add_argument("--semantic-review-approved", action="store_true")
    reporting = sub.add_parser("report", help="Strict offline replay and complete-report validation.")
    reporting.add_argument("--frozen", type=Path, required=True)
    reporting.add_argument("--execution", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            protocol = register(args.source, args.candidate, args.flat,
                                {name: getattr(args, name) for name in SPLITS}, args.out,
                                study_id=args.study_id, selected_method=args.selected_method,
                                mode=args.mode, seed=args.seed, analysis_seed=args.analysis_seed,
                                structural_rounds=args.structural_rounds,
                                max_metric_calls=args.max_metric_calls,
                                teacher_max_calls=args.teacher_max_calls,
                                teacher_model=args.teacher_model,
                                teacher_max_tokens=args.teacher_max_tokens,
                                bootstrap_replicates=args.bootstrap_replicates,
                                randomization_replicates=args.randomization_replicates,
                                data_attestation=(load_document(args.data_attestation)
                                                  if args.data_attestation else None),
                                provider_call_price_cap_usd=args.provider_call_price_cap_usd,
                                teacher_call_price_cap_usd=args.teacher_call_price_cap_usd,
                                external_billing_cap_usd=args.external_billing_cap_usd)
            print(json.dumps({"status": protocol["status"], "protocol_sha256": fingerprint(protocol),
                              "native_attempt_ceiling_total": protocol["budgets"]["native_attempt_ceiling_total"]}))
        elif args.command == "select":
            frozen = select_and_freeze(args.protocol, args.out, allow_paid=args.allow_paid,
                                       share_feedback=args.share_feedback,
                                       approved_protocol_sha256=args.approved_protocol_sha256)
            print(json.dumps({"status": frozen["status"], "frozen_sha256": fingerprint(frozen),
                              "test_executed": False}))
        elif args.command == "test":
            report = test_frozen(args.frozen, args.out, resume=args.resume,
                                 allow_paid=args.allow_paid,
                                 reviewed_frozen_sha256=args.reviewed_frozen_sha256,
                                 semantic_review_approved=args.semantic_review_approved)
            print(json.dumps({"status": report["status"], "n_root": report["n_root"],
                              "synthetic": report["synthetic"]}))
        else:
            report = report_study(args.frozen, args.execution)
            print(json.dumps({"status": report["status"], "n_root": report["n_root"],
                              "synthetic": report["synthetic"]}))
        return 0
    except S1Error as exc:
        print(f"s1-study: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError) as exc:
        # Keep malformed data and local paths out of the default error stream.
        print(f"s1-study: invalid input or local file operation ({type(exc).__name__}).", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
