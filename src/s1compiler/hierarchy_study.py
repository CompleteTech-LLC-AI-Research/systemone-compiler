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
import os
from pathlib import Path
import random
import re
import sys
from typing import Any
from urllib.parse import urlsplit

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
TEXT_NORMALIZATION = "casefold_whitespace_v1"
LIVE_MANIFEST_PROPOSAL_FORMAT = "systemone-hierarchy-live-manifest-proposal/v1"
LIVE_MANIFEST_REVIEW_FORMAT = "systemone-hierarchy-live-manifest-review/v1"
LIVE_MANIFEST_REVIEWED_FORMAT = "systemone-hierarchy-live-manifest-reviewed/v1"
LIVE_MIN_CALIBRATION_SAMPLES = 10
FLAT_CALIBRATION_MAX_ERROR = 0.05
LIVE_REVIEW_ATTESTATIONS = (
    "dataset_labels_independently_reviewed",
    "near_duplicate_and_semantic_independence_reviewed",
    "prior_holdout_exclusion_lists_verified",
    "per_call_price_caps_verified_against_quote",
    "external_billing_cap_enforced",
    "paid_selection_calls_approved",
    "teacher_train_example_sharing_approved",
)
REDACTED_ATTESTATION_LISTS = ("prior_test_input_sha256s", "prior_test_text_sha256s")
SELECTION_LEDGER_FORMAT = "systemone-hierarchy-selection-ledger/v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_text_fingerprints(state: dict[str, Any]) -> set[str]:
    """Hash all nonempty string values, including nested declared input values."""
    found = set()
    pending = [state]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str):
            normalized = " ".join(value.casefold().split())
            if normalized:
                found.add(fingerprint(normalized))
    return found


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
    if selected_method == "dspy_gepa" and max_metric_calls < len(splits["validation"]) + 2:
        raise ConfigurationError("GEPA metric budget must exceed initial validation evaluation.")
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
        text_exclusions = attestation.get("prior_test_text_sha256s")
        if (attestation.get("label_origin") != "independent_human_reviewed" or
            not attestation.get("test_independence_evidence") or
            not attestation.get("reviewer") or
            not isinstance(exclusions, list) or not exclusions or
            any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in exclusions) or
            not isinstance(text_exclusions, list) or not text_exclusions or
            any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in text_exclusions) or
            attestation.get("prior_test_text_normalization") != TEXT_NORMALIZATION or
            selected_method != "dspy_gepa" or not teacher_model):
            raise ConfigurationError("Live registration needs reviewed independent labels, teacher, "
                                     "and exact-state plus normalized-text holdout exclusions.")
        projected = [project_state(source.source.state, row.state)
                     for split in SPLITS for row in splits[split]]
        if {fingerprint(state) for state in projected} & set(exclusions):
            raise DataError("Live hierarchy input overlaps an excluded prior-study holdout.")
        text_exclusion_set = set(text_exclusions)
        if any(_normalized_text_fingerprints(state) & text_exclusion_set for state in projected):
            raise DataError("Live hierarchy text overlaps an excluded prior-study holdout.")
        attestation = {**attestation, "prior_test_input_sha256s": None,
                       "prior_test_text_sha256s": None,
                       "prior_test_exclusion_digest": fingerprint(sorted(set(exclusions))),
                       "prior_test_exclusion_count": len(set(exclusions)),
                       "prior_test_text_exclusion_digest": fingerprint(sorted(text_exclusion_set)),
                       "prior_test_text_exclusion_count": len(text_exclusion_set)}
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


def _teacher_endpoint() -> str:
    """Endpoint from the environment, refusing URLs that could carry a credential."""
    raw = os.environ.get("S1_TEACHER_API_BASE")
    if not raw:
        return "provider_default"
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError as exc:
        raise ConfigurationError("S1_TEACHER_API_BASE is not a valid URL.") from exc
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None or
        parts.password is not None or parts.query or parts.fragment):
        raise ConfigurationError("S1_TEACHER_API_BASE must be an http(s) URL without userinfo, query, or "
                                 "fragment; put credentials only in S1_TEACHER_API_KEY.")
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"{parts.scheme}://{host}{':' + str(port) if port else ''}{parts.path}"


def _unspecified_parameters(value: Any, prefix: str = "") -> list[str]:
    if value is None:
        return [prefix or "<root>"]
    if isinstance(value, dict):
        return [path for key, item in value.items()
                for path in _unspecified_parameters(item, f"{prefix}/{key}" if prefix else str(key))]
    if isinstance(value, list):
        return [path for index, item in enumerate(value)
                for path in _unspecified_parameters(item, f"{prefix}[{index}]")]
    return []


def _live_manifest_proposal(protocol: dict[str, Any], source: HierarchySource) -> dict[str, Any]:
    """Every parameter a live run will use, derived from the protocol and current code; no calls."""
    if protocol["mode"] != "typesafe":
        raise ConfigurationError("Only a live (typesafe) protocol needs a live manifest; mock studies "
                                 "make no paid calls.")
    import inspect
    from .architect import DSPyTeacher
    teacher_defaults = inspect.signature(DSPyTeacher.__init__).parameters
    backend_defaults = inspect.signature(TypeSafeBackend.__init__).parameters
    optimization = protocol["optimization"]
    attestation = {key: value for key, value in protocol["data_attestation"].items()
                   if key not in REDACTED_ATTESTATION_LISTS}
    proposal = {
        "format": LIVE_MANIFEST_PROPOSAL_FORMAT,
        "status": "proposed_unexecuted",
        "study_id": protocol["study_id"],
        "protocol_sha256": fingerprint(protocol),
        "registered_at": protocol["registered_at"],
        "model": source.source.model,
        "source_contract_sha256": protocol["source_contract_sha256"],
        "authored_graph_sha256": protocol["authored_graph_sha256"],
        "candidate_graph_sha256": protocol["candidate_graph_sha256"],
        "flat_program_sha256": protocol["flat_program_sha256"],
        "datasets": {name: {"content_sha256": entry["content_sha256"], "n": entry["n"],
                            "n_groups": entry["n_groups"]}
                     for name, entry in protocol["datasets"].items()},
        "data_attestation": attestation,
        "data_sharing_scope": "train-only teacher examples/traces",
        "native_backend": {"identity": TypeSafeBackend.identity,
                           "sdk_retries": 0,
                           "request_timeout_s": backend_defaults["timeout"].default,
                           "credential": {"environment_variable": "TYPESAFE_API_KEY",
                                          "recorded_in_manifest": False}},
        "teacher": {"model": optimization["teacher_model"],
                    "signature_ceiling": optimization["teacher_max_calls"],
                    "provider_request_ceiling": protocol["budgets"]["teacher_provider_request_ceiling"],
                    "max_tokens": optimization["teacher_max_tokens"],
                    # The endpoint decides provider and price, so it is pinned here and
                    # re-resolved at selection; the credential itself is never recorded.
                    "api_base": _teacher_endpoint(),
                    "temperature": "provider_default",
                    "request_timeout_s": teacher_defaults["timeout"].default,
                    "max_prompt_chars": teacher_defaults["max_prompt_chars"].default,
                    "framework_caches": "disabled",
                    "sdk_retries": 0,
                    "credential": {"environment_variable": "S1_TEACHER_API_KEY",
                                   "recorded_in_manifest": False}},
        "selection": {"method": protocol["selected_method"],
                      "structural_rounds": optimization["structural_rounds"],
                      "max_metric_calls": optimization["max_metric_calls"],
                      "selection_seed": protocol["analysis"]["selection_seed"],
                      "min_calibration_samples": LIVE_MIN_CALIBRATION_SAMPLES,
                      "flat_calibration_max_error": FLAT_CALIBRATION_MAX_ERROR},
        "analysis": protocol["analysis"],
        "call_and_spend_limits": protocol["budgets"],
        "frozen_graph_digests": "assigned by live selection and recorded in frozen.json and "
                                "live-manifest.json; they need a separate semantic review before test",
        "commands": {
            "select": "s1-study select --protocol <protocol> --out <fresh directory> --allow-paid "
                      "--share-feedback --approved-protocol-sha256 " + fingerprint(protocol) +
                      " --reviewed-manifest <reviewed manifest>",
            "test": "s1-study test --frozen <selection directory> --out <fresh directory> --allow-paid "
                    "--semantic-review-approved --reviewed-frozen-sha256 <frozen_sha256 printed by select, after review>"},
        "review_requirements": list(LIVE_REVIEW_ATTESTATIONS),
        "review_template": {"format": LIVE_MANIFEST_REVIEW_FORMAT,
                            "manifest_sha256": "<sha256 printed by s1-study manifest>",
                            "reviewer": "<name or role>", "reviewed_at": "<ISO 8601 timestamp>",
                            "attestations": {name: {"attested": False, "evidence": "<record or reason>"}
                                             for name in LIVE_REVIEW_ATTESTATIONS}},
        "limitations": protocol["limitations"] + [
            "Frozen graph digests, semantic review, and paid test approval follow live selection.",
            "Credentials are resolved from the local environment at run time and never recorded."],
    }
    unspecified = _unspecified_parameters({key: value for key, value in proposal.items()
                                           if key != "review_template"})
    if unspecified:
        raise ConfigurationError("Live manifest has unspecified parameters: " + ", ".join(unspecified))
    proposal["unspecified_parameters"] = []
    return proposal


def propose_live_manifest(protocol_path: str | Path, out: str | Path) -> dict[str, Any]:
    """Write the complete pre-spend live manifest for human review; never construct a provider."""
    output = Path(out)
    if output.exists():
        raise ConfigurationError("Choose a fresh live manifest path; proposals are immutable.")
    protocol = load_protocol(protocol_path)
    paths = {key: Path(value["path"]) for key, value in protocol["source"].items()}
    split_paths = {name: Path(value["path"]) for name, value in protocol["datasets"].items()}
    source, *_ = _inputs(paths["source"], paths["candidate"], paths["flat"], split_paths)
    proposal = _live_manifest_proposal(protocol, source)
    _envelope_write(output, proposal)
    return proposal


def _validate_live_review(proposal: dict[str, Any], review: Any) -> None:
    if (not isinstance(review, dict) or review.get("format") != LIVE_MANIFEST_REVIEW_FORMAT or
        review.get("manifest_sha256") != fingerprint(proposal)):
        raise ConfigurationError("Live manifest review is missing or refers to a different manifest digest.")
    if (not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip() or
        not isinstance(review.get("reviewed_at"), str)):
        raise ConfigurationError("Live manifest review needs a reviewer and an ISO 8601 review time.")
    try:
        datetime.fromisoformat(review["reviewed_at"])
    except ValueError as exc:
        raise ConfigurationError("Live manifest review time is not ISO 8601.") from exc
    attestations = review.get("attestations")
    if not isinstance(attestations, dict) or set(attestations) != set(LIVE_REVIEW_ATTESTATIONS):
        raise ConfigurationError("Live manifest review must answer exactly the required attestations.")
    incomplete = [name for name, item in attestations.items()
                  if not isinstance(item, dict) or item.get("attested") is not True or
                  not isinstance(item.get("evidence"), str) or not item["evidence"].strip() or
                  item["evidence"].strip().startswith("<")]
    if incomplete:
        raise ConfigurationError("Live manifest review has unattested items: " + ", ".join(sorted(incomplete)))


def review_live_manifest(manifest_path: str | Path, review_path: str | Path,
                         out: str | Path) -> dict[str, Any]:
    """Bind a human review to one exact manifest digest; authorizes nothing by itself."""
    output = Path(out)
    if output.exists():
        raise ConfigurationError("Choose a fresh reviewed manifest path.")
    proposal = _envelope_read(Path(manifest_path))
    if proposal.get("format") != LIVE_MANIFEST_PROPOSAL_FORMAT or proposal.get("unspecified_parameters") != []:
        raise DataError("Unsupported or incomplete live manifest proposal.")
    review = load_document(review_path)
    _validate_live_review(proposal, review)
    reviewed = {"format": LIVE_MANIFEST_REVIEWED_FORMAT, "status": "reviewed_unexecuted",
                "study_id": proposal["study_id"], "protocol_sha256": proposal["protocol_sha256"],
                "manifest_sha256": fingerprint(proposal), "proposal": proposal, "review": review,
                "selection_executed": False, "test_executed": False, "deployment_approved": False}
    _envelope_write(output, reviewed)
    return reviewed


def _load_reviewed_manifest(path: str | Path | None, protocol: dict[str, Any],
                            source: HierarchySource) -> dict[str, Any]:
    if path is None:
        raise ConfigurationError("Live selection requires a reviewed live manifest: run "
                                 "s1-study manifest, complete its review, then s1-study review.")
    reviewed = _envelope_read(Path(path))
    if (reviewed.get("format") != LIVE_MANIFEST_REVIEWED_FORMAT or
        reviewed.get("proposal") != _live_manifest_proposal(protocol, source) or
        reviewed.get("manifest_sha256") != fingerprint(reviewed["proposal"])):
        raise ConfigurationError("Reviewed live manifest differs from the current protocol, software "
                                 "parameters, or teacher endpoint; propose and review it again.")
    _validate_live_review(reviewed["proposal"], reviewed.get("review"))
    return reviewed


def _selection_ledger_path(protocol_path: str | Path) -> Path:
    path = Path(protocol_path)
    return path.with_name(path.name + ".selection-ledger.json")


def _read_selection_ledger(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {"format": SELECTION_LEDGER_FORMAT, "protocol_sha256": fingerprint(protocol), "attempts": []}
    ledger = _envelope_read(path)
    if (ledger.get("format") != SELECTION_LEDGER_FORMAT or
        ledger.get("protocol_sha256") != fingerprint(protocol) or not isinstance(ledger.get("attempts"), list)):
        raise DataError("Selection ledger differs from this protocol.")
    return ledger


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
                      reviewed_manifest: str | Path | None = None,
                      acknowledge_prior_selection_sha256: str | None = None,
                      backend_factory=None, teacher=None) -> dict[str, Any]:
    """Select on validation, fit on calibration, then seal all arms before test."""
    protocol = load_protocol(protocol_path)
    if protocol["mode"] == "typesafe" and (
        not allow_paid or not share_feedback or approved_protocol_sha256 != fingerprint(protocol)
    ):
        raise ConfigurationError("Live selection requires separate paid, teacher-sharing, and exact "
                                 "preregistered-protocol approvals.")
    if protocol["mode"] != "typesafe" and reviewed_manifest is not None:
        raise ConfigurationError("Mock studies make no paid calls and take no live manifest.")
    out = Path(out)
    if out.exists():
        raise ConfigurationError("Choose a new selection directory; frozen study state is immutable.")
    paths = {key: Path(value["path"]) for key, value in protocol["source"].items()}
    split_paths = {name: Path(value["path"]) for name, value in protocol["datasets"].items()}
    source, candidate, flat, splits, _, candidate_draft = _inputs(
        paths["source"], paths["candidate"], paths["flat"], split_paths)
    reviewed = None
    if protocol["mode"] == "typesafe":
        # The exact reviewed parameters gate every provider or teacher construction below.
        reviewed = _load_reviewed_manifest(reviewed_manifest, protocol, source)
    ledger_path = _selection_ledger_path(protocol_path)
    ledger = None
    if protocol["mode"] == "typesafe":
        # Ceilings are per run, so earlier paid attempts must be acknowledged, never silently repeated.
        ledger = _read_selection_ledger(ledger_path, protocol)
        if ledger["attempts"] and acknowledge_prior_selection_sha256 != fingerprint(ledger):
            raise ConfigurationError(
                f"This protocol already has {len(ledger['attempts'])} live selection attempt(s), "
                f"last status {ledger['attempts'][-1]['status']}. Review {ledger_path.name} and pass "
                f"--acknowledge-prior-selection-sha256 {fingerprint(ledger)} to select again; the "
                "external billing cap must cover every attempt.")
    if protocol["selected_method"] == "dspy_gepa" and teacher is None:
        if protocol["mode"] != "typesafe":
            raise ConfigurationError("Mock GEPA study needs an explicit synthetic teacher test double.")
        from .architect import DSPyTeacher
        teacher = DSPyTeacher(protocol["optimization"]["teacher_model"], allow_paid=allow_paid,
                              share_feedback=share_feedback,
                              max_calls=protocol["optimization"]["teacher_max_calls"],
                              max_tokens=protocol["optimization"]["teacher_max_tokens"])
    min_calibration_samples = 1 if protocol["mode"] == "mock" else LIVE_MIN_CALIBRATION_SAMPLES
    out.mkdir(parents=True, exist_ok=False)
    _envelope_write(out / "selection-started.json", {
        "protocol_sha256": fingerprint(protocol), "status": "started",
        "synthetic": protocol["mode"] == "mock", "test_executed": False})
    attempt = {"out": str(out), "status": "started",
               "started_at": datetime.now(timezone.utc).isoformat()}
    if ledger is not None:
        ledger["attempts"].append(attempt)
        _envelope_write(ledger_path, ledger)
    spent: dict[str, Any] = {}
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
                flat, splits["calibration"], calibration_results, max_error=FLAT_CALIBRATION_MAX_ERROR,
                min_samples=min_calibration_samples)
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
            spent["flat_authored"] = flat_backend.accounting()
            flat_backend.close()

        current = "hierarchy_authored"
        authored_backend = _study_backend(protocol, current, phase="selection",
                                          allow_paid=allow_paid, backend_factory=backend_factory)
        try:
            authored_compiler = HierarchyCompiler(
                authored_backend, options=HierarchyCompileOptions(
                    min_calibration_samples=min_calibration_samples))
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
            spent["hierarchy_authored"] = authored_backend.accounting()
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
                options = HierarchyCompileOptions(min_calibration_samples=min_calibration_samples)
            else:
                options = HierarchyCompileOptions(
                    architect="dspy", optimizer="gepa",
                    structural_rounds=protocol["optimization"]["structural_rounds"],
                    max_metric_calls=protocol["optimization"]["max_metric_calls"],
                    seed=protocol["analysis"]["selection_seed"],
                    min_calibration_samples=min_calibration_samples)
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
            spent["hierarchy_selected"] = selected_backend.accounting()
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
                                   "teacher_sharing_authorized_for_this_run": bool(share_feedback),
                                   "reviewed_manifest_sha256": fingerprint(reviewed) if reviewed else None},
                  "limits": protocol["budgets"],
                  "live_test_state": "unexecuted_requires_exact_frozen_review"}
        _envelope_write(out / "frozen.json", frozen)
        if reviewed is not None:
            _envelope_write(out / "live-manifest.json", _live_manifest_after_freeze(frozen, reviewed))
        if ledger is not None:
            attempt.update(status="frozen", accounting=spent,
                           teacher_accounting=teacher.accounting() if teacher else None)
            _envelope_write(ledger_path, ledger)
        return frozen
    except BaseException as exc:
        try:
            teacher_accounting = teacher.accounting() if teacher is not None else None
        except Exception:  # A test double or broken teacher must not hide the original failure.
            teacher_accounting = None
        atomic_json(out / "selection-failure.json", {"arm": current, "exception_type": type(exc).__name__,
                                                      "completed_arms": list(arms),
                                                      "accounting": spent,
                                                      "teacher_accounting": teacher_accounting,
                                                      "synthetic": protocol["mode"] == "mock",
                                                      "test_executed": False})
        if ledger is not None:
            attempt.update(status="failed", failed_arm=current, exception_type=type(exc).__name__,
                           accounting=spent, teacher_accounting=teacher_accounting)
            _envelope_write(ledger_path, ledger)
        raise


def _live_manifest_after_freeze(frozen: dict[str, Any], reviewed: dict[str, Any]) -> dict[str, Any]:
    """Post-selection manifest: reviewed pre-spend parameters plus frozen digests awaiting review."""
    proposal, attested = reviewed["proposal"], reviewed["review"]["attestations"]
    return {"format": "systemone-hierarchy-live-manifest/v1",
            "study_id": proposal["study_id"], "status": "frozen_unexecuted_requires_separate_test_approval",
            "model": proposal["model"], "backend": proposal["native_backend"]["identity"],
            "protocol_sha256": frozen["protocol_sha256"], "frozen_sha256": fingerprint(frozen),
            "reviewed_manifest_sha256": fingerprint(reviewed),
            "reviewed_by": reviewed["review"]["reviewer"],
            "source_contract_sha256": proposal["source_contract_sha256"],
            "dataset_hashes": {name: entry["content_sha256"] for name, entry in proposal["datasets"].items()},
            "test_independence": proposal["data_attestation"],
            "frozen_artifacts": {arm: {"content_sha256": item["content_sha256"],
                                       "provenance_sha256": item["provenance_sha256"]}
                                 for arm, item in frozen["arms"].items()},
            "semantic_review": frozen["review_gates"]["semantic_review"],
            "data_sharing_scope": frozen["review_gates"]["data_sharing_scope"],
            "teacher": proposal["teacher"], "selection": proposal["selection"],
            "call_and_spend_limits": proposal["call_and_spend_limits"],
            "approval": {"paid_selection_calls": attested["paid_selection_calls_approved"]["attested"],
                         "teacher_example_sharing": attested["teacher_train_example_sharing_approved"]["attested"],
                         "external_billing_cap_verified": attested["external_billing_cap_enforced"]["attested"],
                         "paid_test_calls": False, "semantic_review": False},
            "test_executed": False, "deployment_approved": False}


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
        try:
            result = Runtime(program, backend, enforce_release=False).run(row.state)
        except BaseException:
            # The budget is reserved before dispatch, so an unchanged count proves no request left
            # this process and the row can be retried. Anything else stays uncertain.
            if backend.budget.used == prior_used:
                marker.unlink(missing_ok=True)
            raise
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


def reconcile_flat_attempt(frozen_dir: str | Path, execution_dir: str | Path, *, root_id: str,
                           reviewer: str, attestation: str) -> dict[str, Any]:
    """Record a human claim that an in-flight flat request never reached the provider.

    A hard crash can leave a started marker with no result. The marker normally blocks resume because
    the request may have been charged. This command does not prove anything: it stores who asserted
    that no request was sent, and why, then lets resume retry that one row. A wrong claim means an
    uncounted provider call, so the external billing cap must still cover it.
    """
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(attestation, str) or (
        not attestation.strip()
    ):
        raise ConfigurationError("Reconciliation needs a reviewer and the evidence for the claim.")
    frozen_dir, execution_dir = Path(frozen_dir), Path(execution_dir)
    frozen, programs = load_frozen(frozen_dir)
    protocol = frozen["protocol"]
    rows = read_hierarchy_jsonl(Path(protocol["datasets"]["test"]["path"]), programs["hierarchy_authored"])
    start = _envelope_read(execution_dir / "test-started.json")
    if start.get("frozen_sha256") != fingerprint(frozen):
        raise DataError("Execution directory belongs to a different frozen study.")
    by_id = {row.id: row for row in rows}
    if root_id not in by_id:
        raise DataError("Unknown held-out root ID.")
    item = next(entry for entry in start["schedule"] if entry["arm"] == "flat_authored" and entry["id"] == root_id)
    path = _record_path(execution_dir, item)
    marker = path.with_suffix(".started.json")
    identity = _record_identity(item, by_id[root_id], frozen)
    from .hierarchy_evidence import _lock, _unlock
    with (execution_dir / ".owner.lock").open("a+b") as owner:
        _lock(owner)
        try:
            if path.exists():
                raise DataError("This row already has a result; nothing to reconcile.")
            if not marker.exists():
                raise DataError("This row has no in-flight marker.")
            if _envelope_read(marker).get("identity") != identity:
                raise DataError("In-flight marker identity differs from the schedule.")
            number = len(list(path.parent.glob(f"{item['key']}.reconciled-*.json"))) + 1
            record = {"format": "systemone-hierarchy-study-reconciliation/v1", "identity": identity,
                      "claim": "no provider request was sent for this attempt",
                      "reviewer": reviewer.strip(), "evidence": attestation.strip(),
                      "reconciled_at": datetime.now(timezone.utc).isoformat(),
                      "marker_sha256": _file_sha256(marker), "sequence": number}
            _envelope_write(path.parent / f"{item['key']}.reconciled-{number}.json", record)
            marker.unlink()
        finally:
            _unlock(owner)
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
                    if not path.name.endswith(".started.json") and ".reconciled-" not in path.name}
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
            record["backend"] != ("mock-lexical/v1" if frozen["synthetic"] else TypeSafeBackend.identity)
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
    reconciliations = [{"arm": arm, "file": path.name, "reviewer": _envelope_read(path).get("reviewer")}
                       for arm in ARMS for path in sorted((execution_dir / "records" / arm).glob("*.reconciled-*.json"))
                       ] if (execution_dir / "records").is_dir() else []
    report = {"format": "systemone-hierarchy-study-report/v1", "study_id": protocol["study_id"],
              "reconciled_attempts": reconciliations,
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
    selection.add_argument("--acknowledge-prior-selection-sha256",
                           help="Digest of the selection ledger; required to select again after a prior live attempt.")
    selection.add_argument("--reviewed-manifest", type=Path,
                           help="Reviewed live manifest from `s1-study review`; required for live studies.")
    manifest = sub.add_parser("manifest", help="Write the complete pre-spend live manifest; no calls.")
    manifest.add_argument("--protocol", type=Path, required=True)
    manifest.add_argument("--out", type=Path, required=True)
    review = sub.add_parser("review", help="Bind a completed human review to one manifest digest.")
    review.add_argument("--manifest", type=Path, required=True)
    review.add_argument("--review", type=Path, required=True,
                        help="Completed copy of the proposal's review_template (JSON or YAML).")
    review.add_argument("--out", type=Path, required=True)
    test = sub.add_parser("test", help="Run only a frozen study's held-out roots.")
    test.add_argument("--frozen", type=Path, required=True)
    test.add_argument("--out", type=Path, required=True)
    test.add_argument("--resume", action="store_true")
    test.add_argument("--allow-paid", action="store_true")
    test.add_argument("--reviewed-frozen-sha256")
    test.add_argument("--semantic-review-approved", action="store_true")
    reconcile = sub.add_parser("reconcile", help="Record a human claim that a flat attempt was never sent.")
    reconcile.add_argument("--frozen", type=Path, required=True)
    reconcile.add_argument("--execution", type=Path, required=True)
    reconcile.add_argument("--root-id", required=True)
    reconcile.add_argument("--reviewer", required=True)
    reconcile.add_argument("--evidence", required=True, help="Why no provider request could have been sent.")
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
        elif args.command == "manifest":
            proposal = propose_live_manifest(args.protocol, args.out)
            print(json.dumps({"status": proposal["status"], "manifest_sha256": fingerprint(proposal),
                              "unspecified_parameters": proposal["unspecified_parameters"]}))
        elif args.command == "review":
            reviewed = review_live_manifest(args.manifest, args.review, args.out)
            print(json.dumps({"status": reviewed["status"],
                              "reviewed_manifest_sha256": fingerprint(reviewed),
                              "selection_executed": False, "test_executed": False}))
        elif args.command == "select":
            frozen = select_and_freeze(args.protocol, args.out, allow_paid=args.allow_paid,
                                       share_feedback=args.share_feedback,
                                       approved_protocol_sha256=args.approved_protocol_sha256,
                                       reviewed_manifest=args.reviewed_manifest,
                                       acknowledge_prior_selection_sha256=args.acknowledge_prior_selection_sha256)
            print(json.dumps({"status": frozen["status"], "frozen_sha256": fingerprint(frozen),
                              "test_executed": False}))
        elif args.command == "test":
            report = test_frozen(args.frozen, args.out, resume=args.resume,
                                 allow_paid=args.allow_paid,
                                 reviewed_frozen_sha256=args.reviewed_frozen_sha256,
                                 semantic_review_approved=args.semantic_review_approved)
            print(json.dumps({"status": report["status"], "n_root": report["n_root"],
                              "synthetic": report["synthetic"]}))
        elif args.command == "reconcile":
            record = reconcile_flat_attempt(args.frozen, args.execution, root_id=args.root_id,
                                            reviewer=args.reviewer, attestation=args.evidence)
            print(json.dumps({"status": "reconciled_unverified_claim", "sequence": record["sequence"],
                              "claim": record["claim"]}))
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
