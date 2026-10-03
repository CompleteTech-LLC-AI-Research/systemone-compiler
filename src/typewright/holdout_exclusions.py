"""Read-only export of verified prior flat test-input exclusion fingerprints."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

from .errors import ConfigurationError, DataError, S1Error
from .hierarchy_study import TEXT_NORMALIZATION, _normalized_text_fingerprints
from .io import atomic_json, fingerprint, load_document
from .models import project_state
from .research_data import TASKS, load_dataset


def export(protocol_path: Path, out: Path, *, data_root: Path | None = None) -> dict:
    """Verify all registered bytes before writing a fresh sensitive directory.

    Archived implementation hashes are retained as provenance, not compared to
    today's code: this exports inputs, rather than resuming the old experiment.
    Labels are validated by the dataset reader but are never exported or sent.
    """
    if out.exists():
        raise ConfigurationError("Choose a fresh holdout exclusion directory.")
    envelope = load_document(protocol_path)
    if not isinstance(envelope, dict) or set(envelope) != {"content", "sha256"}:
        raise DataError("Malformed prior research protocol envelope.")
    protocol = envelope["content"]
    if envelope["sha256"] != fingerprint(protocol) or not isinstance(protocol, dict) or (
        protocol.get("format") != "jev-research-protocol/v1"
    ):
        raise DataError("Prior research protocol checksum or format differs.")
    registered = protocol.get("datasets")
    if not isinstance(registered, dict) or not registered or set(registered) - set(TASKS):
        raise DataError("Prior research protocol has missing or unknown tasks.")
    inputs, texts = {}, {}
    for task, entry in sorted(registered.items()):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise DataError("Malformed registered dataset entry.")
        path = data_root / task if data_root is not None else Path(entry["path"])
        source, splits, manifest, _ = load_dataset(path)
        digest = fingerprint(manifest)
        if manifest["task"] != task or digest != entry.get("manifest_sha256") or (
            manifest != entry.get("manifest")
        ):
            raise DataError("Prior registered dataset manifest differs.")
        rows = splits["test"]
        if not rows:
            raise DataError("Prior registered test split is empty.")
        states = [project_state(source.state, row.state) for row in rows]
        inputs[task] = {"n": len(rows), "manifest_sha256": digest,
                        "declared_input_fields": sorted(source.state),
                        "projected_input_sha256s": sorted({fingerprint(state) for state in states})}
        texts[task] = {"n": len(rows), "manifest_sha256": digest,
                       "normalized_text_sha256s": sorted(set().union(
                           *(_normalized_text_fingerprints(state) for state in states)))}
    shared = {"source_protocol_sha256": envelope["sha256"],
              "sensitive": True, "human_independence_review_required": True,
              "labels_predictions_results_exported": False}
    input_export = {"format": "systemone-prior-flat-test-input-exclusions/v1", **shared, "tasks": inputs}
    text_export = {"format": "systemone-prior-flat-test-text-exclusions/v1", **shared,
                   "normalization": TEXT_NORMALIZATION, "tasks": texts}
    # No source directory is modified; mkdir also refuses a racing existing output.
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        atomic_json(out / "prior-test-input-exclusions.json", input_export)
        atomic_json(out / "prior-test-text-exclusions.json", text_export)
    except BaseException:
        # Never leave a half-written sensitive directory that also blocks a clean retry.
        shutil.rmtree(out, ignore_errors=True)
        raise
    return {"tasks": len(inputs), "test_rows": sum(task["n"] for task in inputs.values()),
            "input_export_sha256": fingerprint(input_export), "text_export_sha256": fingerprint(text_export)}


def read_holdout_exclusions(input_path: Path, text_path: Path, task: str) -> dict:
    """Validate one export pair for live registration without raw row data."""
    import re

    inputs, texts = load_document(input_path), load_document(text_path)
    try:
        if (inputs["format"] != "systemone-prior-flat-test-input-exclusions/v1" or
                texts["format"] != "systemone-prior-flat-test-text-exclusions/v1" or
                inputs["source_protocol_sha256"] != texts["source_protocol_sha256"] or
                not isinstance(inputs["source_protocol_sha256"], str) or
                not re.fullmatch(r"[0-9a-f]{64}", inputs["source_protocol_sha256"]) or
                texts["normalization"] != TEXT_NORMALIZATION):
            raise ValueError
        entry, text_entry = inputs["tasks"][task], texts["tasks"][task]
        fields = entry["declared_input_fields"]
        exact, normalized = entry["projected_input_sha256s"], text_entry["normalized_text_sha256s"]
        if (entry["manifest_sha256"] != text_entry["manifest_sha256"] or
                not isinstance(entry["manifest_sha256"], str) or
                not re.fullmatch(r"[0-9a-f]{64}", entry["manifest_sha256"]) or
                type(entry["n"]) is not int or entry["n"] < 1 or
                entry["n"] != text_entry["n"] or
                not isinstance(fields, list) or not fields or
                any(not isinstance(field, str) or not field for field in fields) or
                len(set(fields)) != len(fields) or
                any(not isinstance(values, list) or not values or any(
                    not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in values) for values in (exact, normalized))):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError("Malformed or mismatched prior-holdout export pair.") from exc
    return {"prior_test_input_sha256s": exact, "prior_test_input_fields": fields,
            "prior_test_text_sha256s": normalized,
            "prior_test_text_normalization": TEXT_NORMALIZATION}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Relocated verified dataset root, containing task directories.")
    parser.add_argument("--out", type=Path, required=True, help="Fresh local sensitive output directory.")
    args = parser.parse_args(argv)
    try:
        # Print aggregate counts/digests only, never individual low-entropy hashes.
        print(export(args.protocol, args.out, data_root=args.data_root))
    except S1Error as exc:
        # The message only: a chained validation error can embed a raw dataset row.
        print(f"holdout_exclusions: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError) as exc:
        print(f"holdout_exclusions: invalid input or local file operation ({type(exc).__name__}).",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
