"""Read-only export of verified prior flat test-input exclusion fingerprints."""
from __future__ import annotations

import argparse
from pathlib import Path

from .errors import ConfigurationError, DataError
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
    atomic_json(out / "prior-test-input-exclusions.json", input_export)
    atomic_json(out / "prior-test-text-exclusions.json", text_export)
    return {"tasks": len(inputs), "test_rows": sum(task["n"] for task in inputs.values()),
            "input_export_sha256": fingerprint(input_export), "text_export_sha256": fingerprint(text_export)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Relocated verified dataset root, containing task directories.")
    parser.add_argument("--out", type=Path, required=True, help="Fresh local sensitive output directory.")
    args = parser.parse_args(argv)
    # Print aggregate counts/digests only, never individual low-entropy hashes.
    print(export(args.protocol, args.out, data_root=args.data_root))


if __name__ == "__main__":
    main()
