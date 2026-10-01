from __future__ import annotations
import math
from pathlib import Path
from typing import Any
from pydantic import Field

from .errors import DataError
from .io import fingerprint, json_loads
from .models import StrictModel, UseCase, project_state


class Example(StrictModel):
    id: str = Field(min_length=1)
    state: dict[str, Any]
    expected: dict[str, Any]
    group: str | None = None


def validate_example(example: Example, source: UseCase) -> None:
    project_state(source.state, example.state)
    if set(example.expected) != set(source.decisions):
        raise DataError(f"{example.id}: expected must contain every declared decision, and nothing else.")
    for key, decision in source.decisions.items():
        y = example.expected[key]
        valid = False
        if decision.type == "choice":
            valid = isinstance(y, str) and y in decision.criteria
        elif decision.type == "noul":
            valid = type(y) is bool
        else:
            valid = type(y) in (int, float) and math.isfinite(y) and 0 <= y <= len(decision.criteria) - 1
        if not valid:
            raise DataError(f"{example.id}: invalid gold label for {key} ({decision.type}).")


def read_jsonl(path: str | Path, source: UseCase) -> list[Example]:
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if len(line) > 1_000_000:
                raise DataError(f"JSONL line {line_no} exceeds the 1 MB limit.")
            try:
                row = Example.model_validate(json_loads(line))
                validate_example(row, source)
            except Exception as exc:
                # Avoid printing row content (potentially sensitive) in the default CLI.
                raise DataError(f"Invalid dataset record at line {line_no} of {Path(path).name}.") from exc
            rows.append(row)
    if not rows:
        raise DataError(f"Empty dataset: {Path(path).name}")
    return rows


def assert_disjoint(splits: dict[str, list[Example]], source: UseCase) -> None:
    """Reject reused IDs, model-visible states, and cross-split group leakage."""
    seen_ids, seen_states, seen_groups = {}, {}, {}
    for split, examples in splits.items():
        if not examples:
            raise DataError(f"Empty split: {split}")
        for row in examples:
            validate_example(row, source)
            if row.id in seen_ids:
                raise DataError(f"Duplicate example ID detected in {split} and {seen_ids[row.id]}.")
            seen_ids[row.id] = split
            digest = fingerprint(project_state(source.state, row.state))
            if digest in seen_states:
                raise DataError(f"Duplicate model-visible state in {split} and {seen_states[digest]}.")
            seen_states[digest] = split
            if row.group:
                previous = seen_groups.get(row.group)
                if previous is not None and previous != split:
                    raise DataError(f"Group leakage detected between {previous} and {split}.")
                seen_groups[row.group] = split


def dataset_hash(rows: list[Example]) -> str:
    return fingerprint([r.model_dump(mode="json") for r in rows])
