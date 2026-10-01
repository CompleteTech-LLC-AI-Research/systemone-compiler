"""Synthetic fixtures verify public-data filtering, never provider quality."""
from pathlib import Path

import pytest

from typewright.errors import DataError
from typewright.hierarchy_research_data import _read_archive, split_records, text_hash


def pair(identity, partition="train", text=None, english=None):
    common = {"id": str(identity), "partition": partition, "scenario": "alarm", "intent": "alarm_set"}
    french = {**common, "locale": "fr-FR", "utt": text or f"synthetic french {identity}",
              "judgments": [{"worker_id": str(i), "intent_score": 1,
                             "language_identification": "target"} for i in range(3)]}
    source = {**common, "locale": "en-US", "utt": english or f"synthetic english {identity}"}
    return french, source


def test_bilingual_prior_holdouts_are_excluded_from_every_partition():
    pairs = [pair(i, partition) for i, partition in enumerate(("train", "dev", "test"))]
    french, english = map(list, zip(*pairs))
    exclusions = {text_hash(english[0]["utt"]), text_hash(french[1]["utt"]), text_hash(english[2]["utt"])}
    splits, audit = split_records(french, english, exclusions)
    assert all(not rows for rows in splits.values())
    assert audit["removed"]["prior_holdout_text_or_aligned_english_overlap"] == 3


def test_duplicate_aligned_text_discards_both_rows_without_moving_test():
    first = pair(0, "train", english="same source")
    second = pair(1, "test", english=" SAME\tSOURCE ")
    third = pair(2, "test")
    splits, audit = split_records([first[0], second[0], third[0]], [first[1], second[1], third[1]], set())
    assert [row["id"] for row in splits["test"]] == ["massive_fr_2"]
    assert not splits["train"]
    assert audit["removed"]["duplicate_french_or_aligned_english_text"] == 2


def test_development_assignment_is_order_independent_and_test_stays_test():
    pairs = [pair(i, "dev" if i < 9 else "test") for i in range(10)]
    french, english = map(list, zip(*pairs))
    splits, _ = split_records(french, english, set())
    reordered, _ = split_records(list(reversed(french)), list(reversed(english)), set())
    assert reordered == splits
    assert len(splits["validation"]) + len(splits["calibration"]) == 9
    assert [row["id"] for row in splits["test"]] == ["massive_fr_9"]


def test_human_majority_and_unique_voters_are_required():
    french, english = pair(0)
    french["judgments"][0]["intent_score"] = 0
    french["judgments"][1]["intent_score"] = 2
    splits, audit = split_records([french], [english], set())
    assert all(not rows for rows in splits.values())
    assert audit["removed"]["fewer_than_two_positive_human_intent_judgments"] == 1
    french["judgments"][1]["worker_id"] = "0"
    with pytest.raises(DataError, match="repeated"):
        split_records([french], [english], set())


def test_alignment_drift_and_duplicate_id_fail_closed():
    french, english = pair(0)
    english["partition"] = "test"
    with pytest.raises(DataError, match="contract differs"):
        split_records([french], [english], set())
    with pytest.raises(DataError, match="duplicate aligned"):
        split_records([french], [english, english], set())


def test_unpinned_archive_is_rejected_before_parsing(tmp_path: Path):
    path = tmp_path / "untrusted.tar.gz"
    path.write_bytes(b"obviously fake fixture")
    with pytest.raises(DataError, match="pinned byte"):
        _read_archive(path)
