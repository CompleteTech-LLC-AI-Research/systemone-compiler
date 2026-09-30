# Preregistered flat-versus-hierarchy study

`s1-study` is a separate study identity and evidence tree. It never reads or
modifies the existing flat benchmark worker, manifest, holdout, or spending
reservation. The shipped end-to-end check uses synthetic support-ticket labels
and the lexical mock. Its numerical results are **not Jev performance evidence**.

## Offline software check

From the repository root after installation, use fresh paths:

```bash
s1-study register --study-id synthetic_support_h14 \
  --source examples/hierarchy/support/source.json \
  --candidate examples/hierarchy/support/source.json \
  --flat examples/hierarchy/support/flat_baseline.s1.json \
  --train examples/hierarchy/support/train.jsonl \
  --validation examples/hierarchy/support/validation.jsonl \
  --calibration examples/hierarchy/support/calibration.jsonl \
  --test examples/hierarchy/support/test.jsonl \
  --out runs/h14-protocol.json
s1-study select --protocol runs/h14-protocol.json --out runs/h14-frozen
s1-study test --frozen runs/h14-frozen --out runs/h14-test
s1-study report --frozen runs/h14-frozen --execution runs/h14-test
```

If test execution stops, rerun the same `test` command with `--resume`. The
runner locks the study owner, verifies the frozen identity and deterministic
arm/root schedule, and skips completed records. A graph's durable stage receipts
resume through the existing hierarchy evidence reconciler. A flat request with
an in-flight marker but no result is *uncertain* and cannot be retried
automatically. This preserves request accounting instead of hiding a possible
charged call. `report` strictly replays every graph without provider calls,
recomputes flat decisions from typed answers, and refuses incomplete evidence.

## Preregistered comparison

All three arms share the same four disjoint splits, root IDs, groups, fixed
public source contract, and pinned `jev-1.13.0` model. The flat arm is authored;
the hierarchy arms are authored and validation-selected. A separate opt-in mode
uses bounded DSPy structure proposals and GEPA wording search. Train alone
supplies teacher examples/traces. Validation selects candidates, calibration
fits final review gates, and the complete artifacts freeze before test.

The primary statistic is paired mean root utility: completed Choice/Noul
correctness or normalized Score error, with review/incomplete roots scoring
zero. The two declared primary comparisons are selected hierarchy versus flat
and versus authored hierarchy. Group-paired sign-flip tests and cluster
bootstrap intervals use fixed seeds; Holm adjusts the two primary p values.
Authored hierarchy versus flat, route/path errors, coverage, native attempts,
latency, unknown usage, and review rates are secondary diagnostics. Conditional
branch Choice probabilities are not global posteriors. The three arms receive
equal *hard native-call ceilings*, but graph topology and search produce
different realized calls and token use. The report shows both estimates and
actual ledgers; it never infers billed dollars from unknown usage.

## Live proposal and review gates

A live registration needs a new, independently human-reviewed four-split
dataset, an attested test origin, a reviewer, and two SHA256 exclusion lists in
`--data-attestation`: `prior_test_input_sha256s` for whole projected states and
`prior_test_text_sha256s` for nonempty string values in prior holdout inputs.
Set `prior_test_text_normalization` to `casefold_whitespace_v1` (Unicode
casefold, then collapse whitespace). The software checks exact projected-input,
normalized-text, and cross-split group leakage, including text nested in a
declared input. It stores list digests and counts rather than the hash lists in
the protocol. Hashes can expose low-entropy values; keep the lists local. These
checks do not establish semantic independence or replace human near-duplicate
review, and the attestation itself still needs human review.
It also requires fixed teacher/model settings, native and teacher call ceilings,
per-call price caps from a verified quote, and an externally enforced billing
cap large enough for the declared worst case. These caps are assumptions until
independently verified; hashes are integrity checks, not signatures or
anonymization.

Registering a live protocol makes **zero** provider calls. Live selection then
requires separate `--allow-paid`, `--share-feedback`, and an exact
`--approved-protocol-sha256`; it creates frozen graph digests and
`live-manifest.json` with data scope, price assumptions, call ceilings, and
semantic-review changes. Live held-out test requires another explicit paid
approval, `--semantic-review-approved`, and the exact
`--reviewed-frozen-sha256`. No live study or teacher upload is authorized by
this document. The checked-in synthetic fixture is unsuitable for a live gain
claim; no independent human-labeled live manifest or measured result is bundled.

Use `s1-study --help` and each subcommand's `--help` for the exact arguments.
No automatic gain or production-readiness claim follows from a positive
comparison; it requires complete independent measured evidence and review.
