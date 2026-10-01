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

The runner removes a flat in-flight marker itself when the owner budget proves no request
was admitted (for example a failure before dispatch). After a hard crash the marker stays and
resume refuses. If a human can show that no request was sent, `s1-study reconcile --frozen F
--execution E --root-id ID --reviewer NAME --evidence TEXT` records that unverified claim in a
`.reconciled-N.json` file next to the row, removes the marker, and lets `test --resume` retry
that one row. The command proves nothing. A wrong claim means one uncounted provider call, so
the external billing cap must still cover it. `report` lists every reconciliation.

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

For a pinned public corpus with upstream human judgments and a reproducible
four-split preparation, see [human-annotated study preparation](HIERARCHY_LIVE_PREPARATION.md).
Its pending review record cannot authorize live registration or teacher sharing.

A live registration needs a new, independently human-reviewed four-split
dataset, an attested test origin, a reviewer, and two SHA256 exclusion lists in
`--data-attestation`: `prior_test_input_sha256s` for whole projected states and
`prior_test_text_sha256s` for nonempty string values in prior holdout inputs.
Set `prior_test_text_normalization` to `casefold_whitespace_v1` (Unicode
casefold, then collapse whitespace). The software checks exact projected-input
and normalized-text overlap against every train, validation, calibration, and
test row, including text nested in a declared input. It also checks cross-split
group leakage. It stores list digests and counts rather than the hash lists in
the protocol. Hashes can expose low-entropy values; keep the lists local. These
checks do not establish semantic independence or replace human near-duplicate
review, and the attestation itself still needs human review.
It also requires fixed teacher/model settings, native and teacher call ceilings,
per-call price caps from a verified quote, and an externally enforced billing
cap large enough for the declared worst case. These caps are assumptions until
independently verified; hashes are integrity checks, not signatures or
anonymization.

Registering a live protocol makes **zero** provider calls. So does
`s1-study manifest --protocol <protocol> --out <manifest>`, which writes the
complete pre-spend live manifest: pinned Jev model, native SDK identity with
retries disabled, teacher model, endpoint, and provider ceilings, selection method, seeds,
calibration minimums, per-arm call ceilings, price caps and worst-case spend,
dataset digests, the redacted independence attestation, the exact `select` and
`test` commands, and a `review_template`. It refuses mock protocols and any
protocol with an unspecified parameter; credentials are never recorded. A human
reviewer copies the template, fills in the printed manifest digest, their name
or role, an ISO 8601 time, and evidence for every required attestation (dataset
labels, near-duplicate and semantic independence, exclusion lists, price quote,
externally enforced billing cap, paid selection, and train-only teacher
sharing), then runs `s1-study review --manifest <manifest> --review <filled
review> --out <reviewed manifest>`. Review binds the attestations to one exact
manifest digest; it rejects placeholders, missing or extra items, and any other
digest, and it authorizes nothing by itself.

Live selection then requires separate `--allow-paid`, `--share-feedback`, an
exact `--approved-protocol-sha256`, and `--reviewed-manifest`. Before any
provider or teacher object is built, it regenerates the manifest from the
protocol, current code, and the current teacher endpoint, and refuses to run if
anything differs from what was reviewed, so a live run cannot change an
unspecified parameter. Each live attempt, successful or not, is recorded with its per-arm and teacher
accounting in `<protocol>.selection-ledger.json` next to the protocol, and a failed run's
`selection-failure.json` carries the same accounting. Ceilings apply per run, so a
second live `select` on the same protocol is refused unless it passes
`--acknowledge-prior-selection-sha256` with the ledger digest the error prints; the
external billing cap must cover every attempt. A successful selection then creates
frozen graph digests and `live-manifest.json`, which records the reviewed
manifest digest and marks the held-out test and semantic review as unapproved.
Live held-out test requires another explicit paid approval,
`--semantic-review-approved`, and the exact `--reviewed-frozen-sha256`, which is
the `frozen_sha256` that `select` prints and `live-manifest.json` records, not
the file hash of `frozen.json`. The teacher endpoint must be a plain http(s) URL;
the manifest refuses one carrying userinfo, a query, or a fragment. No live
study or teacher upload is authorized by this document. The checked-in synthetic
fixture is unsuitable for a live gain claim; no independent human-labeled
dataset, completed review, or measured result is bundled.

Use `s1-study --help` and each subcommand's `--help` for the exact arguments.
No automatic gain or production-readiness claim follows from a positive
comparison; it requires complete independent measured evidence and review.
