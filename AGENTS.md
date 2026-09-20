# Project instructions for coding agents

## Design invariants

- This is a compiler/runtime library. DSPy and GEPA belong to compile time.
  Importing `s1compiler` or running a frozen program must not import either.
- Target Jev's native typed API. Choice probabilities, Noul P(true), Score
  expectations, and vendor confidence are distinct values.
- Keep the source contract fixed: output names/types, Choice labels, Score scale
  lengths, allowed input fields, and pinned target model. Generated internal
  wording can change; semantic drift still needs human review.
- All generated content is validated JSON. No eval/exec, LLM-generated Python,
  dynamic shell execution, or loading untrusted pickle checkpoints.
- Only Score outputs may use the current weighted numeric composition. Normalize
  component scales, rescale outputs, and never invent joint probabilities.
- Ordinary application code owns side effects. Do not add autonomous actions to
  file-organization, moderation, PR-review, or other examples.

## Evaluation and optimization

Four disjoint splits are mandatory for compile: train, validation, calibration,
test. Only train examples/traces enter the teacher. Validation selects candidates.
Calibration fits thresholds/review gates after prompt selection. Freeze before
test. Duplicate IDs, exact projected input duplicates, and cross-split groups are
errors. Do not weaken checks to make a benchmark pass.

GEPA optimizes text components through `JevGEPAAdapter`; DSPy programs the teacher
that proposes architecture/components. Structural proposals are bounded and
validated. Provider/auth/model-drift/budget failures are not zero-quality examples:
raise a clear failure. Invalid candidate JSON is a rejected candidate.

Never label a mock score, generated label, or test double as a real Jev result.
No claims of gains without honest independent evaluation. `measured` does not
mean production approved. Never remove warnings by simply changing provenance.

## Changes and tests

Read the source and failing test before editing. Keep patches scoped, add
regression tests, and run `python -m pytest -q`. Run optional integration tests
with real dependencies when available; preserve skips and report them otherwise.
Use `python -m ruff check src tests examples` when Ruff is installed. Validate
wheel package data and no-key demo after packaging changes.

Official integration URLs and version targets are in `docs/SOURCES.md` and
`docs/INTEGRATIONS.md`. Recheck current primary docs before updating adapters.
No fully resolved dependency lock was fabricated for this source release.

## Data, secrets, and budget

No credentials in code, tests, reports, Git, or chat. Test keys must be obviously
fake fixtures. Paid calls require explicit consent; teacher example uploads need
separate sharing consent. Default to mock. Do not enable raw HTTP/SDK/DSPy prompt
logs, external experiment tracking, or disk caches without a stated need.

Cache entries are sensitive even without raw input text. Hashes are not
anonymization; checksums are not signatures. Fail on provider/model identity
mismatch. Respect request and signature-call budgets and explain their limits.
Do not silently increase spending after a failure.

## Reporting

Report executed results, unexecuted steps, and failures separately. Do not claim
that shipped CI configuration has run remotely. No live deployment or trained
weights are bundled. Read `SETUP_PROMPT.md` for the end-to-end setup workflow.
