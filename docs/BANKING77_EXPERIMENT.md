# BANKING77: DSPy prompt engineering for native Jev

This protocol tests whether a DSPy-programmed teacher with GEPA improves a fixed
Jev classifier. It is an experiment, not evidence of improvement. No teacher runs
at inference time. No application side effects, trained weights, generated Python,
or model-provider credentials are part of the artifact.

## Task and data

Input: the original request text only. Output: one native Choice named `intent`
with all 77 original BANKING77 labels, including their capitalization and
punctuation. The model is `jev-1.13.0`. There is exactly one question per request.
The baseline uses explicit, authored definitions of all labels and instructions
about transaction type, status, and requested action. Definitions are not claimed
to be benchmark-author rubrics; review them before a live study.

The [official dataset](https://github.com/PolyAI-LDN/task-specific-datasets) is
CC-BY-4.0. Cite Casanueva et al., *Efficient Intent Detection with Dual Sentence
Encoders* (2020), when publishing results. The downloader pins commit
`57ec275d8078af65b7731c2a98be812d844a6d6b` and verifies these bytes:

| File | Rows | SHA256 |
| --- | ---: | --- |
| train.csv | 10,003 | b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b |
| test.csv | 3,080 | d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d |

The default deterministic split seed is `20260919`. Per-label 60/20/20 allocation
of official training examples, with integer rounding, yields 5,969 train, 1,998
validation, and 2,036 calibration examples. Official test remains 3,080 examples.
The prepared manifest pins the source contract and all four splits. Every load
checks those hashes plus duplicate IDs, exact projected inputs, and group overlap.
The downloaded files have no exact duplicate inputs. There are no upstream
customer/conversation group IDs, so this cannot certify customer-level separation
or detect every paraphrase duplicate. Model pretraining contamination is unknown.

Only train examples and train execution traces reach the teacher. Validation
scores select candidates. Calibration fits review thresholds after selection.
All programs must be frozen before any held-out evaluation. Public data and mock
predictions do not become real Jev evidence merely by being packaged together.

## Arms and controls

| Arm | Definition |
| --- | --- |
| A | Fixed authored baseline; no teacher. Run once, shared across seed comparisons. |
| B | One DSPy rewrite of all text components, using three seeded train examples and their gold labels, without Jev predictions. No validation-based selection. Invalid JSON falls back to A and is reported. |
| C | Standalone GEPA with `JevGEPAAdapter` and the DSPy proposer; three-example train reflection minibatches, all text components editable, no structural search or merge. |
| D | Independent DSPy rewrites, each starting from A with three seeded train examples and no Jev error feedback. Select the best validation objective, with A retained on ties. |

Optimization seeds: `7`, `17`, `29`. These control local sampling/search; they do
not guarantee deterministic remote teacher generation. B is intentionally a
three-example one-pass control, not an exhaustive few-shot baseline. C and D may
edit the same component set and have the same prompt-length ceiling. Choice labels,
model, state fields, output declaration, and bindings cannot change. A human must
still review wording for semantic drift.

The common objective for C and D is `1 - multiclass Brier / 2`, matching the
compiler's existing Choice objective. Primary evaluation is test accuracy.
This measures the DSPy-teacher/GEPA workflow as a whole, not an isolated causal
effect of the DSPy library. C and D have equal budget ceilings, not necessarily
equal realized teacher tokens, proposals, or validation evaluations. GEPA screens
proposals on training minibatches before full validation; D evaluates each valid
independent proposal. Report actual usage and this methodological difference.

## Predeclared analysis

- Report every seed's accuracy, macro-F1, confusion matrix, Brier, log loss, ECE,
  latency, request counts, reported Jev token usage, and teacher signature counts.
- Compare each C seed against A, its B seed, and its D seed using paired accuracy
  differences. Use 5,000 bootstrap replicates, analysis seed `90210`, nominal 95%
  intervals, and Bonferroni percentile intervals for all `3 * number_of_seeds`
  comparisons. These approximate intervals condition on frozen prompts; they do
  not quantify variation over new optimization runs. Do not pool repeated
  predictions as independent examples or choose a winner by test score.
- A practical target is +2 percentage points over A, positive adjusted intervals
  against A and D, and consistent direction across seeds. This is a target, not a
  promised detectable effect or a power analysis. No automatic success claim is
  emitted. A one-seed run is exploratory.
- Fit a review gate separately for every arm on calibration: largest accepted set
  with <=5% empirical error and at least 30 accepted examples. If no threshold
  qualifies, use review-only. Report actual test coverage, accepted error, and its
  descriptive Wilson upper endpoint. This is not guaranteed selective risk.
- Gate on selected Choice probability. Keep vendor confidence separately; do not
  treat it as a class probability. No probability recalibration is performed.

## Budgets

Defaults: 10,000 Jev search requests per C/D seed; 30 teacher signatures per C/D
seed; one signature per B seed; 8,192 maximum output tokens per teacher call;
24,000 characters for question definitions across all arms. C stops before a
round that could exceed the request or signature ceiling and reserves its final
validation evaluation. The ManagedBackend request counter remains a hard guard.
Budget/provider/auth/model-drift failures propagate, rather than becoming zero
quality examples. Failure summaries record exception types and available counters,
not raw prompts or exception payloads. No automatic retries of failed experiments.

For the prepared full dataset and three seeds:

| Phase | Maximum requests/signatures |
| --- | ---: |
| Jev selection including calibration | 80,360 |
| Jev final test | 30,800 |
| Total Jev | 111,160 |
| DSPy teacher signatures | 183 |

At this search budget, D can evaluate only four independent rewrites per seed.
C has a different screening schedule and may use more proposals but fewer full
validation evaluations. This is a bounded pilot, not a broad prompt search.

These are NOT hard aggregate token or dollar ceilings. DSPy can make more than one
provider request per signature, and input-token usage varies. Teacher billed cost
is currently unknown in the adapter's accounting. Configure provider-side spending
limits and agree an explicit teacher/provider before a live run. The runner makes
this limitation explicit rather than fabricating a cost estimate or silently
raising a budget. Cache and GEPA prompt logging are disabled; no experiment tracker
or persistent optimizer checkpoint is enabled.

## Commands (PowerShell, repository root)

Prepare public data only; no model calls. Choose fresh directories for every run:

```powershell
.venv/Scripts/python.exe -m s1compiler.banking77 prepare --out runs/banking77-data
.venv/Scripts/python.exe -m s1compiler.banking77 plan --data runs/banking77-data
```

Exercise all four arms with a mock model and an identity proposer, using the real
installed GEPA package. This is ONLY an offline plumbing check:

```powershell
.venv/Scripts/python.exe -m s1compiler.banking77 select --data runs/banking77-data --out runs/banking77-offline --seeds 7
.venv/Scripts/python.exe -m s1compiler.banking77 test --data runs/banking77-data --frozen runs/banking77-offline --out runs/banking77-offline-results
```

Before live search, review `usecase.json` and verify a full 77-label native request
with one separately authorized call. Existing CLI commands can build and run the
baseline against a locally prepared train state:

```powershell
.venv/Scripts/python.exe -m s1compiler draft runs/banking77-data/usecase.json --out runs/banking77-smoke.s1.json
# Create smoke-state.json with {"text": "..."} from a TRAIN example only.
.venv/Scripts/python.exe -m s1compiler run runs/banking77-smoke.s1.json --state smoke-state.json --backend typesafe --allow-paid --allow-unvalidated --max-calls 1 --no-cache
```

After explicit paid-call and teacher-sharing consent, configure credentials
locally. `TYPESAFE_API_KEY` and the chosen teacher provider's credentials are
required. `.env` is not loaded automatically. Never put credentials in command
arguments or reports. Replace the model placeholder with the agreed real teacher:

```powershell
.venv/Scripts/python.exe -m s1compiler.banking77 select --data runs/banking77-data --out runs/banking77-live --backend typesafe --allow-paid --share-feedback --acknowledge-budget-limits --teacher-model PROVIDER/MODEL
```

`select` writes all frozen `.s1.json` artifacts, per-arm selection/accounting
records, the pre-execution protocol, and `frozen.json`; it prints the manifest
SHA256. It makes no test requests. Review every frozen prompt's meaning before
acknowledging that exact digest. Then, with separately budgeted test calls:

```powershell
.venv/Scripts/python.exe -m s1compiler.banking77 test --data runs/banking77-data --frozen runs/banking77-live --out runs/banking77-live-results --backend typesafe --allow-paid --acknowledge-budget-limits --reviewed-manifest MANIFEST_SHA256
```

Testing verifies every program, the dataset, and frozen implementation hashes
before calling the provider. An exclusive `test-started.json` marker prevents
accidental retesting of that experiment even if the first attempt fails. This is
an operational guard, not a security boundary against someone copying directories.
Do not retune after inspecting test results. Preserve partial outputs on failure.

`report.md` is the compact comparison; `report.json` contains full metrics and
intervals. Per-arm evidence saves IDs, gold/predicted labels, and review decisions,
without request text. Artifacts contain prompts deliberately for semantic review;
they and evaluation outputs remain sensitive. Hashes are integrity checks, not
anonymization or signatures. Nothing is marked production approved.

## Integration references

The existing integration versions are unchanged. The benchmark's GEPA call uses
the documented [`module_selector="all"` and `stop_callbacks` API](https://gepa-ai.github.io/gepa/api/core/optimize/),
checked against installed GEPA 0.1.4 and primary documentation on 2026-09-19.
See also [TypeSafe Choice](https://docs.typesafe.ai/primitives/choice) and
[confidence semantics](https://docs.typesafe.ai/confidence).
