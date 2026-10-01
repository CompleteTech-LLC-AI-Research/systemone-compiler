# Multi-benchmark Jev prompt-engineering study

This suite adds BoolQ, SST-5, CLINC150, GoEmotions, and ANLI to the existing
BANKING77 work. It implements a locally preregistered, reproducible experiment
and reporting procedure. Software verification and synthetic results are not
evidence that DSPy improves Jev. Scientific conclusions require completed live
runs, the declared comparisons, uncertainty assessment, and semantic review.

## Questions and fixed contracts

| Benchmark | Model-visible inputs | Native Jev output | Primary metric | Secondary diagnostics |
| --- | --- | --- | --- | --- |
| BoolQ | question, passage | Noul `answer` | accuracy at P(yes) >= 0.5 | Brier, log loss, ROC AUC, average precision, reliability bins |
| SST-5 | whole review sentence | Score `sentiment`, five fixed levels 0–4 | MAE of expected score | RMSE, argmax-level accuracy, macro-F1, quadratic weighted kappa, Brier/log loss |
| CLINC150 | request text | Choice `intent`, 150 original intents plus `oos` | macro-F1 across all 151 labels | in-scope accuracy; out-of-scope precision, recall, false acceptance, ROC AUC and average precision |
| GoEmotions | comment text | 28 Noul questions: 27 emotions and neutral | macro-F1 at fixed 0.5 thresholds | micro-F1, exact match, Hamming loss, per-label support/F1/Brier/reliability, binary log loss |
| ANLI | premise, hypothesis | Choice `relation`: entailment, contradiction, neutral | pooled accuracy | R1/R2/R3 slices, macro-F1, confusion, Brier/log loss |

Primary metrics are declared before model execution. CLINC's macro-F1 gives each
intent and out-of-scope equal weight rather than allowing its class imbalance
to determine the headline result. GoEmotions remains multi-label; there is no
invented joint probability. SST's MAE assumes equal distances between its ordinal
levels. Its classification diagnostics use the most probable level, not a rounded
expected score. Ties choose the lowest Score level; native Choice selections are
preserved. A zero precision/recall/F1 denominator produces zero and absent labels
remain in macro averages; support counts are always reported.

Use `jev-1.13.0` throughout. Output names, kinds, labels, Score lengths, input
fields, bindings, and question counts are fixed. Only question instructions and
criteria may change. Baseline definitions are authored experiment inputs, not
benchmark-author rubrics. Review baseline and selected wording for semantic drift.
The same generated program runs without importing DSPy or GEPA at inference time.

## Sources and provenance

Every downloaded file has a SHA256 and byte-length pin in `research_data.py`.
Existing cache files are also checked. Downloads are data only: no executable
dataset builders, untrusted pickle, archive extraction, or generated code.
The prepared manifest records source URLs, hashes, preprocessing source hashes,
the split seed, group counts, label counts, and every split's content hash.
No benchmark data are included in the wheel or source manifest.

- [BoolQ](https://github.com/google-research-datasets/boolean-questions): Clark et al., NAACL 2019.
  The original Google Storage URLs returned HTTP errors during this run. The
  pinned SuperGLUE `BoolQ.zip` release is used instead. Only labeled train and
  validation are read. Official validation becomes the **local test holdout**;
  this is not an official hidden-test or leaderboard score. The release lacks
  page titles, so grouping uses normalized passages, not Wikipedia article IDs.
- [SST](https://nlp.stanford.edu/sentiment/): Socher et al., EMNLP 2013.
  The official PTB trees supply root sentiment labels and sentence leaves only.
  Constituents are never separate examples. Leaves retain upstream PTB tokenization.
  No movie/reviewer identifiers are supplied, so sentence grouping cannot certify
  review-level separation.
- [CLINC150](https://github.com/clinc/oos-eval): Larson et al., EMNLP-IJCNLP 2019.
  Use `data_full.json`, including the separate out-of-scope splits, pinned to
  commit `828f8093932c8fe6ca7936c3d2e52903b1c523de`. Group by normalized request;
  the file supplies no speaker or paraphrase-seed IDs.
- [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions):
  Demszky et al., ACL 2020. Use curated agreement-filtered TSV labels, with
  `link_id` metadata from the three pinned raw annotation CSVs. Raw rater labels
  are not substituted for curated labels. Group by Reddit thread across splits;
  repeated author appearances in distinct threads remain a limitation.
  GitHub files are pinned to `4700efb9afa54286b0e04473ba80a13e8461e25f`.
- [ANLI](https://github.com/facebookresearch/anli): Nie et al., ACL 2020.
  Use the official v1.0 archive. The actual release calls its premise `context`;
  the adapter explicitly maps it to `premise`. Neither annotation `reason`,
  original model predictions, genre, tags, nor annotator fields enter Jev or the
  teacher. Group the normalized premise across all three rounds.

Original licenses and citations remain applicable. See dataset manifests and
upstream releases; the repository's code license does not relicense downloaded
data. Public comments, cache files, hashes, and prediction records may still be
sensitive. Raw GoEmotions metadata are downloaded specifically to audit grouping;
authors and thread strings are not exported into model inputs.

## Strict audits and explicit derived datasets

`prepare` is strict by default: repeated IDs, duplicate projected inputs, or
development/holdout group overlap prevent generation of a runnable dataset.
`raw-audit.json` records the reason. The existing compiler checks are unchanged.

The separately requested `--clean-derived` variant applies this deterministic,
pre-model data-curation rule:

1. Stop on repeated upstream IDs.
2. Remove **all** copies of an exact projected input when their human gold labels
   conflict. Record every removed ID and its original split.
3. For identical inputs with identical labels, retain one, preferring the holdout
   then lexicographic ID; record the other copies.
4. Remove development examples whose group appears in the retained holdout.
5. Pool the remaining upstream train/development records and assign whole groups
   approximately 60/20/20 to train/validation/calibration with seed `20260920`.
   Group strata use the dominant complete gold-label vector, with deterministic
   ties. This is approximate group stratification, not guaranteed exact class
   proportions or iterative multilabel stratification.
6. Re-run all disjointness checks and save exclusion records plus label/group
   counts. No labels are relabeled, generated, or selected using model performance.

This is transparent **derived-benchmark curation**, not a way to report a cleaned
score as an official benchmark result. Conflict detection uses existing published
annotations during data curation, before the experimental protocol is registered.
Do not choose cleaning rules after looking at model results.

Observed raw-data problems include shared BoolQ passages, repeated SST sentences,
conflicting CLINC labels, duplicate GoEmotions comments/shared threads, and many
shared ANLI premises. The verification report records exact retained counts.
CLINC and GoEmotions cleaning also changes the holdout; every report states how
many holdout rows were removed and sets `official_leaderboard_comparable=false`.

## Experimental controls

Four arms are frozen for each task:

- A: one authored baseline, shared across the paired seed comparisons.
- B: one DSPy rewrite using three sampled train examples, without execution
  feedback or validation selection. It is a three-example control, not a strong
  exhaustive few-shot baseline. Invalid candidates are reported and fall back to A.
- C: DSPy component proposer through the real standalone GEPA engine and native
  `JevGEPAAdapter`, with three-example train reflection minibatches. All text
  components are editable; architecture changes and merging are disabled.
- D: independent rewrites from A, with three train examples and no Jev error
  traces. Pick the best validation objective, retaining A on ties.

Seeds are `7`, `17`, `29`. Report every seed. These seeds control local sampling
and GEPA, not nondeterministic provider internals. Train is the only source of
teacher demonstrations or traces. Validation selects prompts. Calibration fits
review gates after selection. All tasks/arms freeze before any test execution.

Selection uses the compiler's existing typed probability/score objective. The
primary benchmark classification thresholds remain fixed at 0.5 for Noul.
Calibration-fitted thresholds and review decisions are reported separately; they
do not silently redefine the primary benchmark metric. Review fitting uses a
5% empirical error target and minimum 30 accepted calibration examples. It is
not a guaranteed risk bound. Wilson intervals for review errors are descriptive
binomial diagnostics, not cluster-adjusted inferential guarantees.

C and D have equal ceilings and the same allowed component set, not necessarily
equal realized requests, tokens, dollars, or proposal counts. GEPA's screening
schedule differs from D's full-validation schedule. Actual usage must accompany
any comparison. A direct DeepSeek inference baseline is explicitly **not included**
in this protocol; it would require a separately specified output/scoring contract.

## Statistical analysis and reporting

The primary family contains **ten tests**: C versus A and C versus D for each of
the five tasks. C versus B and per-round/per-label slices are exploratory.

For each primary contrast:

- Calculate the task metric separately for each predeclared optimization seed,
  then the mean paired metric difference. Also report every seed difference and
  its sample standard deviation; do not pick the best test seed.
- Draw 5,000 paired cluster-bootstrap replicates. Whole passages, threads,
  premises, or normalized requests are resampled with replacement. The same
  sampled groups apply to both arms and all seeds. Recompute nonlinear macro-F1
  inside each replicate, rather than averaging per-example F1.
- Report a nominal percentile 95% interval and a Bonferroni percentile interval
  using all ten planned comparisons. Improvement is sign-adjusted so positive
  always favors C; a reduction in SST MAE is positive improvement.
- Draw 10,000 paired whole-group label-swap randomizations, using the same swaps
  across fixed seeds. Report a two-sided Monte Carlo p value with the +1 correction
  and its attainable resolution. Apply Holm's step-down adjustment over all ten
  primary hypotheses. Unexecuted planned tests contribute p=1, not a smaller family.
- Use analysis seed `90210`. Record the number of examples and independent groups.
  Mark degenerate bootstrap distributions. These are approximate intervals and
  exchangeability-based randomization tests, not unconditional guarantees.

This inference **conditions on the frozen prompts and fixed optimization seeds**.
Repeated predictions of one example are not additional independent samples.
The across-seed SD is descriptive, not an estimate from many independent
optimization trials. No formal power claim or cross-dataset replicability claim
is made. Practical thresholds (+0.02 accuracy/macro-F1, 0.10 lower SST MAE) are
preregistered targets rather than promised detectable effects. No automatic
success assertion is produced.

Method references: [Dror et al. (2018)](https://aclanthology.org/P18-1128/) on
statistical testing in NLP; [Holm (1979)](https://doi.org/10.2307/4615733) on
multiple comparisons. The group-resampling design is an explicit choice for this
suite, not a claim that either paper certifies the implementation. Strong claims
of broad generalization additionally need fresh independently labeled application
data; public benchmark contamination cannot be ruled out.

`report.json` includes confusion matrices, class support, per-label metrics,
calibration bins, CLINC out-of-scope diagnostics, ANLI round slices, review
coverage/error, latency distributions, per-arm compilation/runtime accounting,
all comparisons, actual environment versions, and provenance hashes.
`report.md` and `primary-table.json` provide compact tables. Publication figures
are `primary-effects.pdf`, `.svg`, and `.png`, with a separate axis/units per task.
Synthetic outputs are prominently labeled. No heterogeneous aggregate score is
computed. Subset suites keep all ten primary hypotheses in their family.

## Reproducibility, accounting, and failure handling

Registration is local and checksummed, not an independently timestamped public
preregistration service. Register before provider calls and archive the digest.
It records all source-code hashes and actual installed package versions; this is
an environment inventory, not a fabricated dependency lock. Use the frozen code
version for replay. `--data` can relocate a prepared dataset root without changing
its recorded hashes or the original protocol.

The default search ceiling is explicitly **100,000 Jev requests per C/D task/seed**,
larger than BANKING77's original pilot because ANLI's validation set alone has
roughly 25,000 examples. This suite is a separate plan, not a silently enlarged
BANKING77 run. Registration computes all request ceilings without making calls.
There are 30 teacher signatures per C/D seed, one per B seed, a separate teacher
provider-forward ceiling of twice the signature allowance, a per-call 8,192
output-token limit, and a common 24,000-character question ceiling.

The research teacher meters synchronous DSPy LM-forward invocations, response
model IDs, provider-reported input/output tokens, and available SDK cost estimates.
The declared response model is checked exactly; aliases are not treated as proof
of immutable weights. No raw DSPy history is retained. SDK retries are disabled.
Failed attempts retain counters, and unavailable usage is explicit. SDK cost
estimates are distinct from billed cost (`dollar_cost=null`). Neither aggregate
tokens nor dollars are hard-capped by this runner. The user's no-dollar-cap
preference does not remove the experiment's fixed request ceilings.

Testing randomizes arm order within each example to reduce service-time confounds.
Each example/arm is called once; caches are disabled. Typed answers, decisions,
labels, IDs, grouping, latency, and usage are streamed to evidence JSONL without
input text. These records remain sensitive. Freeze and execution manifests pin
their hashes. Replay verifies every example and reproduces decisions from stored
answers and frozen policies, without requesting another prediction.

Provider, budget, model drift, malformed responses, or missing predictions stop
the run. They are not scored as wrong answers or omitted. A failed execution cannot
produce a success report. Completed artifacts, partial evidence, and sanitized
failure metadata are retained. An exclusive marker prevents accidental retesting
of the same frozen experiment, including after failure. Checksums/markers are
integrity and workflow tools, not signatures or adversarial security boundaries.

## Commands

PowerShell, repository root. All output directories/files below must be fresh.

```powershell
.venv/Scripts/python.exe -m pip install -e ".[all,dev,research]"
.venv/Scripts/python.exe -m typewright.research prepare --out runs/research-data --clean-derived
.venv/Scripts/python.exe -m typewright.research verify --data runs/research-data
```

Omit `--clean-derived` to run strict raw-source preparation; audited overlap
causes a refusal with `raw-audit.json`, rather than weakening a compiler check.

Bounded offline verification using actual benchmark labels, an identity proposer,
and the mock backend. `smoke-data` samples whole groups independently of labels
and marks the result offline-only; live registration refuses it:

```powershell
.venv/Scripts/python.exe -m typewright.research smoke-data --data runs/research-data --out runs/research-smoke-data --smoke-groups 8
.venv/Scripts/python.exe -m typewright.research register --data runs/research-smoke-data --out runs/research-smoke-protocol.json --seeds 7 --search-calls 10000 --teacher-calls 2
.venv/Scripts/python.exe -m typewright.research select --protocol runs/research-smoke-protocol.json --out runs/research-smoke-frozen
.venv/Scripts/python.exe -m typewright.research test --frozen runs/research-smoke-frozen --out runs/research-smoke-results
.venv/Scripts/python.exe -m typewright.research report --frozen runs/research-smoke-results
```

For an authorized live study, first register full datasets with
`--backend typesafe --teacher-model deepseek/deepseek-flash`. The default expected
teacher response ID is `deepseek-flash`; a different provider response mapping
must be explicitly specified with `--teacher-response-model` before registration.
Review the budget and baseline contracts, and configure credentials locally.
`.env` is not automatically read. Provider calls require `--allow-paid` and
`--acknowledge-budget-limits`; live selection separately requires
`--share-feedback`. These flags must reflect actual consent.

```powershell
.venv/Scripts/python.exe -m typewright.research register --data runs/research-data --out runs/research-live-protocol.json --backend typesafe --teacher-model deepseek/deepseek-flash
.venv/Scripts/python.exe -m typewright.research select --protocol runs/research-live-protocol.json --out runs/research-live-frozen --allow-paid --share-feedback --acknowledge-budget-limits
# Review every frozen prompt. Substitute the printed frozen-manifest digest:
.venv/Scripts/python.exe -m typewright.research test --frozen runs/research-live-frozen --out runs/research-live-results --allow-paid --acknowledge-budget-limits --reviewed-manifest SHA256
```

Live availability is not established by this implementation. Existing experiment
notes ([EXPERIMENTS.md](EXPERIMENTS.md)) report that Jev quantizes probabilities, which
broke the original fixed probability-sum tolerance at 77 labels. `runtime.py` now
derives the tolerance from the decimal grid actually returned, capped at
`MAX_SUM_DRIFT` (0.05); full-precision responses stay at 1e-4. Distributions outside
that bound abort and are reported. This benchmark suite did not change the
tolerance.
