# Implemented architecture

## System boundaries

```text
UseCase YAML + labeled JSONL splits
                |
      validate schema and split isolation
                |
      template baseline (always present)
                |
 DSPy architect [optional; train demonstrations]
                |
 bounded structural proposals [optional; train errors]
                |
 standalone GEPA [optional; DSPy component proposer]
                |
    select on validation; reject regressions/ties
                |
    fit policy using calibration, not validation/test
                |
       freeze program + provenance + checksum
                |
      baseline and frozen-program test reports
                |
        portable program.s1.json
                |
   runtime: project -> native Jev request -> validate
                |
 deterministic numeric composition + review decisions
```

The optimizer proposes data, not executable programs. Architecture proposals
contain exactly `state_fields`, `questions`, and `bindings`. The compiler rebuilds
and validates a `Program` against the source declaration. The original decisions
and pinned model are not negotiable search parameters.

## Modules

| Module | Responsibility |
|---|---|
| `models.py`, `io.py` | Declarative contracts, shape validation, safe parsing, checksums. |
| `data.py` | JSONL labels, split isolation, dataset hashes. |
| `architect.py` | Template generator, DSPy Signatures and Predict modules. |
| `gepa_adapter.py` | JSON text components, native task evaluation, reflective feedback. |
| `compiler.py` | Bounded stage ordering, validation selection, freeze then test. |
| `backends.py` | SDK/mock providers, call budget, cache, model identity, accounting. |
| `runtime.py` | Projection, answer normalization, deterministic binding/policy logic. |
| `metrics.py`, `policy.py` | Scoring, diagnostics, empirical gate/threshold fitting. |
| `reporting.py`, `hardening.py` | Reports and review-required robustness proposals. |
| `cli.py` | Beginner-facing workflow, doctor, demo, schema/playground export. |
| `hierarchy.py`, `hierarchy_validation.py`, `hierarchy_data.py` | Versioned graph models, lowering, typed references, and split lineage. |
| `hierarchy_compiler.py`, `hierarchy_architect.py`, `hierarchy_gepa.py` | Bounded graph selection, wording optimization, calibration, and freeze. |
| `hierarchy_runtime.py`, `hierarchy_evidence.py`, `hierarchy_metrics.py` | Serial graph execution, durable attempt replay, and root/path reporting. |
| `hierarchy_study.py` | Separate preregistered flat-versus-graph research runner, live manifest review, selection ledger, and reconcile step. |
| `hierarchy_research_data.py`, `holdout_exclusions.py` | Public human-judged study inputs and prior-holdout exclusion export, with no provider calls. |
| `research.py`, `research_data.py`, `research_stats.py`, `research_teacher.py` | Flat five-benchmark research suite, dataset checks, statistics, and teacher boundary. |
| `banking77.py` | Flat Banking77 experiment runner. |
| `resilience.py`, `errors.py` | Bounded batch recovery and the framework error types. |

The hierarchy path is a sequence of native typed leaf requests, not one native
request with a global posterior. The frozen graph owns routing and final output
mapping; each leaf uses the pinned model and the caller's shared request budget.
The flat v1 artifact and its runtime keep their original one-request behavior.
See [the contract](HIERARCHY_CONTRACT.md) and [migration guide](HIERARCHY_MIGRATION.md).

## DSPy and GEPA integration

DSPy controls the *teacher*, not the Jev runtime. `Design` converts a UseCase plus
training examples/errors into a flat typed plan. `HierarchyDesign` proposes
bounded graph JSON from train-only hierarchy traces; authored source contracts
and limits are copied by the compiler, then the full graph is validated.
`Revise` updates requested instruction/rubric entries. All use `dspy.Predict` inside an explicit
`dspy.context(lm=...)`. A user-selected teacher provider is mandatory.

Standalone GEPA receives a candidate mapping `str -> str`. Each value is the JSON
encoding of a TypeSafe entry; the entry itself may be a string, object, list, or
null. Keys identify question instructions and indexed criteria/Score levels.
Numeric indices avoid problems with arbitrary Choice label delimiters.

`JevGEPAAdapter.evaluate` calls the native typed runtime and returns outputs,
scores, and optional trajectories via `EvaluationBatch`. Training trajectories
include projected input, expected labels, typed predictions/probabilities, and
quality/error diagnostics. `make_reflective_dataset` structures this information.
`propose_new_texts` delegates to the DSPy teacher and rejects malformed mutations.
There is no assumption that `dspy.LM` can be passed directly as GEPA's generic
reflection callable, and no emulated Jev chat-completion endpoint.

GEPA selection uses validation scores. Reflection comes from the training path.
No dataset records from calibration/test enter the teacher. Repeated validation
selection can still overfit; validation is not the final generalization estimate.
For hierarchy, qualified expanded-stage component keys preserve reused-subgraph
identity. `HierarchyGEPAAdapter` scores complete root executions, and only
registered train traces enter reflection. GEPA metric rows and native child
requests have separate counters and budgets. Structure is selected first;
wording is selected next; final review gates are fitted after both.

## Supported structural search

The outer loop can propose add/drop/revise questions, select a subset of allowed
state fields, change internal primitive choices where bindings remain valid, and
propose positive numeric composition weights. Direct Choice/Noul outputs must
retain their primitives. Score outputs may use weighted means of Score/Noul
question values. Every question must be referenced by a binding.

At most ten structural rounds are allowed; each complete candidate is evaluated
on validation before acceptance. Hierarchy proposals may use typed stage-to-stage
dataflow, conditional calls, fan-in and bounded reusable subgraphs. The root
contract, target model and limits stay fixed. Only train examples and sealed
train traces enter the teacher. Invalid graphs are rejected atomically; provider
and budget failures remain failures. Structure selection precedes wording
optimization and final calibration; there is no claim of joint global search.
Flat questions within one native request remain independent. Neither path
performs autonomous external actions.

## State minimization and portability

State declarations allow top-level string/integer/number/boolean/array/object
fields. The compiler may drop fields but cannot introduce undeclared inputs.
Projection strips undeclared top-level fields before provider/teacher use.
Declared objects/arrays are not recursively minimized or PII-redacted.

The saved envelope is `{ "program": {...}, "sha256": "..." }`. The envelope
checksum covers the full program, including provenance; the runtime content hash
excludes provenance. A changed artifact must be revalidated and saved through the
model. SHA256 detects accidental/tampered-content mismatch relative to the stored
hash, but is not an authenticity signature: an attacker can recompute it.

Legacy names are kept on purpose after the rename from System One Compiler to
Typewright, so existing artifacts and checksums stay valid: the `.s1.json`
extension, the `systemone-*` format identifiers (such as `systemone-program/v1`),
the `"systemone-compiler"` provenance key (also printed by `doctor`), and the
`S1_TEACHER_*` environment variables. Renaming any of them is a format change that
needs a migration, not a find-and-replace.

Production only requires the base package and `live` extra. JSON schema and native
playground export make the questions inspectable outside the compiler. The native
export does **not** export local composition, policy fitting, or action execution.

## Exact score semantics

For example i, the search objective is the positive-weighted mean of per-decision
qualities. A one-hot class target is y, predicted probabilities are p, and a Score
has K ordered levels:

```text
Choice quality = 1 - sum_j((p_j - y_j)^2) / 2
Noul quality   = 1 - (p_true - y)^2
Score quality  = 1 - abs(predicted_score - target_score) / (K - 1)
```

This gives values in [0,1] for valid answers. There is no arbitrary latency/cost
weight mixture in the default objective. Request budgets constrain execution;
reported cost/latency remain separate measurements. Review/abstention flags do
not change this objective, avoiding the trivial review-everything optimum.

Choice/Noul reports include accuracy, macro-F1, confusion counts, Brier, log loss,
and ten-bin expected calibration error. Score reports include MAE/RMSE and the
fraction within the declaration's tolerance. ECE from a small sample is noisy.
Latency summaries exclude cache hits; all-cached runs have no invented fresh-call
latency. Unknown usage/cost is explicitly unknown.

## Probability, confidence, and policies

- Choice: selected category probability is the review gate; vendor confidence is
  recorded separately and is not assumed equal to max probability.
- Noul: output is P(true), with no vendor confidence. A fitted threshold determines
  true/false. The gate is the probability of the **selected** outcome, so a false
  decision at p=0.6 and threshold=0.8 has gate 0.4, not 0.6.
- Direct Score: gate uses the native reported confidence.
- Weighted Score: each Score component is divided by its maximum level and each
  Noul component supplies P(true). A positive-weighted average is rescaled to the
  output's 0..K-1 scale. Gate is the minimum component gate, explicitly a heuristic.
  No composite posterior or vendor confidence is invented.

Policy fitting first selects a Noul classification threshold from calibration
predictions, then maximizes accepted calibration coverage subject to a minimum
accepted count and an empirical error constraint. If no gate qualifies, it sets
`force_review=true`. A separate probability recalibrator such as isotonic
regression or temperature scaling is **not implemented**.

The reported Wilson interval is descriptive; it is not a corrected guarantee
under adaptive threshold selection or distribution shift. Per-output review flags
are not a joint guarantee for the whole decision vector. Production needs its own
risk analysis, appropriate samples, monitoring, and authorization.

## Failure and release behavior

Malformed candidate edits are rejected. Authentication, timeout, rate-limit,
response-schema, model-drift, and exhausted-budget failures stop the run rather
than masquerading as poor prediction scores. The implementation does not silently
retry a paid request or switch to mock after a live failure.

A measured artifact means the compiler completed evaluation using a non-synthetic
backend. It is not an approval: `deployment_approved` remains false. The runtime
can enforce that unmeasured live artifacts need an explicit experimental override;
it does not certify the source labels, dataset independence beyond implemented
checks, external actions, or operational safety.
