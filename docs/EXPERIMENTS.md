# Experiment log and backlog

Started 2026-09-19. Records experiments actually executed against this package,
with their raw outputs, and a prioritized backlog of proposed ones.

**E1-E3 and the BANKING77 section used the synthetic `mock-lexical/v1` fixture
and are not Jev measurements. E4 and E5 are real `jev-1.13.0` measurements** (48
live requests, operator-authorized). No claim of real prompt-quality improvement
is made anywhere in this file; E4 in fact measures the improvement as negligible.

Raw outputs: `runs/experiments/`. Source run under analysis:
`runs/live-teacher-001/` (real DSPy teacher `openai/cinf/deepseek-v4.1-flash`,
real GEPA, mock evaluation target).

---

## Executed

### E1 — Which change moved the objective? **Answer: criteria only.**

Ablated the compiled artifact against the template by swapping one layer at a
time. Objective on each split, delta vs template:

| Variant | train | validation | calibration | test | mean Δ |
|---|---|---|---|---|---|
| A template (baseline) | 0.688489 | 0.712382 | 0.727066 | 0.669721 | +0.000000 |
| B + compiled **instructions** only | 0.688489 | 0.712382 | 0.727066 | 0.669721 | **+0.000000** |
| C + compiled **criteria** only | 0.757290 | 0.728051 | 0.747714 | 0.705422 | **+0.035205** |
| D + compiled **policies** only | 0.688489 | 0.712382 | 0.727066 | 0.669721 | **+0.000000** |
| E compiled (all) | 0.757290 | 0.728051 | 0.747714 | 0.705422 | +0.035205 |

**The mock fixture is completely blind to `instructions`.** `MockBackend` scores
lexical overlap against *criteria text only*; instruction fields — including
`inspect`, `guidance`, and the question wording — never reach its scoring path.
Policies do not enter the objective either (Brier/MAE are threshold-independent).

Consequence: the structural change in `runs/live-teacher-001` — narrowing
`inspect` from `` [`message`, `customer_plan`] `` to `` [`message`] `` on all
three questions — has **exactly zero** measured effect. That change is plausible
on its merits and may well matter against a real model, which does read
instructions. The fixture simply cannot tell us.

### E2 — Is +0.035 sampling noise? **Answer: not within the fixture.**

Pooled all 48 rows, resampled eval sets 300×, recomputed the template-vs-compiled
delta:

| eval n | mean Δ | sd | 95% interval | P(Δ ≤ 0) |
|---|---|---|---|---|
| 12 | +0.034390 | 0.014211 | [+0.010904, +0.067045] | 0.000 |
| 24 | +0.035282 | 0.008675 | [+0.019022, +0.052586] | 0.000 |

The effect is stable in sign across every resample. **Caveat that limits this
result:** the compiled program was *selected* using these same 48 rows, so this
measures metric sampling spread, not out-of-sample generalization. A clean
estimate needs rows never seen during selection.

### E3 — Why did GEPA accept nothing? **Answer: half its search space is inert.**

Swapped each GEPA component from template → compiled individually, on train:

| Component | predictions changed | objective Δ |
|---|---|---|
| `department/instructions` | 0/12 | +0.000000 |
| `department/criterion/0` | 2/12 | +0.029098 |
| `department/criterion/1` | 1/12 | +0.015233 |
| `department/criterion/2` | 0/12 | +0.000406 |
| `department/criterion/3` | 1/12 | +0.005976 |
| `urgent/instructions` | 0/12 | +0.000000 |
| `urgent/criterion/0` | 0/12 | **−0.042835** |
| `urgent/criterion/1` | 11/12 | +0.015446 |
| `frustration/instructions` | 0/12 | +0.000000 |
| `frustration/level/0` | 0/12 | +0.000000 |
| `frustration/level/1` | 0/12 | +0.000000 |
| `frustration/level/2` | 0/12 | +0.000000 |

**6 of 12 components are completely inert** under this fixture: all three
`*/instructions` (per E1), and all three `frustration/level/*`. The Score
question is degenerate here — its level texts share no vocabulary with the ticket
messages, so the distribution stays uniform (0.333/0.333/0.333) and the expected
score is pinned at 1.0 regardless of wording.

GEPA proposes one component per iteration, so roughly **half its budget lands on
components that cannot change the score by construction** — which is exactly the
byte-identical old/new subsample scores in the run log (8 of 12 iterations).
Contributing factors worth separating: `reflection_minibatch_size` is hardcoded
to `min(3, len(train))` = 3, and `use_merge=False`.

Also note the deltas are strongly non-additive: the individual changes sum to
+0.0233 on train, while applying all of them together gives +0.0688.

### Corrections this log supersedes

- `SETUP_REPORT.md` originally attributed the gain to "one structural change."
  E1 falsifies that: instructions contribute exactly zero. It also understated
  the diff — `inspect` was narrowed on **all three** questions, not just `urgent`.
- An earlier hypothesis that the delta was a verbosity artifact was tested and
  **rejected**: 40 words of meaningless filler moved the objective +0.000000, and
  40 topical words lifted from the test states moved it **−0.030978** (padding
  every criterion flattens the softmax and hurts).

### E4 — Frozen A/B on **real Jev**. Executed with operator authorization.

`s1 evaluate --backend typesafe --allow-paid --max-calls 13 --no-cache`, twelve
test rows per program, no fitting. Backend `typesafe-sdk/0.7.0`,
`synthetic: false`, model `jev-1.13.0`.

| | mock objective | **Jev objective** | input tokens |
|---|---|---|---|
| template | 0.669721 | **0.974631** | 8,795 |
| compiled | 0.705422 | **0.975844** | 12,947 |
| **delta** | **+0.035205** | **+0.001350** | **+47%** |

Paired per-example analysis on Jev (n=12): mean +0.001350, sd 0.005077,
11/12 positive, exact sign test **p = 0.0063**, 95% interval
**[−0.001876, +0.004576]**.

Read carefully, that says two different things:

- The **direction** is consistent and unlikely to be chance (p = 0.0063).
- The **magnitude** is negligible and its interval includes zero. Jev's template
  baseline is already 0.9746, leaving only 0.0248 of headroom; the compiled
  program captures **5.4%** of it.

**The mock fixture overstated the improvement by 26×.** And the compiled program
costs **47% more input tokens** to obtain it. On a cost-per-quality basis this
compile is a **net negative** — it buys a statistically-detectable but
practically irrelevant direction at half again the token price.

### E5 — Does mock predict Jev? Partially: ρ ≈ 0.53, wrong regime.

24 paired per-example scores (both programs × 12 rows), 24 additional Jev calls.

| | mean | sd |
|---|---|---|
| mock quality | 0.6876 | 0.1068 |
| Jev quality | 0.9759 | 0.0479 |

**Spearman rank correlation: +0.5261.**

So the fixture is not noise — it identifies genuinely hard examples. The two
lowest mock scores (`billing_11` 0.5655, `technical_09` 0.5653) are also Jev's
two hardest rows (0.8428, 0.9050). That is real signal.

But mock and Jev operate in **different regimes**. Mock sits at 0.688 with 0.31
of headroom; Jev sits at 0.976 with 0.025. Optimizing against mock is chasing a
gap that largely does not exist on the target model, which is precisely how a
+0.035 mock gain collapses to +0.0014 on Jev.

**Practical rule this supports:** use the fixture for shape-testing, plumbing,
and hard-example triage. Do **not** use it to select between candidate programs,
and never quote its deltas as expected improvements.

### Spend

48 live Jev requests total (12 + 12 + 24), ~43.5k input and ~3.7k output tokens,
model `jev-1.13.0`. `dollar_cost` is `null` — the package does not estimate price.

---

## Backlog

### Tier 2 — real optimization (gated on E5 showing signal)

**E6. Live-target compile** against a **fresh** held-out split. The current test
split has been examined and is now descriptive.

**E7. Teacher ablation** — `openai/cinf/deepseek-v4.1-flash` vs `openai/gpt-5.4`
vs `anthropic/claude-opus-5`, identical seeds and budgets. Only interpretable
once there is a real signal to improve.

**E8. GEPA budget efficiency.** Restrict proposals to components E3 shows are
live, and vary `reflection_minibatch_size`. Hypothesis: the current setup wastes
~50% of `max_metric_calls`. Cheap to test on mock, but only worth shipping if
E5 says mock means anything.

### Tier 3 — the binding constraint

**E9. Dataset scale-up and power analysis.** E2 gives sd ≈ 0.0142 at n=12 and
≈ 0.0087 at n=24. Compute the *n* needed to resolve effects of the size we care
about, then decide whether to label more data or accept that this fixture is
permanently a smoke test. Everything in Tier 2 is gated on this.

**E10. Hardening set as a robustness benchmark.** Use `s1 harden` output as a
separate perturbation eval **after human label review** — never as a training
signal.

### Recommended documentation change

`docs/ARCHITECTURE.md` and `examples/support_triage/DATASET_CARD.md` should state
the two structural limits E1/E3 established: the fixture reads criteria only, and
its Score question is degenerate. Both are properties of the bundled fixture, not
defects in the compiler — but without them stated, a mock run looks like it
evaluates the whole program when it evaluates roughly half of it.

---

## BANKING77 harness — validated on mock, live run costed but not authorized

The `banking77` module (authored outside this session) was reviewed and exercised
end to end. It is the E9 dataset scale-up, and it is the right vehicle: E4/E5
showed `support_triage` is saturated on Jev (0.9746 baseline, 0.0248 headroom),
so that task cannot measure optimization at all.

**Dataset prepared** — pinned commit `57ec275d…`, both CSV checksums verified:
5,969 train / 1,998 validation / 2,036 calibration / 3,080 test.

**Design.** Four arms: `A` authored baseline, `B` one-pass rewrite with no
validation selection, `C` GEPA search, `D` independent restarts selected on
validation at matched budget. **D is the control most prompt-optimization work
omits** — it isolates whether GEPA's reflective search beats plain best-of-N at
equal spend. Paired bootstrap with Bonferroni across the three comparisons.

**Mock select + test completed** (12,320 evaluation requests, ~22s total).
Guards verified in practice: re-running `test` into a used directory refuses
("Choose a fresh result directory"); live `select` refuses without both
`--allow-paid` and `--acknowledge-budget-limits`, and additionally requires
`--share-feedback` plus an explicit `--teacher-model`.

Mock result — **all four arms identical, every delta exactly 0.0**, because mock
mode substitutes `IdentityTeacher`. This validates plumbing only; it is not a
null result about optimization.

| arm | accuracy | macro F1 | Brier |
|---|---|---|---|
| A / B-7 / C-7 / D-7 | 0.3166 | 0.3166 | 0.8816 |

Worth noting: the lexical fixture reaches 31.7% on a 77-way task where chance is
1.3%. Unlike `support_triage`, this task is **nowhere near ceiling**, so a live
run would have real headroom to measure.

### Live cost — requires an explicit decision

Baseline prompt is 6,324 chars (~1,581 tokens) per request.

| configuration | requests | ~input tokens |
|---|---:|---:|
| minimal, 1 seed | 32,664 | ~51.6M |
| default, 3 seeds | 111,160 | ~175.7M |

For scale, E4/E5 together were 48 requests and 43.5k input tokens. The minimal
configuration is roughly **1,000×** that. The package deliberately provides no
dollar estimator and reports `dollar_cost: null`; set provider-side spending
limits before authorizing either configuration.
