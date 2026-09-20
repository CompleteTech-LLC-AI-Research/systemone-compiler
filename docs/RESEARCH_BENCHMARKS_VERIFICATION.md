# Research suite verification — 2026-09-20

This records executed software and data checks, not model-quality results.
See [the study protocol and commands](RESEARCH_BENCHMARKS.md) for the methods,
controls, statistical assumptions, source citations, and limitations.

## Executed checks

- Full suite: `python -m pytest -q` — **177 passed in 12.54 seconds**.
  One upstream Pydantic warning notes that a DSPy/LiteLLM response TypedDict's
  `ReadOnly` qualifier is not mutation protection. It was not suppressed.
- `python -m ruff check src tests examples` — passed.
- All five upstream datasets downloaded and checked against pinned SHA256 and
  byte counts. Complete derived datasets passed the existing strict four-split
  disjointness checks. Downloads remain local; benchmark data are not bundled.
- All five actual-source smoke subsets completed registration, selection,
  calibration, freezing, testing, reporting, and independent offline replay.
  This used the mock prediction backend, identity teacher, and installed GEPA;
  it made **zero Jev or DeepSeek requests**. Real gold labels do not turn mock
  predictions into real Jev results.
- The smoke used seed 7 and four arms per task: 20 frozen programs, 41 unique
  held-out examples, and 164 stored test predictions. Statistical reporting used
  5,000 paired cluster-bootstrap replicates and 10,000 paired cluster
  randomizations, with the full ten-comparison primary family retained.
- JSON and Markdown reports and PDF/SVG/PNG figures generated successfully.
  The PNG was visually inspected: separate task panels, readable labels,
  direction-of-improvement labels, and a prominent synthetic-results notice.
  The identity mock controls tie; their zero differences are not gain evidence.
- Actual DSPy integration boundaries were exercised with fake provider responses:
  token accounting, response-model mismatch, independent request ceilings,
  and history suppression. These tests do not establish live provider behavior.
- Wheel and sdist built successfully under `dist/research-suite-verified/`.
  The wheel contains all four new research modules and the starter template,
  with no benchmark downloads or run artifacts. A separate target installation
  passed the no-key demo (36 mock requests, zero network calls) and research CLI
  help. Importing the installed library and running its frozen demo did not load
  DSPy, GEPA, NumPy, or Matplotlib.

## Prepared dataset inventory

Location: `runs/research-suite-20260920-final/`. All rows retain upstream human
annotations. Variant: `grouped-clean-derived-v1`. Counts below are examples,
not independent groups; each dataset manifest separately records group counts.

| Task | Train | Validation | Calibration | Local test | Original holdout rows removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| BoolQ | 4,795 | 1,598 | 1,597 | 3,270 | 0 |
| SST-5 | 5,781 | 1,926 | 1,924 | 2,210 | 0 |
| CLINC150 | 10,920 | 3,640 | 3,633 | 5,498 | 2 |
| GoEmotions | 27,191 | 8,885 | 8,713 | 5,406 | 21 |
| ANLI | 75,882 | 25,307 | 25,241 | 3,200 | 0 |

BoolQ's official validation is this study's local test. Other tasks use their
published labeled test sets, subject to the explicit exclusions shown above.
These are derived-dataset results, not directly comparable leaderboard scores.

Exclusions across all source splits, recorded individually in `exclusions.json`:

| Task | Development group overlaps holdout | Duplicate input | Conflicting input labels |
| --- | ---: | ---: | ---: |
| BoolQ | 1,437 | 0 | 0 |
| SST-5 | 0 | 14 | 0 |
| CLINC150 | 0 | 1 | 8 |
| GoEmotions | 3,725 | 141 | 202 |
| ANLI | 39,500 | 131 | 4 |

Strict preparation refuses such overlaps. The explicit derived mode removes
conflicting inputs in full, preserves holdout priority for consistent duplicates,
and excludes development groups shared with the holdout. Compiler checks were
not relaxed. Earlier BoolQ source URLs failed; the pinned SuperGLUE release was
used, as documented in the methodology.

## Reviewable artifacts

- Full unexecuted live protocol: `runs/research-suite-20260920-full-protocol.json`.
  SHA256 envelope digest:
  `0fd1347621bcd741896681724cdf77b86ef0f6bae70c885db865dcaae9976578`.
  Target `jev-1.13.0`; teacher `deepseek/deepseek-flash`; seeds 7, 17, 29.
  Its total Jev request allowance is **3,606,920**, a ceiling rather than an
  estimate or a spending commitment. Registration made no provider calls.
- Smoke report: `runs/research-suite-20260920-smoke-results/report.md` and
  `report.json`, alongside hashed per-arm evidence and exportable figures.
- Smoke protocol digest:
  `ed1555773a23b24eadf61d7491ffa49a8171a770f7955aac465cd2bd1fae7941`.
- Smoke frozen-manifest digest:
  `06328295f723f140afd06d19286f175b38dbd38f1c2ddc86e5ce113721a99cc5`.

## Not executed and remaining limitations

No full live study, live teacher upload, or new live prediction was performed for
this addition. Separate teacher-data sharing consent is still required under
the project instructions. Live execution also requires working credentials and
semantic review of the frozen prompts before test. Unlimited dollar budget does
not remove the registered request ceilings or authorize sharing by itself.

No accuracy improvement, production readiness, live-provider compatibility,
formal power, or remote CI success is claimed. Public-data pretraining
contamination cannot be ruled out. Inference conditions on the frozen prompts
and fixed seeds. Direct DeepSeek inference is not an arm of this protocol.
Existing Jev probability normalization failures remain hard failures; no runtime
tolerance was loosened. Provider-reported usage and SDK cost estimates remain
distinct from actual billed dollars.
