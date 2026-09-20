# System One compilation report

Evidence status: **SYNTHETIC**

This is a software evaluation report, not deployment approval.

> All model predictions in this report come from the deterministic lexical fixture.
> They are NOT Jev measurements and do not establish prompt optimization gains.

## Held-out results

| Decision | Metric | Template | Compiled |
|---|---|---:|---:|
| department | accuracy | 0.4167 | 0.4167 |
| department | macro_f1 | 0.3654 | 0.3654 |
| department | brier | 0.7541 | 0.7541 |
| department | log_loss | 1.3479 | 1.3479 |
| department | coverage | 0.0833 | 0.0000 |
| department | selective_error | 0.0000 | not available |
| urgent | accuracy | 0.1667 | 0.8333 |
| urgent | macro_f1 | 0.1429 | 0.7000 |
| urgent | brier | 0.2531 | 0.2531 |
| urgent | log_loss | 0.6911 | 0.6911 |
| urgent | coverage | 0.0833 | 0.1667 |
| urgent | selective_error | 0.0000 | 0.5000 |
| frustration | mae | 0.7215 | 0.7215 |
| frustration | rmse | 0.8303 | 0.8303 |
| frustration | coverage | 0.0000 | 0.4167 |
| frustration | selective_error | not available | 0.2000 |

Held-out objective difference: +0.000000

A zero or negative difference is a valid result; no improvement is promised.

## Execution accounting

Backend: `mock-lexical/v1`
Requests attempted: 36
Cache hits: 12
Dollar cost: not calculated. Inspect provider billing for actual cost.

## Boundaries

- Synthetic fixture metrics never establish Jev quality or optimization gains.
- A held-out test score is descriptive; choose deployment acceptance criteria independently.
- Changing labels, Score meaning, or target model requires a new dataset version and evaluation.
- Selection is sequential structure-then-wording; not a global joint-search optimizer.
- No autonomous actions, arbitrary code execution, or trained model weights are included.

Thresholds are fitted to empirical calibration data, not statistically certified.
Full metrics, provenance, and policy settings are in report.json.
