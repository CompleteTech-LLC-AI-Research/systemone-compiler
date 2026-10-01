# Hierarchy evaluation

`s1compiler.hierarchy_metrics` evaluates frozen graphs against registered root
examples. Supply the same `HierarchySplitGuard` used for execution, and pass an
explicit split. `evaluate_hierarchy` returns a report and the individual graph
results. `replay_hierarchy_evaluation` rebuilds that report from complete durable
evidence without provider calls. A provider, budget, or cancellation failure has
no candidate-selection score; the caller must resolve it before comparison.

The fixed selection objective is the mean root utility across **all** examples.
A completed Choice or Noul output earns 1 for an exact match and 0 otherwise.
A completed Score output earns `max(0, 1 - absolute_error / (scale_length - 1))`.
Public decision weights combine outputs. A review-required or incomplete root
earns 0. This prevents a router that skips hard examples or reviews everything
from appearing to improve quality. The report also exposes completed-only
diagnostics, coverage, stage statuses, route counts, and final-output errors;
those diagnostics do not replace the primary objective.

Probability diagnostics are scoped to completed **full-contract** Choice and
Noul outputs. Conditional child probabilities are not a global public-label
distribution, so Brier, log loss, and calibration are unavailable for them.
Scores have numeric MAE/RMSE diagnostics instead. The report gives reasons for
exclusions and labels synthetic execution explicitly. Unknown token usage stays
unknown, and an all-cache run has no uncached native latency estimate.

`paired_flat_hierarchy` accepts repeated seeds only when the flat and graph
records have identical ordered root IDs, groups, and gold labels. It reports
descriptive per-seed and per-cluster deltas. Stage observations are grouped by
root cluster; they are not independent samples. Oracle path matches, when
supplied, remain diagnostic and never enter the measured objective. The research
protocol and statistical inference are implemented in the [study runner](HIERARCHY_STUDY.md);
only the human review inputs for a live run remain open.
