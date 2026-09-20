from __future__ import annotations
from pathlib import Path
from typing import Any
from .io import atomic_json


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# System One compilation report", "",
             f"Evidence status: **{report['status'].upper()}**", "",
             "This is a software evaluation report, not deployment approval.", ""]
    if report["status"] == "synthetic":
        lines += ["> All model predictions in this report come from the deterministic lexical fixture.",
                  "> They are NOT Jev measurements and do not establish prompt optimization gains.", ""]
    lines += ["## Held-out results", "", "| Decision | Metric | Template | Compiled |",
              "|---|---|---:|---:|"]
    baseline = report["baseline_test"]["decisions"]
    for name, values in report["compiled_test"]["decisions"].items():
        for metric in ("accuracy", "macro_f1", "brier", "log_loss", "mae", "rmse", "coverage", "selective_error"):
            if metric in values:
                def show(value):
                    return "not available" if value is None else f"{value:.4f}"
                lines.append(f"| {name} | {metric} | {show(baseline[name].get(metric))} | {show(values[metric])} |")
    lines += ["", f"Held-out objective difference: {report['test_objective_delta']:+.6f}", "",
              "A zero or negative difference is a valid result; no improvement is promised.", "",
              "## Execution accounting", "",
              f"Backend: `{report['accounting']['backend']}`",
              f"Requests attempted: {report['accounting']['requests_attempted']}",
              f"Cache hits: {report['accounting']['cache_hits']}",
              "Dollar cost: not calculated. Inspect provider billing for actual cost.", "",
              "## Boundaries", ""]
    lines.extend(f"- {text}" for text in report["limitations"])
    lines += ["", "Thresholds are fitted to empirical calibration data, not statistically certified.",
              "Full metrics, provenance, and policy settings are in report.json.", ""]
    return "\n".join(lines)


def save_compilation(directory: str | Path, program, report):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    program.save(directory / "program.s1.json")
    atomic_json(directory / "report.json", report)
    (directory / "report.md").write_text(render_markdown(report), encoding="utf-8")
    atomic_json(directory / "questions.json", {k: q.wire() for k, q in program.questions.items()})
