"""Prepare, preregister, freeze and evaluate the five-task Jev research suite.

Usage: python -m s1compiler.research --help
No provider calls occur in prepare, register, report, or verify operations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

from .architect import template_program
from .banking77 import IdentityTeacher, make_backend, prompt_chars, select_arm
from .errors import ConfigurationError, DataError
from .io import atomic_json, canonical, fingerprint, json_loads, load_document
from .metrics import evaluate
from .models import Program
from .policy import fit_policies
from .research_data import CARDS, TASKS, load_dataset, prepare, smoke_subset
from .research_stats import clustered_pair, holm, np_module, strata_metrics, task_metrics
from .research_teacher import AuditedDSPyTeacher
from .backends import Response
from .runtime import Runtime, apply_policy, normalize_answers


def now():
    return datetime.now(timezone.utc).isoformat()


def implementation():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}


def environment():
    # Real installed distribution versions, not a fabricated dependency lock.
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}


def report_dependencies():
    np_module()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot  # noqa: F401


def envelope_write(path, content):
    atomic_json(path, {"content": content, "sha256": fingerprint(content)})


def envelope_read(path):
    value = load_document(path)
    if set(value) != {"content", "sha256"} or fingerprint(value["content"]) != value["sha256"]:
        raise DataError("Research artifact checksum mismatch.")
    return value["content"]


def arm_names(seeds):
    return ["A"] + [f"{arm}-{seed}" for seed in seeds for arm in ("B", "C", "D")]


def register(data_root, out, *, tasks=TASKS, seeds=(7, 17, 29), search_calls=100000, teacher_calls=30,
             teacher_model="deepseek/deepseek-flash", backend="mock", max_chars=24000,
             max_output_tokens=8192, bootstrap=5000, permutations=10000, analysis_seed=90210,
             teacher_response_model=None):
    report_dependencies()
    if (out.exists() or not tasks or len(set(tasks)) != len(tasks) or not set(tasks).issubset(TASKS)
            or not seeds or len(set(seeds)) != len(seeds) or teacher_calls < 1 or max_chars < 1
            or max_output_tokens < 1 or bootstrap < 100 or permutations < 100 or not teacher_model
            or backend not in {"mock", "typesafe"}):
        raise ConfigurationError("Invalid registration configuration or output already exists.")
    registered = {}
    for task in tasks:
        path = (data_root / task).resolve()
        source, splits, manifest, _ = load_dataset(path)
        if backend == "typesafe" and (manifest.get("label_origin") != "upstream_human_annotations" or manifest.get("smoke_only")):
            raise ConfigurationError("Live benchmark registration requires audited upstream human labels.")
        minimum = 3*len(splits["validation"])+2*min(3, len(splits["train"]))
        if search_calls < minimum:
            raise ConfigurationError(f"{task}: search_calls must be at least {minimum}; no automatic increase.")
        if prompt_chars(template_program(source)) > max_chars:
            raise ConfigurationError(f"{task}: baseline exceeds common prompt ceiling.")
        n_arms = len(arm_names(seeds))
        registered[task] = {"path": str(path), "manifest_sha256": fingerprint(manifest),
            "manifest": manifest, "primary": CARDS[task]["primary"], "direction": CARDS[task]["direction"],
            "practical_delta": CARDS[task]["practical_delta"],
            "request_ceiling_selection": 2*len(seeds)*search_calls+n_arms*len(splits["calibration"]),
            "request_ceiling_test": n_arms*len(splits["test"]),
            "teacher_signature_ceiling": len(seeds)*(1+2*teacher_calls),
            "teacher_provider_request_ceiling": 2*len(seeds)*(1+2*teacher_calls)}
    content = {"format": "jev-research-protocol/v1", "registered_at": now(), "datasets": registered,
        "seeds": list(seeds), "backend": backend, "teacher_model": teacher_model,
        "teacher_response_model": teacher_response_model or teacher_model.split("/", 1)[-1],
        "teacher_provider_calls_per_signature_limit": 2,
        "search_calls_per_arm_seed": search_calls, "teacher_calls_per_search_arm_seed": teacher_calls,
        "teacher_max_output_tokens": max_output_tokens, "max_prompt_chars": max_chars,
        "review_error_target": .05, "minimum_calibration_accepted": 30,
        "analysis": {"bootstrap": bootstrap, "permutations": permutations, "seed": analysis_seed,
                     "primary_family": [f"{task}:C-{other}" for task in TASKS for other in ("A", "D")],
                     "secondary": "C-B and slices are exploratory; no confirmatory claims.",
                     "estimand": "Mean paired task-metric difference across the fixed optimization seeds; "
                                  "example-weighted metrics and cluster resampling shared across arms/seeds.",
                     "binary_threshold": .5, "score_classification": "argmax level probability; lowest-index tie",
                     "multiplicity": "Holm over 10 prespecified primary tests; missing tests p=1. "
                                     "Bonferroni cluster bootstrap intervals use family size 10.",
                     "goemotions": "All 28 labels, fixed .5 threshold, zero-division F1=0; no joint probabilities."},
        "arms": {"A": "authored baseline", "B": "one-pass rewrite, three train examples, no validation selection",
                 "C": "DSPy/GEPA all-component feedback search", "D": "independent rewrites selected on validation"},
        "selection_objective": "existing typed objective: 1-Brier/2 Choice, 1-Brier Noul, 1-normalized MAE Score",
        "direct_teacher_reference": "not included; separate protocol and implementation needed for fair direct-LM inference",
        "budget_limits": "Hard Jev request, DSPy signature and synchronous teacher LM-forward ceilings; "
                         "per-call output-token limit. Provider-reported usage is metered when available. "
                         "Aggregate tokens and dollars are not capped; SDK cost estimates are not bills. "
                         "Equal ceilings, not equal spend.",
        "test_order": "Preseeded random arm order within each example; exactly one request per arm/example. No test retries.",
        "implementation_sha256": implementation(), "environment": environment(),
        "deployment_approved": False,
        "limitations": ["Unknown public-data pretraining contamination; fresh application holdout remains necessary.",
            "Authored label definitions and optimized wording require human semantic review.",
            "No formal power claim; report observed uncertainty, clusters and seed dispersion.",
            "No overall score combining heterogeneous benchmark metrics.",
            "Derived datasets are not directly comparable with official leaderboard results."]}
    envelope_write(out, content)
    return content


def validate_protocol(protocol, data_root=None):
    if protocol["format"] != "jev-research-protocol/v1" or protocol["implementation_sha256"] != implementation():
        raise DataError("Protocol format or implementation differs from registration.")
    report_dependencies()
    datasets = {}
    for task, entry in protocol["datasets"].items():
        data = load_dataset(data_root / task if data_root else Path(entry["path"]))
        if fingerprint(data[2]) != entry["manifest_sha256"]:
            raise DataError("Registered dataset changed.")
        datasets[task] = data
    return datasets


def select(protocol, out, *, backend_factory=None, teacher_factory=None, data_root=None):
    datasets = validate_protocol(protocol, data_root)  # All tasks checked before a provider is constructed.
    if out.exists():
        raise ConfigurationError("Choose a fresh selection directory.")
    live = protocol["backend"] == "typesafe"
    backend_factory = backend_factory or (lambda n: make_backend(live, n))
    teacher_factory = teacher_factory or (lambda n: AuditedDSPyTeacher(protocol["teacher_model"], allow_paid=True,
        share_feedback=True, max_calls=n, max_tokens=protocol["teacher_max_output_tokens"],
        expected_response_model=protocol["teacher_response_model"],
        max_provider_calls=n*protocol["teacher_provider_calls_per_signature_limit"]) if live else IdentityTeacher())
    out.mkdir(parents=True, exist_ok=False)
    frozen = {"format": "jev-research-frozen/v1", "protocol": protocol, "programs": {},
              "synthetic": not live, "started_at": now(), "deployment_approved": False}
    envelope_write(out / "started.json", frozen)
    current, phase, backend, teacher = None, None, None, None
    try:
        for task, (source, splits, _, _) in datasets.items():
            for name in arm_names(protocol["seeds"]):
                arm = name[0]
                seed = protocol["seeds"][0] if name == "A" else int(name[2:])
                current, phase = f"{task}/{name}", "selection"
                teacher, backend = None, None
                entry = {"task": task, "arm": name, "started_at": now()}
                backend = backend_factory(protocol["search_calls_per_arm_seed"])
                try:
                    if backend.synthetic != (not live):
                        raise ConfigurationError("Search backend mode mismatch.")
                    teacher = teacher_factory(1 if arm == "B" else protocol["teacher_calls_per_search_arm_seed"]) if arm != "A" else None
                    selected, history = select_arm(arm, template_program(source), splits, backend, teacher, seed=seed,
                        search_calls=protocol["search_calls_per_arm_seed"],
                        proposals=protocol["teacher_calls_per_search_arm_seed"], max_chars=protocol["max_prompt_chars"])
                    entry.update(history=history, selection_accounting=backend.accounting(),
                                 teacher_accounting=teacher.accounting() if teacher else None)
                finally:
                    backend.close()
                phase = "calibration"
                backend = backend_factory(len(splits["calibration"]))
                try:
                    if backend.synthetic != (not live):
                        raise ConfigurationError("Calibration backend mode mismatch.")
                    _, predictions = evaluate(selected, splits["calibration"], backend)
                    selected, fit = fit_policies(selected, splits["calibration"], predictions,
                        max_error=protocol["review_error_target"], min_samples=protocol["minimum_calibration_accepted"])
                    entry.update(calibration_fit=fit, calibration_accounting=backend.accounting())
                finally:
                    backend.close()
                selected.provenance = {"status": "synthetic" if not live else "benchmark_frozen",
                    "deployment_approved": False, "task": task, "arm": name, "seed": seed,
                    "protocol_sha256": fingerprint(protocol)}
                path = f"{task}-{name}.s1.json"
                selected.save(out / path)
                entry.update(file=path, artifact_sha256=fingerprint(selected.model_dump(mode="json")),
                             content_sha256=selected.content_hash, prompt_chars=prompt_chars(selected), completed_at=now())
                frozen["programs"][current] = entry
                atomic_json(out / f"{task}-{name}.selection.json", entry)
        frozen["completed_at"] = now()
        envelope_write(out / "frozen.json", frozen)
        return frozen
    except Exception as exc:
        atomic_json(out / "failure.json", {"phase": phase, "arm": current, "exception_type": type(exc).__name__,
            "backend": backend.accounting() if backend else None, "teacher": teacher.accounting() if teacher else None,
            "completed_arms": list(frozen["programs"]), "at": now(), "synthetic": not live})
        raise


def load_frozen(directory, *, reviewed=None, data_root=None):
    frozen = envelope_read(directory / "frozen.json")
    protocol = frozen["protocol"]
    if not frozen["synthetic"] and reviewed != fingerprint(frozen):
        raise ConfigurationError("Review all frozen prompts and supply their --reviewed-manifest digest.")
    datasets = validate_protocol(protocol, data_root)
    expected = {f"{task}/{name}" for task in datasets for name in arm_names(protocol["seeds"])}
    if set(frozen["programs"]) != expected or frozen["synthetic"] != (protocol["backend"] == "mock"):
        raise DataError("Incomplete or mixed-mode frozen suite.")
    programs = {}
    for key, entry in frozen["programs"].items():
        task, arm = key.split("/")
        if entry["file"] != f"{task}-{arm}.s1.json":
            raise DataError("Unexpected frozen artifact path.")
        program = Program.load(directory / entry["file"])
        if fingerprint(program.model_dump(mode="json")) != entry["artifact_sha256"]:
            raise DataError("Program changed after freeze.")
        programs[key] = program
    return frozen, datasets, programs


def test_suite(directory, out, *, reviewed=None, backend_factory=None, data_root=None):
    frozen, datasets, programs = load_frozen(directory, reviewed=reviewed, data_root=data_root)
    protocol = frozen["protocol"]
    live = not frozen["synthetic"]
    backend_factory = backend_factory or (lambda n: make_backend(live, n))
    if out.exists():
        raise ConfigurationError("Choose a fresh test directory.")
    # Exclusive marker consumes this frozen experiment even if a provider fails midway.
    with (directory / "test-started.json").open("x", encoding="utf-8") as stream:
        stream.write(canonical({"at": now(), "frozen_sha256": fingerprint(frozen), "out": str(out.resolve())}))
    out.mkdir(parents=True, exist_ok=False)
    envelope_write(out / "frozen.json", frozen)
    for key, program in programs.items():
        program.save(out / frozen["programs"][key]["file"])
    summary = {"format": "jev-research-execution/v1", "frozen_sha256": fingerprint(frozen),
               "started_at": now(), "synthetic": not live, "arms": {}, "status": "running"}
    current, phase = None, "test"
    backends, streams = {}, {}
    try:
        for task, (source, splits, _, metadata) in datasets.items():
            names = arm_names(protocol["seeds"])
            for name in names:
                key = f"{task}/{name}"
                backends[key] = backend_factory(len(splits["test"]))
                if backends[key].synthetic != (not live):
                    raise ConfigurationError("Test backend mode mismatch.")
                streams[key] = (out / f"{task}-{name}.evidence.jsonl").open("x", encoding="utf-8")
            rng = random.Random(protocol["analysis"]["seed"])
            # Interleave arms per example to reduce confounding from time-varying service latency.
            for row in splits["test"]:
                order = list(names)
                rng.shuffle(order)
                for name in order:
                    current = f"{task}/{name}"
                    result = Runtime(programs[current], backends[current]).run(row.state)
                    record = {"id": row.id, "group": row.group, "stratum": metadata[row.id]["stratum"],
                        "gold": row.expected, "decisions": result["decisions"], "answers": result["answers"],
                        "usage": result["usage"],
                        "latency_ms": result["latency_ms"], "cache_hit": result["cache_hit"],
                        "model": result["model"], "synthetic": result["synthetic"],
                        "program_sha256": result["program_sha256"]}
                    streams[current].write(canonical(record)+"\n")
                    streams[current].flush()
            for name in names:
                key = f"{task}/{name}"
                streams[key].close()
                evidence_path = out / f"{task}-{name}.evidence.jsonl"
                summary["arms"][key] = {"file": evidence_path.name,
                    "sha256": file_hash(evidence_path), "rows": len(splits["test"]),
                    "accounting": backends[key].accounting()}
                backends[key].close()
                del backends[key]
        summary.update(status="synthetic" if not live else "measured", completed_at=now())
        envelope_write(out / "execution.json", summary)
    except Exception as exc:
        summary.update(status="failed", failure={"phase": phase, "arm": current,
            "exception_type": type(exc).__name__, "at": now()},
            partial_accounting={key: b.accounting() for key, b in backends.items()})
        envelope_write(out / "execution.json", summary)
        raise
    finally:
        for stream in streams.values():
            if not stream.closed:
                stream.close()
        for backend in backends.values():
            backend.close()
    return report(out, data_root=data_root)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_evidence(path, entry, rows, metadata, program, synthetic):
    if file_hash(path) != entry["sha256"]:
        raise DataError("Evidence checksum mismatch.")
    with path.open(encoding="utf-8") as stream:
        evidence = [json_loads(line) for line in stream]
    if len(evidence) != len(rows) or entry["rows"] != len(rows):
        raise DataError("Incomplete evidence; missing predictions cannot be silently excluded.")
    for result, row in zip(evidence, rows):
        if (result["id"] != row.id or result["gold"] != row.expected or result["group"] != row.group
                or result["stratum"] != metadata[row.id]["stratum"] or result["synthetic"] != synthetic
                or result["model"] != program.model or result["program_sha256"] != program.content_hash):
            raise DataError("Evidence identity, model, gold, or grouping mismatch.")
        normalize_answers(program, Response(result["answers"], result["model"], synthetic=synthetic))
        # Already normalized at execution. Do not repeatedly renormalize floats during replay.
        if apply_policy(program, result["answers"]) != result["decisions"]:
            raise DataError("Stored decisions do not reproduce from typed answers and frozen policy.")
    return evidence


def report(directory, *, data_root=None):
    """Recompute publication tables from immutable evidence, without calling any model."""
    frozen = envelope_read(directory / "frozen.json")
    protocol = frozen["protocol"]
    datasets = validate_protocol(protocol, data_root)
    execution = envelope_read(directory / "execution.json")
    if execution["frozen_sha256"] != fingerprint(frozen):
        raise DataError("Execution and frozen manifest differ.")
    if execution["synthetic"] != frozen["synthetic"]:
        raise DataError("Execution provenance differs from the frozen experiment.")
    if execution["status"] not in {"synthetic", "measured"}:
        raise DataError("Execution incomplete or failed; no success report can be generated.")
    if execution["status"] != ("synthetic" if frozen["synthetic"] else "measured"):
        raise DataError("Execution status misrepresents synthetic/live provenance.")
    expected = {f"{task}/{name}" for task in datasets for name in arm_names(protocol["seeds"])}
    if set(execution["arms"]) != expected:
        raise DataError("Execution omits prespecified arms.")
    result = {"format": "jev-research-report/v1", "synthetic": frozen["synthetic"],
        "protocol_sha256": fingerprint(protocol), "frozen_sha256": fingerprint(frozen),
        "execution_sha256": fingerprint(execution), "analysis": protocol["analysis"],
        "datasets": {}, "primary_tests": {}, "deployment_approved": False,
        "limitations": protocol["limitations"], "cost_note": protocol["budget_limits"],
        "environment": protocol["environment"], "report_environment": environment()}
    p_values = {key: 1. for key in protocol["analysis"]["primary_family"]}
    for task, (source, splits, manifest, metadata) in datasets.items():
        evidence, arms = {}, {}
        for name in arm_names(protocol["seeds"]):
            key = f"{task}/{name}"
            entry = execution["arms"][key]
            if entry["file"] != f"{task}-{name}.evidence.jsonl":
                raise DataError("Unexpected evidence path.")
            artifact = frozen["programs"][key]
            if artifact["file"] != f"{task}-{name}.s1.json":
                raise DataError("Unexpected program path in replay.")
            program = Program.load(directory / artifact["file"])
            if fingerprint(program.model_dump(mode="json")) != artifact["artifact_sha256"]:
                raise DataError("Frozen prompt changed before report replay.")
            evidence[name] = read_evidence(directory / entry["file"], entry, splits["test"], metadata,
                                            program, frozen["synthetic"])
            arms[name] = {"metrics": task_metrics(task, evidence[name], source),
                          "slices": strata_metrics(task, evidence[name], source),
                          "runtime_accounting": entry["accounting"],
                          "compilation": frozen["programs"][key]}
        task_report = {"primary_metric": CARDS[task]["primary"], "direction": CARDS[task]["direction"],
                       "manifest": manifest, "arms": arms, "comparisons": {}}
        for comparator in ("A", "D", "B"):
            left = [evidence[f"C-{seed}"] for seed in protocol["seeds"]]
            right = [evidence["A" if comparator == "A" else f"{comparator}-{seed}"] for seed in protocol["seeds"]]
            analysis = protocol["analysis"]
            comparison = clustered_pair(task, left, right, source, bootstrap=analysis["bootstrap"],
                permutations=analysis["permutations"], seed=analysis["seed"], family_size=len(p_values))
            comparison["status"] = "primary" if comparator != "B" else "exploratory"
            task_report["comparisons"][f"C-{comparator}"] = comparison
            if comparator != "B":
                key = f"{task}:C-{comparator}"
                p_values[key] = comparison["two_sided_cluster_randomization_p"]
                result["primary_tests"][key] = comparison
        result["datasets"][task] = task_report
    adjusted = holm(p_values)
    for key in p_values:
        if key in result["primary_tests"]:
            result["primary_tests"][key]["holm_adjusted_p"] = adjusted[key]
        else:
            result["primary_tests"][key] = {"status": "not_executed", "p_for_family_accounting": 1.,
                                            "holm_adjusted_p": adjusted[key]}
    result["full_primary_family_executed"] = len(datasets) == len(TASKS)
    atomic_json(directory / "report.json", result)
    render_report(directory, result)
    return result


def render_report(directory, result):
    lines = ["# Jev prompt-engineering research suite", "",
             "**SYNTHETIC SOFTWARE CHECK — NOT JEV QUALITY EVIDENCE.**" if result["synthetic"] else
             "**Measured frozen Jev programs; not production approved.**", "",
             "Protocol SHA256: `"+result["protocol_sha256"]+"`", "",
             "## Task results", "",
             "| Task | Arm | Primary metric | Value | Rows | Clusters | p95 latency (ms) |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    table = []
    for task, data in result["datasets"].items():
        for name, arm in data["arms"].items():
            metric = arm["metrics"]
            lines.append(f"| {task} | {name} | {data['primary_metric']} | {metric['primary']:.6f} | "
                         f"{metric['n']} | {metric['n_clusters']} | {metric['latency_ms']['p95']} |")
            table.append({"task": task, "arm": name, "metric": data["primary_metric"], "value": metric["primary"],
                          "n": metric["n"], "clusters": metric["n_clusters"], "synthetic": result["synthetic"]})
    lines.extend(["", "## Prespecified primary comparisons", "",
                  "Positive improvement always favors C (MAE direction reversed). Intervals are cluster-bootstrap",
                  "Bonferroni intervals across ten planned comparisons. P values are two-sided cluster randomization",
                  "tests with Holm correction. Seeds are fixed; their predictions are not independent examples.", "",
                  "| Comparison | Improvement | Family-adjusted interval | Holm p | Seed delta SD |",
                  "| --- | ---: | --- | ---: | ---: |"])
    for key, comparison in result["primary_tests"].items():
        if comparison["status"] == "not_executed":
            lines.append(f"| {key} | not executed | — | — | — |")
        else:
            lo, hi = comparison["improvement_bonferroni_ci"]
            lines.append(f"| {key} | {comparison['improvement']:.6f} | [{lo:.6f}, {hi:.6f}] | "
                         f"{comparison['holm_adjusted_p']:.6f} | {comparison['seed_delta_sd']} |")
    lines.extend(["", "## Provenance and interpretation", "",
        "Full distributions, per-label metrics, confusion matrices, reliability bins, ANLI round slices,",
        "CLINC out-of-scope diagnostics, review coverage/error, and compilation/runtime accounting are in report.json.",
        "Evidence JSONL contains labels and predictions but no input text. Hashes do not anonymize it.", "",
        result["cost_note"], "", "Teacher reported tokens and SDK cost estimates are separate; billed dollars stay null.",
        "No automatic gain or production-readiness claim is made, especially for synthetic or one-seed runs.", ""])
    for task, data in result["datasets"].items():
        manifest = data["manifest"]
        lines.extend([f"- {task}: {manifest['variant']}; {manifest['card']['holdout']}; "
                      f"{manifest['removed_holdout_rows']} holdout rows removed. "
                      f"Source: {manifest['card']['source']}"])
    lines.extend(["", *[f"- {text}" for text in result["limitations"]]])
    (directory / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    # JSON tables avoid lossy spreadsheet coercion and keep explicit provenance.
    atomic_json(directory / "primary-table.json", {"rows": table, "synthetic": result["synthetic"]})
    plot_report(directory, result)


def plot_report(directory, result):
    """Standalone publication figures; each task keeps its own metric scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"svg.hashsalt": result["protocol_sha256"], "font.family": "DejaVu Sans", "font.size": 9}):
        tasks = list(result["datasets"])
        fig, axes = plt.subplots(len(tasks), 1, figsize=(8, 2.0*len(tasks)), squeeze=False, layout="constrained")
        for ax, task in zip(axes[:, 0], tasks):
            for y, comparator in enumerate(("A", "D")):
                comparison = result["primary_tests"][f"{task}:C-{comparator}"]
                value = comparison["improvement"]
                lo, hi = comparison["improvement_bonferroni_ci"]
                ax.plot([lo, hi], [y, y], color="#24527a", linewidth=2)
                ax.scatter([value], [y], color="#24527a", s=30, zorder=3)
            ax.axvline(0, color="#666666", linestyle="--", linewidth=1)
            ax.set_yticks([0, 1], ["C vs authored baseline", "C vs independent search"])
            ax.set_ylim(-.5, 1.5)
            ax.set_title(task, loc="left", fontweight="bold")
            metric = result["datasets"][task]["primary_metric"]
            ax.set_xlabel(("Reduction in " if metric == "mae" else "Increase in ")+metric+" (positive favors C)")
            ax.grid(axis="x", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(("SYNTHETIC CHECK — NOT JEV RESULTS\n" if result["synthetic"] else "Jev prompt-engineering study\n")+
                     "Paired cluster bootstrap; family-adjusted intervals; fixed optimization seeds", fontsize=12)
        fig.savefig(directory / "primary-effects.svg", metadata={"Date": None})
        fig.savefig(directory / "primary-effects.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(directory / "primary-effects.png", dpi=150, metadata={"Software": "systemone-compiler research suite"})
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "smoke-data", "register", "select", "test", "report", "verify"])
    parser.add_argument("--task", choices=TASKS, nargs="+", default=list(TASKS))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("runs/research-downloads"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--clean-derived", action="store_true")
    parser.add_argument("--smoke-groups", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--search-calls", type=int, default=100000)
    parser.add_argument("--teacher-calls", type=int, default=30)
    parser.add_argument("--teacher-model", default="deepseek/deepseek-flash")
    parser.add_argument("--teacher-response-model")
    parser.add_argument("--backend", choices=["mock", "typesafe"], default="mock")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--reviewed-manifest")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--share-feedback", action="store_true")
    parser.add_argument("--acknowledge-budget-limits", action="store_true")
    args = parser.parse_args(argv)
    if args.command in {"prepare", "smoke-data", "register", "select", "test"} and not args.out:
        parser.error("--out is required")
    if args.command in {"register", "verify", "smoke-data"} and not args.data:
        parser.error("--data is required")
    if args.command == "prepare":
        for task in args.task:
            manifest = prepare(task, args.out / task, args.cache, clean=args.clean_derived)
            print(canonical({"task": task, "variant": manifest["variant"],
                             "counts": {k: v["n"] for k, v in manifest["splits"].items()}}))
    elif args.command == "smoke-data":
        for task in args.task:
            manifest = smoke_subset(args.data / task, args.out / task, groups=args.smoke_groups)
            print(canonical({"task": task, "smoke_only": True, "counts": {k: v["n"] for k, v in manifest["splits"].items()}}))
    elif args.command == "verify":
        for task in args.task:
            data = load_dataset(args.data / task)
            print(canonical({"task": task, "verified": True, "sha256": fingerprint(data[2])}))
    elif args.command == "register":
        protocol = register(args.data, args.out, tasks=args.task, seeds=args.seeds, search_calls=args.search_calls,
            teacher_calls=args.teacher_calls, teacher_model=args.teacher_model, backend=args.backend,
            bootstrap=args.bootstrap, permutations=args.permutations, teacher_response_model=args.teacher_response_model)
        print(canonical({"protocol_sha256": fingerprint(protocol), "provider_calls": 0,
                         "requests": sum(v["request_ceiling_selection"]+v["request_ceiling_test"]
                                         for v in protocol["datasets"].values())}))
    elif args.command == "report":
        if not args.frozen:
            parser.error("report requires --frozen pointing to the test-results directory")
        report(args.frozen, data_root=args.data)
        print(canonical({"report": str(args.frozen / "report.md"), "provider_calls": 0}))
    else:
        if args.command == "select":
            if not args.protocol:
                parser.error("select requires --protocol")
            protocol = envelope_read(args.protocol)
        else:
            if not args.frozen:
                parser.error("test requires --frozen pointing to the selection directory")
            protocol = envelope_read(args.frozen / "frozen.json")["protocol"]
        if protocol["backend"] == "typesafe":
            if not args.allow_paid or not args.acknowledge_budget_limits:
                parser.error("Live runs require --allow-paid and --acknowledge-budget-limits")
            if args.command == "select" and not args.share_feedback:
                parser.error("Teacher examples/traces require --share-feedback")
        if args.command == "select":
            frozen = select(protocol, args.out, data_root=args.data)
            print(canonical({"frozen_sha256": fingerprint(frozen), "synthetic": frozen["synthetic"]}))
        else:
            test_suite(args.frozen, args.out, reviewed=args.reviewed_manifest, data_root=args.data)
            print(canonical({"report": str(args.out / "report.md")}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Research workflow stopped: {type(exc).__name__}. See saved audit/failure records.") from None
