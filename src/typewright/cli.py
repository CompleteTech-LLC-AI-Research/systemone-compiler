from __future__ import annotations
import argparse
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

from pydantic import ValidationError

from .architect import DSPyTeacher, template_program
from .backends import AnswerCache, ManagedBackend, MockBackend, TypeSafeBackend
from .compiler import CompileOptions, Compiler, versions
from .data import read_jsonl
from .errors import ConfigurationError, S1Error
from .hardening import propose_cases
from .hierarchy import HierarchyArtifact, HierarchySource, load_artifact, lower_hierarchy
from .hierarchy_compiler import HierarchyCompileOptions, HierarchyCompiler
from .hierarchy_data import HierarchySplitGuard, read_hierarchy_jsonl
from .hierarchy_metrics import evaluate_hierarchy
from .hierarchy_runtime import HierarchyRuntime
from .hierarchy_validation import validate_hierarchy_compile_inputs
from .io import atomic_json, load_document
from .metrics import evaluate
from .models import Program, UseCase, project_state
from .reporting import save_compilation
from .runtime import Runtime


def backend_args(parser):
    parser.add_argument("--backend", choices=["mock", "typesafe"], default="mock")
    parser.add_argument("--allow-paid", action="store_true", help="Authorize this invocation's live provider calls.")
    parser.add_argument("--max-calls", type=int, default=500, help="Maximum uncached TypeSafe SDK request attempts.")
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--cache", type=Path, help="Opt-in SQLite output cache; treat it as sensitive.")
    parser.add_argument("--no-cache", action="store_true", help="Disable even the default in-memory cache.")


def teacher_args(parser, *, add_paid=False):
    if add_paid:
        parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--share-feedback", action="store_true", help="Consent to send training examples to teacher.")
    parser.add_argument("--teacher-model", default=os.getenv("S1_TEACHER_MODEL"))
    parser.add_argument("--teacher-max-calls", type=int, default=20)
    parser.add_argument("--teacher-max-tokens", type=int, default=4096)
    parser.add_argument("--teacher-temperature", type=float, default=None,
                        help="Omitted by default; current reasoning models reject sampling parameters.")
    parser.add_argument("--teacher-timeout", type=float, default=120)


def make_backend(args):
    if args.cache and args.no_cache:
        raise ConfigurationError("Choose --cache or --no-cache, not both.")
    if args.max_calls < 1 or args.request_timeout <= 0:
        raise ConfigurationError("Request budget and timeout must be positive.")
    provider = MockBackend() if args.backend == "mock" else TypeSafeBackend(
        allow_paid=args.allow_paid, timeout=args.request_timeout)
    cache = None if args.no_cache else AnswerCache(args.cache)
    return ManagedBackend(provider, max_calls=args.max_calls, cache=cache)


def make_teacher(args):
    return DSPyTeacher(args.teacher_model, allow_paid=args.allow_paid, share_feedback=args.share_feedback,
                       max_calls=args.teacher_max_calls, max_tokens=args.teacher_max_tokens,
                       temperature=args.teacher_temperature, timeout=args.teacher_timeout)


def source_for(program):
    return UseCase(name=program.name, model=program.model, state=program.state, decisions=program.decisions)


def init_project(directory: Path, *, starter: str = "flat"):
    if starter not in {"flat", "hierarchy"}:
        raise ConfigurationError("Unknown starter; choose flat or hierarchy.")
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ConfigurationError("Target directory must be missing or empty; existing files will not be overwritten.")
    source = Path(__file__).parent / "templates" / ("support" if starter == "flat" else "hierarchy")
    shutil.copytree(source, directory, dirs_exist_ok=True)


def ensure_new_output(directory: Path):
    if directory.exists() and any(directory.iterdir()):
        raise ConfigurationError("Output directory is not empty; choose a new run directory.")
    directory.mkdir(parents=True, exist_ok=True)


def build_parser():
    parser = argparse.ArgumentParser(prog="s1", description="Declare, optimize, and run typed Jev use cases.")
    parser.add_argument("--version", action="version", version="typewright 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create an editable support-triage project and synthetic dataset.")
    init.add_argument("directory", type=Path)
    init.add_argument("--starter", choices=["flat", "hierarchy"], default="flat")
    draft = sub.add_parser("draft", help="Compile a typed draft, optionally using a DSPy architect.")
    draft.add_argument("spec", type=Path)
    draft.add_argument("--out", type=Path, required=True)
    draft.add_argument("--architect", choices=["template", "dspy"], default="template")
    teacher_args(draft, add_paid=True)
    compile_p = sub.add_parser("compile", help="Optimize, fit policy on calibration, freeze, then test.")
    compile_p.add_argument("spec", type=Path)
    for name in ("train", "validation", "calibration", "test"):
        compile_p.add_argument(f"--{name}", type=Path, required=True)
    compile_p.add_argument("--out", type=Path, required=True)
    compile_p.add_argument("--architect", choices=["template", "dspy"], default="template")
    compile_p.add_argument("--optimizer", choices=["none", "gepa"], default="none")
    compile_p.add_argument("--structural-rounds", type=int, default=0)
    compile_p.add_argument("--max-metric-calls", type=int, default=128)
    compile_p.add_argument("--seed", type=int, default=7)
    compile_p.add_argument("--max-calibration-error", type=float, default=0.05)
    compile_p.add_argument("--min-calibration-samples", type=int, default=10)
    backend_args(compile_p)
    teacher_args(compile_p)
    run = sub.add_parser("run", help="Run a compiled JSON artifact. Does not execute application actions.")
    run.add_argument("artifact", type=Path)
    run.add_argument("--state", type=Path, required=True)
    run.add_argument("--out", type=Path)
    run.add_argument("--allow-unvalidated", action="store_true", help="Explicit experimental execution of a draft.")
    backend_args(run)
    ev = sub.add_parser("evaluate", help="Evaluate a frozen artifact without fitting or changing it.")
    ev.add_argument("artifact", type=Path)
    ev.add_argument("--data", type=Path)
    for name in ("train", "validation", "calibration", "test"):
        ev.add_argument(f"--{name}", type=Path)
    ev.add_argument("--split", choices=["train", "validation", "calibration", "test"], default="test")
    ev.add_argument("--out", type=Path, required=True)
    backend_args(ev)
    doctor = sub.add_parser("doctor", help="Local dependency and contract checks; never calls a model.")
    doctor.add_argument("--check-optional", action="store_true")
    demo = sub.add_parser("demo", help="No-key, no-network end-to-end synthetic smoke test.")
    demo.add_argument("--out", type=Path, default=Path("runs/demo"))
    demo.add_argument("--starter", choices=["flat", "hierarchy"], default="flat")
    inspect_p = sub.add_parser("inspect", help="Validate artifact checksum and print its typed plan.")
    inspect_p.add_argument("artifact", type=Path)
    harden = sub.add_parser("harden", help="Create robustness proposals that require human label review.")
    harden.add_argument("spec", type=Path)
    harden.add_argument("--data", type=Path, required=True)
    harden.add_argument("--out", type=Path, required=True)
    schema = sub.add_parser("schema", help="Export Pydantic JSON schemas for editors and tooling.")
    schema.add_argument("--out", type=Path, required=True)
    export = sub.add_parser("export-playground", help="Export native state/model/questions; policies remain local.")
    export.add_argument("artifact", type=Path)
    export.add_argument("--state", type=Path, help="Root state for a directly resolvable stage or flat program.")
    export.add_argument("--resolved-state", type=Path,
                        help="Concrete inputs already resolved for the selected leaf; does not prove its route.")
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--stage", help="Expanded leaf stage ID for a hierarchy artifact.")
    return parser


def doctor(check_optional):
    details = {"python": sys.version.split()[0], "versions": versions(), "network_calls": 0,
               "typesafe_key_present": bool(os.getenv("TYPESAFE_API_KEY")),
               "teacher_model_set": bool(os.getenv("S1_TEACHER_MODEL")), "contracts": {}}
    if check_optional:
        for module_name in ("typesafe_sdk", "dspy", "gepa"):
            try:
                module = __import__(module_name)
                if module_name == "typesafe_sdk":
                    for name, kwargs in (("Noul", {"instructions": "Is the flag present?"}),
                        ("Choice", {"instructions": "Pick a label", "criteria": {"a": "A", "b": "B"}}),
                        ("Score", {"instructions": "Rate severity", "criteria": ["Low", "High"]})):
                        getattr(module, name)(**kwargs)
                    params = inspect.signature(module.TypeSafeClient.system_one).parameters
                    if not {"state", "questions", "model"}.issubset(params):
                        raise RuntimeError("system_one signature mismatch")
                elif module_name == "gepa":
                    params = inspect.signature(module.optimize).parameters
                    if not {"adapter", "seed_candidate", "trainset", "valset", "max_metric_calls"}.issubset(params):
                        raise RuntimeError("GEPA optimize signature mismatch")
                else:
                    for name in ("LM", "Signature", "Predict", "context", "configure_cache"):
                        if not hasattr(module, name):
                            raise RuntimeError(f"Missing DSPy entry point {name}")
                details["contracts"][module_name] = "local import/shape check passed"
            except Exception as exc:
                details["contracts"][module_name] = f"not verified: {type(exc).__name__}"
    details["ok"] = all(value.startswith("local") for value in details["contracts"].values())
    print(json.dumps(details, indent=2))
    return 0 if details["ok"] else 2


def dispatch(args):
    if args.command == "init":
        init_project(args.directory, starter=args.starter)
        name = "usecase.yaml" if args.starter == "flat" else "source.json"
        print(f"Created {args.directory}. Edit {name}; bundled labels are synthetic examples.")
    elif args.command == "doctor":
        return doctor(args.check_optional)
    elif args.command == "draft":
        if args.out.exists():
            raise ConfigurationError("Draft output already exists; choose a new filename.")
        if load_document(args.spec).get("format") == "systemone-hierarchy-source/v1":
            if args.architect != "template":
                raise ConfigurationError("Hierarchy DSPy structure search requires compile with four splits.")
            lower_hierarchy(HierarchySource.load(args.spec)).save(args.out)
            print(f"Hierarchy draft saved: {args.out}. Validate and calibrate before deployment.")
            return 0
        source = UseCase.load(args.spec)
        program = template_program(source)
        if args.architect == "dspy":
            program = make_teacher(args).propose_plan(source, program, [])
        program.save(args.out)
        print(f"Draft saved: {args.out}. No accuracy claims; evaluate before deployment.")
    elif args.command == "compile":
        if load_document(args.spec).get("format") == "systemone-hierarchy-source/v1":
            source = HierarchySource.load(args.spec)
            draft = lower_hierarchy(source)
            splits = {name: read_hierarchy_jsonl(getattr(args, name), draft)
                      for name in ("train", "validation", "calibration", "test")}
            validate_hierarchy_compile_inputs(source, splits)
            options = HierarchyCompileOptions(architect=args.architect, optimizer=args.optimizer,
                structural_rounds=args.structural_rounds, max_metric_calls=args.max_metric_calls,
                seed=args.seed, max_calibration_error=args.max_calibration_error,
                min_calibration_samples=args.min_calibration_samples)
            if args.out.exists():
                raise ConfigurationError("Output directory already exists; choose a new run directory.")
            teacher = make_teacher(args) if (args.architect == "dspy" or args.optimizer == "gepa") else None
            backend = make_backend(args)
            try:
                artifact, report = HierarchyCompiler(backend, teacher=teacher, options=options).compile(
                    source, **splits)
                ensure_new_output(args.out)
                artifact.save(args.out / "hierarchy.s1.json")
                atomic_json(args.out / "report.json", report)
                print(json.dumps({"artifact": str(args.out / "hierarchy.s1.json"),
                                  "status": report["status"], "accounting": report["accounting"]}, indent=2))
            finally:
                backend.close()
            return 0
        source = UseCase.load(args.spec)
        splits = {name: read_jsonl(getattr(args, name), source)
                  for name in ("train", "validation", "calibration", "test")}
        # Data validation precedes teacher setup and all provider calls.
        from .data import assert_disjoint
        assert_disjoint(splits, source)
        ensure_new_output(args.out)
        options = CompileOptions(architect=args.architect, optimizer=args.optimizer,
            structural_rounds=args.structural_rounds, max_metric_calls=args.max_metric_calls,
            seed=args.seed, max_calibration_error=args.max_calibration_error,
            min_calibration_samples=args.min_calibration_samples)
        teacher = make_teacher(args) if (args.architect == "dspy" or args.optimizer == "gepa" or args.structural_rounds) else None
        backend = make_backend(args)
        try:
            program, report = Compiler(backend, teacher=teacher, options=options).compile(source, **splits)
            save_compilation(args.out, program, report)
            print(json.dumps({"artifact": str(args.out / "program.s1.json"), "status": report["status"],
                              "test_objective_delta": report["test_objective_delta"],
                              "requests_attempted": report["accounting"]["requests_attempted"]}, indent=2))
        finally:
            backend.close()
    elif args.command == "demo":
        ensure_new_output(args.out)
        project = args.out / "project"
        init_project(project, starter=args.starter)
        if args.starter == "hierarchy":
            source = HierarchySource.load(project / "source.json")
            draft = lower_hierarchy(source)
            splits = {name: read_hierarchy_jsonl(project / f"{name}.jsonl", draft)
                      for name in ("train", "validation", "calibration", "test")}
            backend = ManagedBackend(MockBackend(), max_calls=100, cache=AnswerCache())
            try:
                artifact, report = HierarchyCompiler(backend, options=HierarchyCompileOptions(
                    min_calibration_samples=1)).compile(source, **splits)
                artifacts = args.out / "artifacts"
                artifacts.mkdir()
                artifact.save(artifacts / "hierarchy.s1.json")
                atomic_json(artifacts / "report.json", report)
                result = HierarchyRuntime(artifact, backend).run(load_document(project / "sample_state.json"))
                atomic_json(args.out / "sample_prediction.json", result)
                print(json.dumps({"status": "SYNTHETIC SOFTWARE SMOKE TEST — NOT JEV RESULTS",
                                  "output": str(args.out), "requests": report["accounting"]["native_calls_this_compile"],
                                  "optimizer": "none", "network_calls": 0}, indent=2))
            finally:
                backend.close()
            return 0
        source = UseCase.load(project / "usecase.yaml")
        splits = {name: read_jsonl(project / f"{name}.jsonl", source)
                  for name in ("train", "validation", "calibration", "test")}
        backend = ManagedBackend(MockBackend(), max_calls=100, cache=AnswerCache())
        try:
            program, report = Compiler(backend, options=CompileOptions(min_calibration_samples=3)).compile(source, **splits)
            save_compilation(args.out / "artifacts", program, report)
            result = Runtime(program, backend).run(load_document(project / "sample_state.json"))
            atomic_json(args.out / "sample_prediction.json", result)
            print(json.dumps({"status": "SYNTHETIC SOFTWARE SMOKE TEST — NOT JEV RESULTS",
                              "output": str(args.out), "requests": report["accounting"]["requests_attempted"],
                              "optimizer": "none", "network_calls": 0}, indent=2))
        finally:
            backend.close()
    elif args.command in {"run", "evaluate"}:
        program = load_artifact(args.artifact)
        backend = make_backend(args)
        try:
            if isinstance(program, HierarchyArtifact):
                if args.command == "run":
                    result = HierarchyRuntime(program, backend,
                        enforce_release=not args.allow_unvalidated).run(load_document(args.state))
                    if args.out:
                        atomic_json(args.out, result)
                    print(json.dumps(result, indent=2))
                    return 2 if result["status"] in {"failed", "cancelled"} else 0
                paths = {name: getattr(args, name) for name in
                         ("train", "validation", "calibration", "test")}
                if any(path is None for path in paths.values()):
                    raise ConfigurationError("Hierarchy evaluation requires --train, --validation, "
                                             "--calibration, and --test for split isolation.")
                splits = {name: read_hierarchy_jsonl(path, program) for name, path in paths.items()}
                guard = HierarchySplitGuard(program, splits)
                report, _ = evaluate_hierarchy(program, splits[args.split], backend,
                                               guard=guard, split=args.split)
                report["accounting"] = backend.accounting()
                atomic_json(args.out, report)
                print(f"Hierarchy evaluation saved: {args.out}")
                return 0
            if args.command == "run":
                runtime = Runtime(program, backend, enforce_release=not args.allow_unvalidated)
                result = runtime.run(load_document(args.state))
                if args.out:
                    atomic_json(args.out, result)
                print(json.dumps(result, indent=2))
            else:
                if args.data is None:
                    raise ConfigurationError("Flat evaluation requires --data.")
                rows = read_jsonl(args.data, source_for(program))
                report, _ = evaluate(program, rows, backend)
                report["accounting"] = backend.accounting()
                atomic_json(args.out, report)
                print(f"Evaluation saved: {args.out}")
        finally:
            backend.close()
    elif args.command == "inspect":
        program = load_artifact(args.artifact)
        if isinstance(program, HierarchyArtifact):
            print(json.dumps({"format": program.format, "graph_sha256": program.content_hash,
                "model": program.source.model, "source_to_nodes": program.source_to_nodes,
                "limits": program.limits.model_dump(mode="json"),
                "nodes": [{"id": node.id, "source_id": node.source_id,
                           "inputs": {key: value.model_dump(mode="json") for key, value in node.inputs.items()},
                           "when": node.when.model_dump(mode="json") if node.when else None,
                           "after": node.after, "on_review": node.on_review}
                          for node in program.nodes],
                "exports": [export.model_dump(mode="json") for export in program.exports],
                "final": {key: value.model_dump(mode="json") for key, value in program.final.items()},
                "final_review_gates": {key: {stage: gate.model_dump(mode="json")
                    for stage, gate in gates.items()} for key, gates in program.final_review_gates.items()},
                "provenance": program.provenance.model_dump(mode="json")}, indent=2))
        else:
            print(program.model_dump_json(indent=2))
    elif args.command == "harden":
        source = UseCase.load(args.spec)
        atomic_json(args.out, propose_cases(source, read_jsonl(args.data, source)))
        print(f"Review-required proposals saved: {args.out}")
    elif args.command == "schema":
        args.out.mkdir(parents=True, exist_ok=True)
        atomic_json(args.out / "usecase.schema.json", UseCase.model_json_schema())
        atomic_json(args.out / "program.schema.json", Program.model_json_schema())
        atomic_json(args.out / "hierarchy-source.schema.json", HierarchySource.model_json_schema())
        atomic_json(args.out / "hierarchy-artifact.schema.json", HierarchyArtifact.model_json_schema())
        print(f"Schemas saved: {args.out}")
    elif args.command == "export-playground":
        program = load_artifact(args.artifact)
        if isinstance(program, HierarchyArtifact):
            if not args.stage:
                raise ConfigurationError("A hierarchy has no single native request; select a leaf with --stage.")
            node = next((node for node in program.nodes if node.id == args.stage), None)
            if node is None:
                raise ConfigurationError(f"Unknown hierarchy leaf stage: {args.stage}")
            from .hierarchy import RootRef
            if args.resolved_state is not None:
                if args.state is not None:
                    raise ConfigurationError("Choose --state or --resolved-state for one stage, not both.")
                mapped = load_document(args.resolved_state)
            else:
                if args.state is None:
                    raise ConfigurationError("Select --state for a root-resolvable stage or --resolved-state "
                                             "for concrete leaf inputs.")
                if node.when or not all(isinstance(ref, RootRef) for ref in node.inputs.values()):
                    raise ConfigurationError(f"Stage {args.stage} needs resolved routing/derived inputs; "
                                             "a root state alone cannot export its native request.")
                root = project_state(program.source.state, load_document(args.state))
                mapped = {port: root[ref.root] if ref.root in root else ref.default
                          for port, ref in node.inputs.items()
                          if ref.root in root or ref.default is not None}
            atomic_json(args.out, {"model": node.program.model,
                "state": project_state(node.program.state, mapped),
                "questions": {name: q.wire() for name, q in node.program.questions.items()}})
            print(f"Stage request saved: {args.out}. A native one-call payload cannot represent the graph; "
                  "supplied resolved inputs do not prove that a condition would select this stage.")
            return 0
        if args.resolved_state is not None or args.state is None or args.stage is not None:
            raise ConfigurationError("Flat export requires --state and does not accept stage options.")
        atomic_json(args.out, {"model": program.model,
            "state": project_state(program.state, load_document(args.state)),
            "questions": {name: q.wire() for name, q in program.questions.items()}})
        print(f"Native request saved: {args.out}. Composition and decision policies execute in the Python runtime.")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return dispatch(args)
    except KeyboardInterrupt:
        print("Interrupted; no completed result is claimed.", file=sys.stderr)
        return 130
    except S1Error as exc:
        print(f"s1: {exc}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        locations = [".".join(str(part) for part in error["loc"]) for error in exc.errors()[:3]]
        print(f"s1: invalid typed document at {', '.join(locations)}.", file=sys.stderr)
        return 2
    except (ValueError, TypeError, OSError) as exc:
        # Pydantic/JSON errors may contain sensitive values. Keep default output terse.
        print(f"s1: invalid input or local file operation ({type(exc).__name__}).", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
