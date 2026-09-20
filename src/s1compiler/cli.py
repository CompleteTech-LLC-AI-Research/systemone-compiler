from __future__ import annotations
import argparse
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

from .architect import DSPyTeacher, template_program
from .backends import AnswerCache, ManagedBackend, MockBackend, TypeSafeBackend
from .compiler import CompileOptions, Compiler, versions
from .data import read_jsonl
from .errors import ConfigurationError, S1Error
from .hardening import propose_cases
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


def init_project(directory: Path):
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ConfigurationError("Target directory must be missing or empty; existing files will not be overwritten.")
    source = Path(__file__).parent / "templates" / "support"
    shutil.copytree(source, directory, dirs_exist_ok=True)


def ensure_new_output(directory: Path):
    if directory.exists() and any(directory.iterdir()):
        raise ConfigurationError("Output directory is not empty; choose a new run directory.")
    directory.mkdir(parents=True, exist_ok=True)


def build_parser():
    parser = argparse.ArgumentParser(prog="s1", description="Declare, optimize, and run typed Jev use cases.")
    parser.add_argument("--version", action="version", version="systemone-compiler 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create an editable support-triage project and synthetic dataset.")
    init.add_argument("directory", type=Path)
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
    ev.add_argument("--data", type=Path, required=True)
    ev.add_argument("--out", type=Path, required=True)
    backend_args(ev)
    doctor = sub.add_parser("doctor", help="Local dependency and contract checks; never calls a model.")
    doctor.add_argument("--check-optional", action="store_true")
    demo = sub.add_parser("demo", help="No-key, no-network end-to-end synthetic smoke test.")
    demo.add_argument("--out", type=Path, default=Path("runs/demo"))
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
    export.add_argument("--state", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
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
        init_project(args.directory)
        print(f"Created {args.directory}. Edit usecase.yaml; bundled labels are synthetic examples.")
    elif args.command == "doctor":
        return doctor(args.check_optional)
    elif args.command == "draft":
        if args.out.exists():
            raise ConfigurationError("Draft output already exists; choose a new filename.")
        source = UseCase.load(args.spec)
        program = template_program(source)
        if args.architect == "dspy":
            program = make_teacher(args).propose_plan(source, program, [])
        program.save(args.out)
        print(f"Draft saved: {args.out}. No accuracy claims; evaluate before deployment.")
    elif args.command == "compile":
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
        init_project(project)
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
        program = Program.load(args.artifact)
        backend = make_backend(args)
        try:
            if args.command == "run":
                runtime = Runtime(program, backend, enforce_release=not args.allow_unvalidated)
                result = runtime.run(load_document(args.state))
                if args.out:
                    atomic_json(args.out, result)
                print(json.dumps(result, indent=2))
            else:
                rows = read_jsonl(args.data, source_for(program))
                report, _ = evaluate(program, rows, backend)
                report["accounting"] = backend.accounting()
                atomic_json(args.out, report)
                print(f"Evaluation saved: {args.out}")
        finally:
            backend.close()
    elif args.command == "inspect":
        program = Program.load(args.artifact)
        print(program.model_dump_json(indent=2))
    elif args.command == "harden":
        source = UseCase.load(args.spec)
        atomic_json(args.out, propose_cases(source, read_jsonl(args.data, source)))
        print(f"Review-required proposals saved: {args.out}")
    elif args.command == "schema":
        args.out.mkdir(parents=True, exist_ok=True)
        atomic_json(args.out / "usecase.schema.json", UseCase.model_json_schema())
        atomic_json(args.out / "program.schema.json", Program.model_json_schema())
        print(f"Schemas saved: {args.out}")
    elif args.command == "export-playground":
        program = Program.load(args.artifact)
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
    except (ValueError, TypeError, OSError) as exc:
        # Pydantic/JSON errors may contain sensitive values. Keep default output terse.
        print(f"s1: invalid input or local file operation ({type(exc).__name__}).", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
