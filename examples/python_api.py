"""Run from the repository root after installation. No credentials or network required."""
from pathlib import Path
from s1compiler import UseCase, Compiler, CompileOptions, Runtime
from s1compiler.backends import AnswerCache, ManagedBackend, MockBackend
from s1compiler.data import read_jsonl
from s1compiler.reporting import save_compilation

root = Path(__file__).parent / "support_triage"
source = UseCase.load(root / "usecase.yaml")
splits = {name: read_jsonl(root / f"{name}.jsonl", source)
          for name in ("train", "validation", "calibration", "test")}
backend = ManagedBackend(MockBackend(), cache=AnswerCache())
try:
    program, report = Compiler(backend, options=CompileOptions(min_calibration_samples=3)).compile(source, **splits)
    save_compilation("runs/python_api", program, report)
    result = Runtime(program, backend).run({
        "message": "Please refund a duplicate charge.", "customer_plan": "team"})
    print(result["decisions"])
    print("SYNTHETIC SOFTWARE EXAMPLE: not a Jev benchmark.")
finally:
    backend.close()
