# Build and verification report

Release: **System One Compiler 0.1.0** (renamed Typewright after this record; see
the update at the end). Verification date: September 19, 2026
(America/New_York; the packaging host's UTC date is September 20).
The sections below describe that September 19 run and are kept as a record;
their counts and commands use the names in effect at the time.

## Executed successfully

- Python **3.13.5**, Linux. Pydantic 2.13.4,
  PyYAML 6.0.3, pytest 9.0.2.
- **111 tests passed, 3 optional tests skipped, 0 failed.**
- Statement coverage: **87.67%** (1116/1273 statements),
  rounded to 88% by the coverage reporter. Coverage is a software-test measure,
  not a model-quality or integration-compatibility score.
- Editable installation and pure-Python wheel build succeeded using installed
  build tools with `--no-deps --no-build-isolation`. This did not resolve the
  optional dependency tree. The wheel is included in `wheels/`; its dependencies
  still need to be installed.
- Installed the wheel in an isolated target directory and asserted imports came
  from that directory, not the source checkout. Ran `init`, `demo`, `draft`,
  `schema`, `export-playground`, and `harden` successfully against that wheel.
- Verified importing the runtime package does not import DSPy, GEPA, or the
  TypeSafe SDK.
- Python compileall and Bash script syntax checks succeeded.

The no-key demo completed with synthetic lexical answers and `optimizer=none`.
It created actual artifacts and reports without network calls. This is not a
real Jev benchmark. See `examples/compiled_demo/` for the generated snapshot.

## Not executed / not established

The real `typesafe_sdk`, `dspy`, and `gepa` packages were unavailable in this
packaging environment. Their three optional tests skipped, and
`doctor --check-optional` correctly returned an unverified status (exit code 2).
A package-resolution attempt did not find the requested GEPA distribution here.
This does not establish that the published package is unavailable elsewhere.

Core tests include explicitly labeled SDK/engine/teacher test doubles. They do
not replace real-package integration tests. No TypeSafe API key or teacher model
was configured. **No live TypeSafe request, teacher request, real GEPA optimization,
or real prompt-quality gain was verified.** The setup prompt directs your coding
agent to close those gaps in your environment with permission and real credentials.

Ruff was unavailable, so a full Ruff run was not performed. Unused imports were
reviewed and Python parsed/compiled successfully; these are not equivalent to
Ruff. Windows/PowerShell, macOS, Python 3.11/3.12, and remote GitHub CI execution
were not exercised. Their setup/CI files are supplied for local or subsequent CI
verification. No fully resolved transitive dependency lock is claimed.

## Reproduce

```text
python -m pip install -e ".[dev]"
python -m pytest -q --cov=typewright --cov-report=term-missing
python -m typewright demo --out runs/reproduce-demo
python -m pip install -e ".[all,dev]"
python -m typewright doctor --check-optional
python -m pytest -q
```

These commands use the current Typewright names; the original run used
`s1compiler`.

Install dependencies before expecting optional tests to run. These commands alone
do not authorize paid inference. Machine-readable status is in
`reports/offline_verification.json`.

## Fresh ZIP extraction

Extracted the archive into a new directory, verified every SHA256 manifest entry,
confirmed imports resolved to the extracted source, and reran the full suite:
**111 passed, 3 optional tests skipped, 0 failed**. The extracted no-key demo also
completed successfully with `optimizer=none` and zero network calls. Only this
verification report and the manifest/archive were updated after that source check.

## Update, October 1, 2026: Typewright rename

Separate from the September 19 record above. Run in the rename worktree on
Windows 11 with Python 3.14.3 and the real `typesafe-sdk` 0.7.0, `dspy` 3.3.1, and
`gepa` 0.1.4 installed (no keys, no network inference):

- **579 tests passed, 0 failed**, including the optional real-package tests.
- `ruff check src tests examples` passed.
- `python -m build` produced the `typewright-0.1.0` wheel and sdist;
  `tests/check_wheel_package.py` verified 49 library/starter files and all four
  console entry points (`typewright`, `typewright-study`, `s1`, `s1-study`).
- `typewright doctor --check-optional` reported the three local contract checks
  passing with zero network calls. This does not verify live behavior.
- The no-key demo ran with `optimizer=none` and zero network calls.

Still not executed: any live TypeSafe or teacher request, real GEPA optimization,
remote CI, macOS/Linux/Python 3.11-3.13 runs of the renamed package, and a
clean-environment install of the new wheel. `MANIFEST.sha256` predates the rename
and was not regenerated.
