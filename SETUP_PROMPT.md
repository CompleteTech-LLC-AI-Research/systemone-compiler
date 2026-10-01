# Coding-agent setup prompt — System One Compiler

You are setting up the existing `typewright` project contained in this
folder. It is an implemented Python framework, not a request to generate a new
architecture. Preserve the working design; fix specific verified defects rather
than rewriting it wholesale.

## Objective

Make this project installable and runnable on my machine, verify its core and
optional integrations, and leave clear setup instructions and a truthful
`SETUP_REPORT.md`. Start with the no-key synthetic demo. Paid inference or upload
of my data requires a separate explicit authorization; this setup prompt alone
does not grant it.

## Read before changing anything

Read `AGENTS.md`, `README.md`, `pyproject.toml`, `docs/ARCHITECTURE.md`,
`docs/INTEGRATIONS.md`, `docs/SECURITY.md`, and `docs/BUILD_REPORT.md`. Inspect the
actual source and tests relevant to any problem you encounter. Do not mistake
mock tests for live provider verification. No claim of real prompt improvement
has been established by the bundled synthetic fixture.

## Complete the following workflow

1. Identify the OS, working directory, Python version, and existing environment.
   Use Python 3.11+ and a project-local `.venv`; do not use sudo/global installs or
   alter unrelated projects. On Windows prefer the venv's `Scripts/python.exe` so
   activation policy is not an obstacle. Preserve existing files and run outputs.
   Verify `MANIFEST.sha256` when present before edits, recording intentional
   modifications afterward rather than concealing a checksum discrepancy.

2. Install the project with development dependencies, then run the local workflow:

   ```text
   python -m pip install -e ".[dev]"
   python -m typewright doctor
   python -m pytest -q --cov=typewright --cov-report=term-missing
   python -m typewright demo --out runs/setup-demo-UNIQUE
   ```

   Replace `python` with the actual venv interpreter. Choose a new run directory;
   never delete an existing run merely to make the command succeed. Inspect
   `program.s1.json`, `report.json`, `report.md`, and `sample_prediction.json`.
   Confirm that output explicitly says synthetic and optimizer none.

3. Install optional integrations with `python -m pip install -e ".[all,dev]"`.
   Run `python -m typewright doctor --check-optional`, the full test suite, and
   `python -m ruff check src tests examples`. The optional tests include real SDK
   shape validation, the real GEPA engine with a mock task/fake proposer, and DSPy
   signature construction without inference. These tests must stop skipping only
   when the packages truly import and run. Do not replace them with fake packages
   and call that compatibility verification.

4. Verify current official documentation if a pinned dependency cannot resolve
   or its interface differs. Integration targets in this package are
   `typesafe-sdk==0.7.0`, `dspy[litellm]==3.3.1`, and `gepa==0.1.4`. Check
   `docs/SOURCES.md` rather than guessing APIs. Fix the smallest necessary adapter
   code, add a regression test, document any pin change, and rerun the checks.
   Preserve native TypeSafe `system_one(state, questions, model)` execution and
   standalone GEPA's adapter/custom-proposer design. Do not fake Jev as a chat LM.
   Generate a resolved lock only after resolution actually succeeds, and record
   the Python/platform/provider compatibility scope of that lock.

5. Run `python -m build`. Install the wheel in a fresh temporary environment or
   target directory, then verify `typewright init`, `typewright demo`, and the packaged starter
   data work outside the source tree. Ensure `import typewright` does not import
   DSPy, GEPA, or the TypeSafe SDK. Keep production/runtime dependencies separate
   from optimizer dependencies.

6. Validate useful workflows without provider calls: create a new use-case
   directory; create a template draft; export JSON schemas; inspect a checksummed
   artifact; export native playground JSON; and generate review-required
   hardening proposals. Do not convert synthetic/hardening labels into purported
   ground truth. Review the support-triage dataset card.

7. Verify the separate hierarchy path without provider calls: run
   `python -m typewright demo --starter hierarchy --out runs/setup-graph-UNIQUE`,
   inspect its frozen artifact and result statuses, and run `typewright-study` through
   register/select/test/report on the checked-in synthetic support fixture using
   fresh output paths. Confirm the study report says synthetic and that the
   installed wheel includes `typewright-study` and hierarchy starter data. See
   `docs/HIERARCHY_MIGRATION.md` and `docs/HIERARCHY_STUDY.md`; neither workflow
   authorizes a live study. A live study additionally needs `typewright-study manifest`
   and a human-completed `typewright-study review` before paid selection.

## Credentials, privacy, and live calls

Do not print, commit, inspect into reports, or ask me to paste API keys. Check only
whether the required variables are present. `.env.example` is a reference and is
not automatically loaded. If credentials are absent, report the exact variable
names and stop the live portion; still complete all available local checks.

A live TypeSafe smoke test needs explicit permission, `TYPESAFE_API_KEY`, and a
versioned target model (currently configured as `jev-1.13.0`). If authorized, use
the README's one-request draft smoke check with `--backend typesafe --allow-paid
--allow-unvalidated --max-calls 1 --no-cache`. Record success/failure, observed
model, and usage without exposing credentials. Do not silently substitute mock
results when live calls fail.

Live DSPy/GEPA optimization additionally needs an explicitly selected real teacher
model and provider credentials, permission to send the projected training data to
that provider, four disjoint splits, and agreed request/signature/token budgets.
Require both `--allow-paid` and `--share-feedback`. Check that data are appropriate
for both services. Do not run paid search until these are supplied and authorized.
The README gives an example command, not spending authorization.

Keep train, validation, calibration, and test roles separate. No test labels,
examples, or failure analysis may be supplied to the teacher/optimizer. Review
policy thresholds fit on calibration only. A test result becomes descriptive
once examined; do not keep optimizing against that same test set. Do not claim
posterior calibration or guaranteed selective risk from empirical thresholds.

Treat budgets honestly: TypeSafe limits SDK attempts with retries disabled;
teacher limits count DSPy signature invocations, not a hard dollar ceiling.
Do not increase budgets or substitute paid providers without approval. Disable
unnecessary verbose request logs and external experiment trackers. No arbitrary
LLM-generated code, pickle loading, shell commands, or autonomous application
actions belong in the compiled JSON artifact.

## Deliverables

Create `SETUP_REPORT.md` with the OS/Python and resolved package versions; actual
commands executed; test pass/fail/skip counts; optional integration results;
wheel/init/demo verification; paths to generated outputs; a list of changes and
why; and precise remaining blockers. Distinguish these statuses explicitly:

- Core software and synthetic demo tested.
- Real installed-package interfaces tested without inference.
- Actual TypeSafe inference tested, only if executed.
- Actual DSPy teacher plus GEPA optimization tested, only if executed.
- Real benchmark gains demonstrated, only if independently measured.

Do not describe a skipped or unexecuted step as passed. End with the exact local
commands I can run next, using the detected OS and environment. Complete the
available setup work now; do not promise future background work or conceal a
blocked step.
