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

## Update, October 1, 2026: DSPy 3.4.0 compatibility (issue #90)

Offline only; no keys, no provider calls, no teacher uploads. Windows 11, Python 3.14.3.
The only socket used is a loopback test double in `tests/test_dspy_compat.py`; its output
is a test double, not a Jev or provider result, and says nothing about quality.

Resolved versions:

| Environment | dspy | gepa | litellm | pydantic | typesafe-sdk |
|---|---|---|---|---|---|
| Worktree `.venv` | 3.3.1 | 0.1.4 | 1.103.2 | 2.13.5 | 0.7.0 |
| Scratch `dspy[litellm]==3.4.0` | 3.4.0 | 0.1.4 | 1.103.2 | 2.13.5 | 0.7.0 |

Executed (counts include the optional real-package tests; **0 skipped** in both):

- DSPy 3.3.1: `python -m pytest -q` -> **582 passed, 0 failed, 0 skipped**.
- DSPy 3.4.0: `python -m pytest -q` -> **582 passed, 0 failed, 0 skipped**
  (3 warnings: the DSPy `BaseLM.forward()` deprecation, a pydantic ReadOnly notice).
  `tests/test_research_teacher.py` and `tests/test_hierarchy_architect.py`: 35 passed.
- `typewright doctor --check-optional` passed (no network) under both versions.
- `ruff check src tests examples`, `python -m build`, `tests/check_wheel_package.py`
  (49 files, four entry points) and the no-key demo (`optimizer=none`, zero network
  calls) ran on the 3.3.1 worktree environment.

Findings:

1. Under `engine="auto"` the teacher's metered `forward` takes DSPy's native lm15 path
   (LiteLLM is not called) and its response has no `_hidden_params`, so SDK cost
   estimates became unknown. Under `engine="litellm"` the 3.3.1 behavior is retained.
   `DSPyTeacher` now pins `engine="litellm"` when `dspy.LM` accepts it. The new test
   fails with `engine="auto"` (verified) and passes with the pin.
2. Unchanged: cache flags, `disable_history`, both call ceilings, and the
   `S1_TEACHER_API_BASE`/`S1_TEACHER_API_KEY` overrides (the request reached the loopback
   double with the bearer key). Consent flags were not touched.
3. DSPy 3.4 deprecates overriding `BaseLM.forward()` and schedules removal in 3.5; the
   teacher meter relies on that. Hence the `<3.5` bound. Follow-up: #93.
4. `dspy` 3.4.0 requires `pydantic>=2.11.0` itself; no extra pin added, base unchanged.

Decision: supported range `dspy[litellm]>=3.3.1,<3.5`, `gepa==0.1.4`. CI's
`optional-contracts` job now has a 3.3.1/3.4.0 matrix; shipped CI config has not run remotely.

Not executed: any live TypeSafe or teacher request, real GEPA optimization, DSPy 3.4.x
patch releases above 3.4.0, `dspy[typesafe]`/`ReAnchor`/native decision types, Python
3.11-3.13 and non-Windows runs, remote CI, `build`/wheel/demo under 3.4.0, and a re-read
of primary documentation (the inspection date is unchanged).

## October 3, 2026 organization delivery checkpoint

These Linux Python 3.14.7 receipts extend the historical records above; they do
not replace Windows, live-provider, or independent quality qualification. The
exact optional profiles used DSPy 3.3.1 or 3.4.0, GEPA 0.1.4, TypeSafe SDK 0.7.0,
LiteLLM 1.103.2 and Pydantic 2.13.5. No paid inference or private sharing was
authorized.

- [PR #99](https://github.com/Jev-Engineering/TypeWright/pull/99), triage
  regressions: head `a7ab0042957f67fc857981ba9ae62775a9604355`, merge
  `5165fb6d8d05307e7b1467f5fd2bbb34cdf8e5d4`; isolated full suite **618 passed**,
  zero skipped; Ruff and independent review passed.
- [PR #100](https://github.com/Jev-Engineering/TypeWright/pull/100), owner-approved
  hierarchy tracking split: merge `0563956cbac62bd899d5fddad734f84a4ae608e2`.
  Software acceptance for #20/#21 is complete; their original live criteria remain
  explicitly open in #74 and #6. Independent scope/state review passed.
- [PR #101](https://github.com/Jev-Engineering/TypeWright/pull/101), opt-in native
  calibration fitting: head `c316283c8fc612990f74246958edaab9aaa4ec08`, merge
  `aa448275822e69a6356ee32ac36ba3ffa73ab950`; isolated full suite **640 passed**,
  zero skipped. Calibration isolation, unchanged results and independently
  committed legacy artifact/checksum compatibility were verified.
- [PR #102](https://github.com/Jev-Engineering/TypeWright/pull/102), teacher raw
  completion metering: head `f9080cba8416a88642bf8ed14d91bdbd505c523d`, merge
  `67a53ee047dd9a14a70abdf13641974087d42457`; isolated full suites **628 passed**,
  zero skipped, on each DSPy profile. An early inherited legacy-completion path
  accidentally attempted an uncredentialed synthetic provider request and received
  HTTP 401. That failure is retained; composition and loopback/socket guards
  replaced the bypass before the successful final suites. No paid inference or
  private examples were sent.
- [PR #103](https://github.com/Jev-Engineering/TypeWright/pull/103), compile-only
  native adapter: head `de7c82dd3fe5ba3b81ceceb49cab07db206c4bae`, merge
  `821b29d627b602b0b042b474fd5a2248febd5c74`; isolated full suite **685 passed** on
  DSPy 3.4.0; **667 passed, 18 skipped** on 3.3.1 because native qualification
  requires exactly 3.4.0. The historical dependency assertion initially failed
  after `all` selected the exact native profile; the revised test preserves the
  separate teacher range and both full reruns passed. Independent source, wiring
  and dependency review passed. Ruff, build, 51 wheel source members, all four
  entrypoints, fresh no-key demos and the synthetic study journey passed. Runtime
  and adapter-module imports loaded no DSPy, GEPA or SDK.

Other retained attempts: #99 initially failed two subprocess imports without the
checkout's explicit `PYTHONPATH`; a mutable-worktree attempt during edits also
failed. The stable isolated rerun passed. #101 had an interrupted mutable-root
run and a nested replay identity failure while a concurrent build created
package metadata. A stable post-build causal check and full suite passed. Early
installed-wheel study attempts combined incompatible fixtures and were correctly
rejected; the final complete fixture passed. These attempts are separate from
the successful final acceptance receipts above.

Each PR above completed **10 successful exact-head hosted checks** and
**five successful checks observed on its actual merge commit**. For #103 the
[merge workflow](https://github.com/Jev-Engineering/TypeWright/actions/runs/37127560430)
ran on `821b29d627b602b0b042b474fd5a2248febd5c74`; this is executed remote
evidence, unlike the earlier shipped-configuration checkpoint. Local main was
fast-forwarded to that merge without changing original user files or deleting
retained packet/worktree evidence.

NOT_RUN: live native provider qualification, representative independent quality
studies, hierarchy paid selection/teacher sharing/held-out dispatch, production
activation, administrative CI/protection changes and a fully resolved dependency
lock. The gateway destination patch in PR #98 still awaits the owner's endpoint
policy decision. Synthetic checks do not establish gains or production approval.
