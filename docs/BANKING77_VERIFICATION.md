# BANKING77 implementation verification — 2026-09-19

Note: the project was renamed from System One Compiler (`s1compiler`) to Typewright (`typewright`) after this record was written; commands and paths below use the names in effect at the time.

Implemented the [experiment protocol](BANKING77_EXPERIMENT.md) in
`src/typewright/banking77.py`, with regression tests in `tests/test_banking77.py`.
The existing compiler, runtime, adapters, and dependency pins were not changed.
README now links the protocol. This directory is not a Git checkout; no commit,
push, PR, or remote CI execution was performed.

## Executed

- Before edits: all 71 entries in `MANIFEST.sha256` matched their files;
  `python -m pytest -q` passed 121 tests.
- Downloaded both official BANKING77 files at the pinned revision and verified
  SHA256, row counts, labels, and exact projected-input disjointness. Prepared
  four splits under `runs/banking77-20260919-data` without removing records.
- Final `.venv/Scripts/python.exe -m pytest -q`: **140 passed**, no skips.
  Includes real installed SDK/DSPy interface tests and real GEPA engine tests
  with mock backends/proposers, without provider inference.
- `.venv/Scripts/python.exe -m ruff check src tests examples`: **passed**.
- Full-data offline CLI selection with seed 7, all four arms, then frozen test:
  **passed**. Final artifacts are in `runs/banking77-20260919-verified`; results
  are in `runs/banking77-20260919-verified-results/report.json` and `report.md`.
  There were 14,166 mock selection evaluations, 8,144 mock calibration
  evaluations, and 12,320 mock test evaluations (3,080 per arm). The identity
  proposer made 1 / 30 / 4 calls in B / C / D. These are synthetic plumbing
  results, not Jev results or DSPy improvements. All three paired comparisons
  and bootstrap outputs were produced.
- Final frozen manifest checksum:
  `ef91e120f4ef38aeaf6a3cbc8e061c5ffdc466cbafbfc0b1756d46fc2c76424d`.
- `.venv/Scripts/python.exe -m build --outdir dist/banking77-verified`: wheel
  and sdist built successfully. Installed the wheel into a fresh target at
  `runs/banking77-wheel-verified`. From outside the source tree, verified
  package provenance, lazy imports, `s1 init`, no-key `s1 demo`, and benchmark
  CLI help. Confirmed the new module and starter data are in the wheel and
  downloaded BANKING77 CSVs are not bundled.
- The installed native SDK accepted the full 77-label Choice schema offline.
  Prepared a baseline smoke artifact and a train-only state at
  `runs/banking77-20260919-smoke.s1.json` and
  `runs/banking77-20260919-smoke-state.json`.
- Prepared a reviewable full-study budget in
  `runs/banking77-20260919-live-plan.json`: 111,160 Jev requests and 183 teacher
  signatures, plus one separately budgeted native smoke request. These are
  request/signature ceilings, not dollar or aggregate token caps.

## Findings addressed

An initial synthetic run exposed GEPA's default prompt-text logger. The final
benchmark disables it with a quiet logger, and a regression test covers this.
Independent rewrites now project input fields before sharing demonstrations;
another regression test ensures undeclared fields cannot reach the teacher.
All earlier scratch runs were preserved. The final run uses the checked source
hashes, rather than those older artifacts.

## Not executed / pending

No paid TypeSafe smoke test, live DSPy teacher call, live optimization, or real
Jev benchmark evaluation was performed. No evidence of improved Jev quality is
claimed. The process environment did not provide `TYPESAFE_API_KEY` or
`S1_TEACHER_MODEL`; `.env` was not read or automatically loaded.

The next live step requires the selected teacher/provider, an agreed dollar
limit enforced by the provider, and explicit consent to share public training
examples and Jev training traces with that teacher. `AGENTS.md` requires paid-call
consent and separate teacher-sharing consent. The full study remains subject to
baseline review, a successful 77-label smoke test, and semantic review of all
selected prompts before final testing.

## Integrity record

The current `MANIFEST.sha256` was updated only for README and the four added
files (benchmark module, tests, protocol, and this report). Historical
`MANIFEST.sha256.as-shipped` was preserved. Downloaded data, scratch outputs,
and credentials are not included in the source manifest. Checksums are not
signatures and do not establish authenticity.
