# Benchmark recovery

The reusable recovery helper and its regression tests are included in this
repository. The v31 launcher, reconciliation receipts, and study-specific tests
described below belong to the local, ignored `runs/live-study-20260920/` directory;
they are not shipped or activated by installing the library.

The study's native request layer already retries transient HTTP and read-timeout
failures twice. Previously, exhaustion of those attempts discarded the active
batch and unwound the optimizer and study worker.

The v31 repair uses `s1compiler.resilience.recover_batch` at the study batch
boundary. It drains in-flight work, retains successful results in their original
order, and retries only failed rows after shared cooldowns of 60, 120 and 240
seconds. No new rows start during recovery. Heartbeats continue while waiting.
Every replacement row reserves from the existing shared retry pool; the native
layer's retries also remain charged. No request, teacher or global ceiling is
increased. A fatal error among in-flight requests takes precedence over a
transient failure. Exhausted transient arms are recorded as incomplete, never
given a fabricated score, and independent eligible arms may continue.

This keeps optimizer state alive through bounded outages. It is **not** a
durable GEPA process checkpoint: a process crash can still lose C-arm optimizer
state. Existing D-arm JSON checkpoints and calibration row journals are retained.
GEPA's installed pickle persistence is deliberately not enabled. A future GEPA
checkpoint implementation requires a validated JSON schema and deterministic
resume tests before use in this study.

The v31 phase amendment repairs the previously blocked ANLI calibration with
2,225 remaining rows and permits one bounded replacement search for each failed
SST-5 C-17/C-29 arm (100,000 selection requests, 30 teacher signatures, 60 teacher
provider attempts, and 1,924 calibration rows each). All past charges, uncertain
reservations and the held-out reserve remain included. The planner refuses the
handoff unless these allocations fit the original cumulative ceilings. The
shared retry pool is carried forward without renewal. Replays and changed retry
latency must be disclosed in the study report.

The transition waits for a cooperative arm boundary under the supervisor lock;
the old implementation and receipts are preserved. The trusted supervisor runs
the offline audit before launching the replacement with explicit `--run`.
Held-out evaluation remains gated on freezing and human semantic review.

Validation: `python -m pytest -q`, plus the study's
`test_batch_recovery_v31.py`. These use synthetic injected failures; they do not
establish provider availability or benchmark accuracy.
