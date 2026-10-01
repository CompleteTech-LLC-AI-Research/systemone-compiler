# Jev gateway adapters and traffic reporting

Opt-in native typed Jev evaluation through TypeSafe, two Vercel credentials,
BeatAPI, OpenCode Zen and Classifier.dev. These examples are not imported by the
compiler/runtime package and do not change its default backend. No credentials,
benchmark data or teacher traces are included.

Install `pip install -r examples/ai-gateway/requirements.txt`; the direct route
also needs the project live extra. Copy this directory's `.env.example` to the
repository-root `.env.local` and populate it locally. The adapter reads that file
explicitly; the core CLI does not load dotenv automatically. Authorize paid calls
and data sharing, and verify model equivalence, before enabling live routes.

## Native study adapter

`paced_routes.make_backend(limit, launch)` returns a backend with eight workers
and a shared request budget. The runner supplies `launch.ROOT`, `launch.RUN`,
`launch.progress` (request/response counters), `launch.checkpoint(**fields)`, and
`launch.make_backend(allow_paid, maximum)` for a cache-disabled direct backend.
Use the adapter's ordered `map` method explicitly in the research runner; this
example does not patch global evaluation functions or route teacher traffic.

| Route | Concurrent cap | Local pacing |
|---|---:|---|
| Native TypeSafe | 8 | 20 starts/sec |
| Vercel key 1 / key 2 | 4 each | 10 starts/sec each; shared 20/sec and cooldown |
| BeatAPI | 1 | 60 seconds after completion |
| OpenCode Zen | 2 | 1 start/sec |
| Classifier.dev | 2 | 10 starts/sec; 3,000 questions/rolling minute; 20,000/UTC day |

Eight is the total executor limit. These are conservative local caps, not promises
of provider capacity or independent upstream throughput. Ready routes receive
work fairly. Classifier's weighted question allowance persists under RUN; only
one worker process may own that state. Other account consumers can still exhaust
shared provider quotas. No verified TPM allowance is assumed.

Programs stay pinned to `jev-1.13.0`; actual returned model IDs are preserved.
This example explicitly maps Vercel `typesafe-ai/jev` and BeatAPI/Zen
`jev-1.13-free` under the study owner's equivalence confirmation, not independent
checkpoint attestation. Unexpected returned identities fail. Classifier uses
native `/v1/systemone`, preserving typed questions without Smart escalation or
conversion through `/v1/classify`.

Transient failures and 429 trigger shared group cooldowns honoring numeric/date
Retry-After and native SDK retry-after-ms. At most three attempts per logical
call are charged against the original budget. Auth, credit, identity and budget
failures remain errors, never zero-quality scores. Gateway-reported extra
attempts are charged when exposed; undisclosed failed internal attempts cannot
be reconstructed. No raw prompt logging or response disk cache is enabled.

Pollinations is disabled: the tested key returned 403 for `typesafe/jev-1.13`
and 400 for `jev-1.13-free`. Enabling this unverified route fails closed.

Consent boundary: these routes belong to the existing benchmark worker and take
their paid-call approval from that worker's launcher, not from a flag in this
module. Routes 3 to 5 (BeatAPI, OpenCode Zen, Classifier.dev) send study inputs to
third-party endpoints and run only when their `AI_GATEWAY_ROUTE_N_ENABLED` value is
`true`; routes 1 and 2 have no such switch. Enable a route only with separate,
recorded approval to share those inputs, and never reuse this module for the
hierarchy study, which has its own consent flags.

## Reporting

Aggregate per-phase telemetry records attempts, responses, tokens, retries and
route usage every ten seconds and at close. On Windows, run
`python examples/ai-gateway/report_activation.py --run-dir <study-directory>`.
This one-shot observer uses the existing selected-resume-12 handoff/status
protocol: it waits for activation, reports the first ten minutes, writes JSON
and Markdown, and submits desktop notifications via notify.ps1. It neither
starts inference nor interrupts workers. It does write `new-route-throughput.*`
files into the run directory, and notify.ps1 registers a notification identity under
`HKCU:\Software\Classes\AppUserModelId` on first use. Requests and tokens per
second are the sum of every `rate-telemetry` file in the run directory divided by
the time since activation, so earlier phases or processes that left telemetry
there inflate both figures; treat them as approximate, not as a benchmark.
Adapt the handoff ID for other studies.

The report's historical four-worker references use different windows and
workloads; they are not a controlled causal speedup comparison. Transport checks
are not accuracy evidence. Independent held-out evaluation and human prompt
review remain required before claiming quality improvements.

## Live probes and validation

Node 24+: `npm ci`, then `npm run models` or `npm run smoke` from this directory
after authorizing API use. The smoke sends one synthetic boolean question through
AI SDK experimental_evaluate with retries disabled. This SDK probe is separate
from the native study adapter and does not validate all Choice/Score fields.

`python examples/ai-gateway/check_native.py --allow-paid` sends four bounded
synthetic probes to the two Vercel keys. Output is sanitized and saved under
ignored runs/. Importing the module makes no calls.

All six enabled routes passed local live synthetic runtime checks in the
user-authorized study. No sustained load or independent quality gain was measured.
Offline tests are in tests/test_gateway_routes.py; CI makes no paid calls.
Operational copies remain in ignored runs/ so publishing cannot alter a loaded
worker. Telemetry and reports contain only aggregate accounting.

## Primary sources

- [Vercel rate limits](https://vercel.com/docs/ai-gateway/rate-limits)
- [BeatAPI Decisions](https://docs.beatapi.io/decisions)
- [OpenCode Zen](https://opencode.ai/docs/zen/)
- [Classifier native compatibility and quotas](https://classifier.dev/)
- [Pollinations model catalog](https://gen.pollinations.ai/text/models)

### Calibration recovery

`make_calibration_backend(limit, launch)` uses only native TypeSafe while retaining
the eight-worker limit and the original attempt ceiling. A calibration allowance
equal to its row count has no spare attempts for retries. If exhausted,
`calibration_budget_exhausted` identifies only that phase and exact exhaustion;
the runner must retain the selected program, mark the arm incomplete without a
score, and continue other arms. It must not silently refill budgets or count a
partial calibration as complete.
