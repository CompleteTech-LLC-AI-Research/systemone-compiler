# Integration targets and local verification

Documentation inspection date: **2026-09-19**. These are the implementation targets:

| Component | Target | Role |
|---|---|---|
| Python | >=3.11 | Core library, CLI, tests. |
| TypeSafe SDK | `typesafe-sdk==0.7.0` | Native `TypeSafeClient.system_one`. |
| Target model | `jev-1.13.0` | Versioned ID, not a moving alias. |
| DSPy teacher | `dspy[litellm]>=3.3.1,<3.5` (verified: 3.3.1 and 3.4.0) | `Signature`, `Predict`, explicit teacher `LM/context`. |
| Native compile client | `dspy[typesafe]==3.4.0`, SDK 0.7.0 (`compile` extra) | Experimental TypeSafe wrapper; absent from frozen runtime. |
| GEPA | `gepa==0.1.4` | Standalone adapter and custom component proposer. |

These are not claims that all dependencies were installed or live-tested in the
packaging environment. See `BUILD_REPORT.md` for the actual boundary. PyPI resolution
may fail on a different Python/platform/provider combination. Verify against
primary documentation and change pins with regression tests rather than silently
installing an arbitrary newest version.

## DSPy 3.3.1 and 3.4.x (issue #90)

The supported range is `>=3.3.1,<3.5`, with `gepa==0.1.4` pinned (DSPy 3.4.0 resolves
the same GEPA). Only 3.3.1 and 3.4.0 were exercised; later 3.4.x patch releases are
accepted by the range but unverified. The upper bound is evidence-based: DSPy 3.4
deprecates overriding `BaseLM.forward()` and schedules its removal in 3.5.
The #93 replacement uses the supported engine on 3.4; only the supported 3.3
legacy branch still meters forward. See `BUILD_REPORT.md` for the executed results.

- Engine: DSPy 3.4 `engine="auto"` prefers native lm15 execution. The native response
  has no LiteLLM `_hidden_params`, so the teacher's SDK cost estimate would silently
  become "unknown". the #90 checkpoint therefore selected `engine="litellm"`. The #93 engine now
  observes the raw LiteLLM response before conversion (3.3 retains its legacy path).
- Unchanged in 3.4.0 (checked offline against a loopback test double): caches off,
  `disable_history=True` leaves no history, signature-call and LM-forward ceilings count,
  `S1_TEACHER_API_BASE`/`S1_TEACHER_API_KEY` reach the request, consent flags are untouched.
- Pydantic: `dspy` 3.4.0 itself requires `pydantic>=2.11.0`, so the `optimize` extra
  resolves to it automatically. The base install stays `pydantic>=2.10,<3`.
- The #90 checkpoint did not cover the native DSPy client. The #91 qualification
  below covers raw native transport, not DSPy Predict decision-type translation
  or ReAnchor. Those remain unused.
- The documentation inspection date above was not changed; this verification used the
  installed packages, not a fresh read of the primary documentation.

## TypeSafe contract

The adapter constructs `TypeSafeClient(retry=RetryPolicy(max_retries=0, timeout=...))`
and calls `system_one(state=..., questions=..., model=...)`. Questions are raw
native dictionaries with type/instructions/criteria. Results use the documented
Pydantic `model_dump(mode="json")` serialization, including answers/model/usage.
Choice has probabilities, selected choice, and confidence; Score has score,
probabilities, legend, and confidence; Noul has `noul` P(true) and no native
confidence. Response model must equal the pinned requested version.

`doctor --check-optional` checks imports, native question constructors, and the
method signature without inference. The optional test exercises native question
serialization. Test doubles separately exercise adapter invocation and response
validation. Only an authorized real request verifies authentication and actual
server behavior.

## DSPy contract

DSPy creates three structured teacher programs (`Design`, `HierarchyDesign`, and `Revise`) using
`dspy.Signature`, `InputField`, `OutputField`, and `Predict`. Inference occurs in
`dspy.context(lm=..., disable_history=True)`. Cache use and automatic retry arguments are explicitly
configured. Signature calls and provider-request attempts have separate ceilings.
DSPy 3.4 uses a custom `complete(Request) -> Response` engine that admits each
attempt before dispatch and observes raw LiteLLM model, usage and SDK cost metadata
before canonical conversion. The outer engine deliberately exposes no
`complete_legacy` hook: that transition hook bypasses `complete` during ordinary
`Predict` calls. DSPy 3.3 retains its supported legacy forward branch; DSPy 3.4
does not override `forward`. The supported range remains `<3.5` pending verification.
Endpoint, key and timeout overrides belong to the custom engine. Both branches
disable SDK retries and caches. Typed budget and response-identity failures remain
failures across DSPy's error boundary, including adapter fallback.
Model-specific restrictions on temperature, tokens, supported output
formats, or provider model IDs may still need local adjustment.

The local optional test constructs the signatures without inference. It does not
prove that a particular remote model can emit the expected JSON. Actual teacher
responses go through strict parsing and schema validation. Invalid generated JSON
is rejected; do not add an unsafe eval fallback.

## GEPA contract

The standalone engine calls adapter `evaluate`, `make_reflective_dataset`, and
`propose_new_texts`; a custom DSPy proposer handles the last method. Evaluation
returns `gepa.core.adapter.EvaluationBatch` with outputs, scores, and trajectories.
The engine entry point is `gepa.optimize(seed_candidate=..., trainset=...,
valset=..., adapter=..., max_metric_calls=..., ...)`. The result's best candidate
is revalidated before use.

The real-package optional GEPA test runs that engine with a fake deterministic
proposer and mock backend, without paid inference. It verifies engine/adapter
compatibility only; it is not a prompt-quality benchmark. The combined real DSPy
teacher + TypeSafe + GEPA workflow still needs authorized end-to-end verification.
An additional offline graph adapter test runs installed GEPA over complete
synthetic hierarchy executions with no provider calls.

## Environment

- `TYPESAFE_API_KEY`: native provider credential.
- `S1_TEACHER_MODEL`: explicit DSPy provider/model identifier.
- Provider-native environment credential: depends on chosen teacher provider.
- `S1_TEACHER_API_BASE`, `S1_TEACHER_API_KEY`: optional teacher endpoint/key overrides.
- `TYPESAFE_BASE_URL`: SDK-native endpoint override; validate the destination before use.

The `S1_` prefix is kept after the rename to Typewright so existing environments
keep working. No dotenv loading is built in. No actual teacher model is chosen for
the user, and no API secrets are packaged. The compiled runtime depends on the base
library plus `live`; DSPy and GEPA are optional compile-time extras.

## Gateway example adapters

`examples/ai-gateway/` holds opt-in adapters that route native typed Jev
evaluation through additional gateways (two Vercel credentials, BeatAPI, OpenCode
Zen, Classifier.dev) with local concurrency caps and pacing, plus traffic
reporting. They are examples, not part of the installed package: the compiler and
runtime never import them and the default backend stays the mock. They read a
local `.env.local` explicitly. They refuse to build a backend unless the caller
passes `allow_paid=True`, and every route runs only when its
`AI_GATEWAY_ROUTE_N_ENABLED` value is `true`; enabling a route sends study inputs
to a third party and needs its own recorded approval. Provider model IDs map to
the pinned `jev-1.13.0` only under the operator's own equivalence confirmation,
not an independent attestation.
Unexpected returned model identities fail. None of these routes has been verified
live in this repository. See `examples/ai-gateway/README.md` for caps and setup.

## Compile-only TypeSafe qualification (#91)

Owner decision on 2026-10-03: use the DSPy client for native compile-time
evaluation and retain the direct SDK for frozen runtime. Install
`python -m pip install -e '.[compile]'`; combine with `optimize` when a generative
teacher/GEPA is required. The teacher's 3.3.1 floor remains supported independently;
the experimental native wrapper refuses all versions except DSPy 3.4.0 and
typesafe-sdk 0.7.0. This deliberately fails closed on unverified patch releases.

Gap analysis against actual pinned upstream code, rechecked 2026-10-03:

| Upstream behavior | Qualified wrapper |
| --- | --- |
| Defaults to `jev-latest`; no returned-model equality check | Versioned explicit model; ManagedBackend validates exact returned identity. |
| SDK kwargs omit RetryPolicy (SDK default is two retries) | Explicit RetryPolicy(max_retries=0), fixed timeout. |
| Cache defaults on; history may retain raw state/questions/response | Per-client cache off; history/callbacks/tracker disabled; no upstream response retention. |
| Response mapping drops Score legend; public result drops model/usage envelope | Raw native dictionaries and full SDK JSON envelope preserved. |
| No paid consent or our attempt ledger | Explicit allow_paid; ManagedBackend pre-dispatch budget/cache/identity and per-leaf reservations/settlement. |
| No billed cost evidence | Dollar cost remains null. |

The wrapper uses exact-version experimental `_sdk_kwargs`, `_response` and
`_finish` hooks. Native client traffic never goes through a generative Predict
program. Compile CLI and flat/hierarchy research selection choose this backend;
frozen run, evaluate and held-out study phases use the direct SDK. Callers of
the Python compiler explicitly supply their backend. The direct SDK backend is
retained for runtime, rather than deprecated away. The live hierarchy manifest
binds separate compile and runtime identities; the code/manifest change needs
a fresh digest-bound review before paid selection. It cannot revive old consent.

Installed-package socket-forbidden doubles verify payload/envelope equivalence,
real SDK constructor arguments, consent/version/model refusal, zero retries,
cache/attempt accounting, chain/conditional/diamond/nested graph receipts,
interrupted resume and zero-dispatch replay. These are offline mechanics, not
live inference, authenticity, measured gains or production qualification.

Primary sources: [pinned TypeSafe client](https://github.com/stanfordnlp/dspy/blob/3.4.0/dspy/clients/typesafe.py),
[official Jev tutorial](https://dspy.ai/current/tutorials/jev_decisions/),
[SDK usage](https://docs.typesafe.ai/sdk/python/usage).
