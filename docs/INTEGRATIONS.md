# Integration targets and local verification

Documentation inspection date: **2026-09-19**. These are the implementation targets:

| Component | Target | Role |
|---|---|---|
| Python | >=3.11 | Core library, CLI, tests. |
| TypeSafe SDK | `typesafe-sdk==0.7.0` | Native `TypeSafeClient.system_one`. |
| Target model | `jev-1.13.0` | Versioned ID, not a moving alias. |
| DSPy | `dspy[litellm]==3.3.1` | `Signature`, `Predict`, explicit teacher `LM/context`. |
| GEPA | `gepa==0.1.4` | Standalone adapter and custom component proposer. |

These are not claims that all dependencies were installed or live-tested in the
packaging environment. See `BUILD_REPORT.md` for the actual boundary. PyPI resolution
may fail on a different Python/platform/provider combination. Verify against
primary documentation and change pins with regression tests rather than silently
installing an arbitrary newest version.

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
configured. Signature calls and LM-forward attempts have separate ceilings.
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

No dotenv loading is built in. No actual teacher model is chosen for the user, and
no API secrets are packaged. The compiled runtime depends on the base library plus
`live`; DSPy and GEPA are optional compile-time extras.
