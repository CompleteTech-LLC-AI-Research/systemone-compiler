# Security, data use, and budget boundaries

## Data flows

A native TypeSafe request contains the projected state and all compiled questions.
A DSPy teacher call contains the source declaration, current plan/components, and
selected projected training examples or training error traces with labels. The
teacher may therefore see sensitive task data. Consent to TypeSafe is not assumed
to authorize sharing with a second provider; `--share-feedback` is required.
Validation/calibration/test records are used by the native evaluator, not the
teacher. Native evaluation still uploads those states to TypeSafe on live runs.

The optional gateway examples in `examples/ai-gateway/` send the same projected
state to additional third-party routes. They refuse to run without
`allow_paid=True`, and each route is off unless its `AI_GATEWAY_ROUTE_N_ENABLED`
setting is `true`. Enable a route only with separate recorded approval; consent to
TypeSafe covers none of them.

Top-level projection is an allowlist, not PII detection. Nested objects and arrays
are sent as declared. Scrub secrets and sensitive content before labeling/loading
the dataset. Do not test on another party's private data without authorization.
Review both configured endpoints, including any environment-variable overrides.

## Credentials and logs

Keep keys in local environment variables or a secret manager. The package includes
only fake test keys and an empty/commented environment template. CLI errors aim
not to echo input values or credentials. Library exceptions retain causes for
local debugging; dumping tracebacks can still reveal information. Third-party
SDKs, DSPy/LiteLLM, tracing callbacks, shell history, and custom integrations are
outside that guarantee. Do not enable raw request/body debug logging with private
data. TypeSafe debug logging can include request/response bodies.

DSPy caching is disabled by this compiler. That setting affects the process-global
DSPy cache configuration; prefer an isolated compile process when embedding in
another DSPy application. No external experiment tracker is configured, and GEPA
pickle checkpoint loading is not used. Provider libraries may retain in-memory
history; shut down/clear a compiler process according to your data policy.

## Cache and artifacts

The default answer cache is in-memory. `--cache PATH` explicitly enables SQLite
persistence with a one-day TTL. Keys hash provider identity, versioned model,
projected state, and question definitions. Values contain returned answers and
metadata, not raw state text. Outputs can still reveal input information, and
hashes permit dictionary attacks on low-entropy inputs. This is not anonymization
or encryption. Treat cache files, reports, generated question descriptions, and
teacher-generated examples as potentially sensitive. Disk permissions are best
effort and OS-dependent. Do not put them in a public repository by default.

Artifact checksums cover provenance too, but are not signed attestations. Review
untrusted artifacts before using any provider: an attacker able to edit an
artifact can also recompute its hash. Model-generated JSON is shape-validated,
not semantically verified; neither typing nor wording rules eliminate prompt
injection or systematic misclassification.

## Budgets and failure modes

Hierarchy executes one native request per selected leaf, so one root can consume
multiple attempts. The same caller-owned `ManagedBackend` budget spans those
leaves; graph and per-node limits can only tighten it. Optional durable evidence
retains typed responses, derived stage inputs, attempt receipts, and routing
decisions. Store its directory with the same protection as source examples and
provider outputs. Resume reconciles reserved attempts; an uncertain flat study
request is never retried automatically. Evidence checksums detect inconsistency,
not authorship or confidentiality. The hierarchy study's estimated dollar cap
uses caller-supplied per-call prices and must be independently checked against a
provider-enforced billing limit before live approval.

Live paths require explicit consent. Target-side budgets count SDK request
attempts before dispatch, with SDK automatic retries disabled. Cache hits do not
consume target calls. The teacher budget counts DSPy signature invocations;
framework-internal formatting fallback or provider behavior can differ from a
one-signature/one-HTTP-request assumption. Output token limits do not bound total
input tokens or exact dollars. Reports leave unknown costs null.

There are request character and document/record size bounds, timeouts, a bounded
structural loop, and GEPA metric-evaluation limits. These are not a full denial-of-
service defense for untrusted network users. The package is a local developer
library, not a hardened multi-tenant server. There is no dollar-price estimator or
provider-side quota administration. Set spending limits through your provider as
needed and do not authorize search with unlimited credentials casually.

Infrastructure failures stop execution. No silent downgrade to mock, alias model,
or alternate paid provider occurs. No completed result is claimed after a failed
compile. Partial debug traces are not a successful artifact.

## Action boundaries

The runtime returns data and review flags only. It does not move/delete files,
execute shell commands, approve PRs, send messages, or modify accounts. Downstream
actions require application-owned permission checks, idempotency, auditability,
and explicit review where appropriate. `review_required=false` is not authority
to perform an action. Sensitive/high-impact deployments need independent review;
the bundled synthetic dataset is not suitable validation for them.
