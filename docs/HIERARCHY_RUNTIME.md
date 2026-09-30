# Frozen hierarchy execution

`HierarchyRuntime` runs a validated `systemone-hierarchy/v1` artifact through
the existing `ManagedBackend` and flat `Runtime`. It uses one caller-provided
backend for every leaf and makes no provider call while loading or validating
the graph. It does not import DSPy, GEPA, or the live vendor SDK unless an
application explicitly constructs the live backend.

```python
from s1compiler import HierarchySource, HierarchyRuntime, lower_hierarchy
from s1compiler.backends import ManagedBackend, MockBackend

source = HierarchySource.load("examples/hierarchy_contract/conditional.json")
artifact = lower_hierarchy(source)
backend = ManagedBackend(MockBackend())
try:
    result = HierarchyRuntime(artifact, backend).run({"message": "refund invoice charge"})
finally:
    backend.close()
```

This example is a **synthetic software check**, not a Jev quality measurement.
`HierarchyRuntime.load(path, backend)` loads a saved artifact. Existing flat
`Runtime` and `Program` APIs are unchanged.

The runner projects declared root fields, builds separate child states from
typed bindings, and publishes each answer only after the leaf runtime validates
model identity and native response semantics. It evaluates conditions and
joins in stable topological order. A skipped or review-blocked predecessor
cannot supply a required child input. Optional ports may use their validated
literal defaults. Reusable subgraphs execute under qualified IDs and only
once per invocation. Backend failures stop the graph and produce `failed`,
without fabricated decisions; a caller cancellation callback produces
`cancelled` before the next stage dispatch.

Results contain `status`, `decisions`, native `executed` leaf IDs, a `path`,
per-stage statuses and reason codes, the graph content hash, model identity,
synthetic/live identity, and attempt/cache-hit counts. A completed result has
all public decisions. Missing outputs or propagated review produce
`review_required`. A branch-local Choice keeps its child probability labels
and `branch_conditional` scope; only its selected value is mapped to the public
label. The runner never multiplies route probabilities into a joint posterior.
No stage executes application actions or modifies the caller's state.

## Shared budget, caching, and recovery

The caller owns one `ManagedBackend` for the whole graph and closes it after
the run; leaves never close sibling resources. Its budget charges every native
attempt before dispatch. `HierarchyRuntime(..., max_graph_attempts=N,
node_attempt_limits={"signal": 1})` can tighten the artifact's per-example
limit and individual leaf caps. A cache hit consumes no attempt. The serial
scheduler shares the cache's SQLite connection under the backend lock.

Only validated native answers enter the optional cache. Its key includes the
provider identity, wire format, pinned model, questions, and projected state.
Changing routing or policy changes the graph hash and recomputes the result;
identical leaf requests may reuse validated raw answers. Cache files and
fingerprints are sensitive and are not anonymous. The result's accounting
separates actual attempts, cache hits, retries, unknown usage, elapsed latency,
and the static worst-case leaf count. Unknown tokens and dollar cost remain
unknown, never estimated from mock answers.

`run(..., timeout_seconds=30, cancel_requested=callback)` checks cancellation
before each dispatch and after a late response. Already admitted work finishes
and stays charged. No retry is the default. An explicit
`GraphRetryPolicy(max_transient_retries=1)` permits bounded retries for transport
timeouts/connections; malformed answers need a separate
`max_invalid_response_retries` setting. Identity, contract, and budget errors
remain fatal. Retry settings are recorded in the attempt ledger. The runner is
serial, so batch-only `recover_batch` is not invoked.

`result["accounting"]["attempt_ledger"]` is an in-memory handoff for the
durable evidence work in H06. Restoring it marks unsettled calls uncertain and
requires the same graph hash, caps, retry policy, and a reconciled owner budget.
It does not restore completed stage outputs or provide exactly-once remote
execution. Applications must retain this evidence securely before attempting
recovery; H06 will define durable checkpoints and strict replay.
