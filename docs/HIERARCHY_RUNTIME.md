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
No stage executes application actions or modifies the caller's state. Shared
deadlines, bounded retries, and durable graph evidence are tracked in the
following runtime issues.
