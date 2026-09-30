# Hierarchy v1 contract (ADR)

Status: accepted design contract for [issue #7](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/7). The current `Program` and `Runtime` remain flat; issues #8–#21 implement this contract. The JSON fixtures in `examples/hierarchy_contract/` are executable specifications for those issues, not a claim that the current package already runs graphs.

## Decision and scope

A hierarchy is a finite, declarative graph of native typed Jev calls. A leaf is an existing `Program` with exactly one target model and one native request per uncached attempt. A child can read a declared root field or a validated output of a predecessor. Conditions choose which child stages execute. Independent selected stages may run before a join. Reusable subgraphs expand statically under qualified instance IDs (`billing/check`, `technical/check`), so each call site has separate provenance. This version uses a deterministic serial scheduler; independent branches need no concurrent execution for correct semantics.

Supported graphs include chains, branches, fan-out/fan-in, and finite nested reuse. A graph does not invoke tools, send messages, change files, or perform any application action. Loops, recursion, arbitrary code and shell expressions, remote includes, cross-model calls, and distributed scheduling are outside the format. Ordinary application code owns side effects.

`systemone-hierarchy-source/v1` is an authoring document. It contains a fixed public `source` (`UseCase`), limits, a root graph, and optional named subgraph `definitions`. `systemone-hierarchy/v1` is the separately versioned frozen artifact created after validation and lowering. A frozen artifact embeds every leaf program and expanded node, the root contract, all edges/conditions/output mappings and policies, a source-to-node map, model identity and provenance. No runtime fetch or arbitrary import is permitted. The existing `systemone-program/v1` loader and hashes stay unchanged; artifact format dispatch is explicit.

The authoring graph has this shape (the fixtures give complete instances):

```json
{
  "format": "systemone-hierarchy-source/v1",
  "source": {"name": "example", "model": "jev-1.13.0", "state": {}, "decisions": {}},
  "limits": {"max_expanded_nodes": 64, "max_depth": 8, "max_native_calls_per_example": 64},
  "definitions": {},
  "graph": {"inputs": {}, "outputs": {}, "stages": [], "final": {}}
}
```

The root `graph.inputs` and `graph.outputs` must exactly equal `source.state` and `source.decisions`. A subgraph definition has the same graph fields but its own declared input/output interface. An invocation of a definition is a stage of kind `subgraph`; other stages have kind `leaf` and embed a strict flat `Program`. The maximum of 64 *expanded* stage instances, depth 8, and 64 possible native leaf calls per example applies after expansion. A leaf retains its existing 64-question bound. These are format limits, not permission to spend 64 calls: the caller's shared budget and any lower configured limits still apply. All leaf model IDs must equal `source.model`.

## Typed references and state

Each stage declares `id`, `kind`, and `inputs`. An input value is bound by one of:

```json
{"root": "message"}
{"stage": "router", "decision": "department", "field": "value"}
```

`root` means a field in the containing graph's declared `inputs`. A stage reference names a preceding stage or subgraph output. Allowed fields are `value`, `probabilities`, `p_true`, `vendor_confidence`, `gate_score`, and `review_required`, according to the decision type. There is no generic JSON path. `value` means a Choice label, a Noul boolean after its frozen threshold, or a Score on its declared scale. `p_true` is a Noul probability and is never vendor confidence. A Choice probability map and Score probability map are typed objects; neither silently becomes a scalar. Optional values such as vendor confidence require optional destination fields and an explicit missing-value rule. Source field projection still strips undeclared top-level fields. The derived namespace is separate from root state; names never shadow root inputs implicitly. Nested arrays/objects are not claimed to be recursively minimized or anonymized.

`Program.state` declares the receiving leaf fields. Every bound input must match the receiving field's type, with finite numeric values and the existing strict boolean/integer distinction. Required inputs must exist on **every** path that enables the stage. An optional input may be absent only when that field is declared optional; any default must be a typed literal in the graph. A reference to a skipped or review-blocked stage cannot resolve. Referenced outputs become available only after the upstream native answer and model identity pass validation. Stage IDs and definition IDs use lowercase snake case; qualified IDs use `/` only as the compiler's path separator. Duplicate qualified IDs are errors.

Static analysis infers dependencies from inputs, conditions, and explicit `after` edges. It rejects cycles, recursive definitions, unreachable stages, missing references, incompatible types, undeclared root fields, and expansion or call limits before execution. Source list order does not define execution order: a stable topological sort by qualified ID resolves ties. An input from a conditionally skipped branch requires an equivalent guard on the consumer, an explicit optional value, or a review terminal; a successful consumer cannot read a missing value. A shared ancestor runs once per graph execution.

## Conditions, routes, joins, and final outputs

`when` is absent for an unconditional stage. Otherwise it is `{ "all": [predicate, ...] }` with a nonempty list. Predicates contain a typed `ref`, an operator, and a literal: `eq`/`in` for Choice or Noul values, and `lt`/`lte`/`gt`/`gte` for finite Score values. Conditions can read root fields only when the literal and field types match. There is no `or`, code expression, implicit truthiness, or dynamic field access; alternatives use separate guarded stages. A condition is evaluated only after all referenced predecessors finish. Missing or invalid conditions fail closed. Multiple enabled branches may run if the graph explicitly needs fan-out. A join stage uses input references and optional `after` edges; it runs only when all required predecessors are completed. Independent selected stages execute serially in deterministic topological order.

Every graph declares `final` mappings for all its public outputs. A mapping contains one or more `candidates` and an explicit `on_missing: "review_required"`. Each candidate names a stage and a decision. At most one candidate may complete on a path; multiple completed candidates are a validation or runtime error, never last-writer-wins. An eligible successful path must resolve every public output, with unchanged name, type, Choice label space, Score scale length, allowed root inputs, and model. Uncovered routes, low-confidence deferment, and impossible joins end with a typed `review_required` outcome that has no fabricated missing decisions. If only some outputs are resolved, the graph is incomplete and cannot be scored as a complete prediction.

A terminal child Choice may use a proper subset of root labels only with an explicit one-to-one `label_map` into the public Choice labels. Its selected value maps to a public label, while its probabilities remain marked `branch_conditional` and cannot be represented as a full root posterior. A full-label child uses `distribution_scope: "full_contract"` and retains the original distribution and confidence as distinct fields. Noul P(true), Score expectations and component Score distributions also retain their respective meaning and scope. The graph never multiplies router and child probabilities, pads absent labels with zero, or invents a joint posterior/vendor confidence. A Score produced by a join may only use the existing positive-weighted composition of Score/Noul components: normalize each Score by its maximum level, use Noul P(true), average with positive weights, and rescale to the declared public Score range. Its gate is a named heuristic, not a probability. Direct Choice/Noul outputs do not use weighted composition.

Each stage has `on_review: "defer"` by default or `"continue_marked"`. Under `defer`, descendants requiring that stage are blocked and the graph enters review when no complete output path remains. Under `continue_marked`, valid answers may route onward, but the path and final result remain review-required even if children are confident. A provider, auth, schema, model-drift, or budget error is a failure, not review and not a zero-quality example. Stage states are `pending`, `running`, `completed`, `skipped`, `review_blocked`, `failed`, and `cancelled`; terminal graph states are `completed`, `review_required`, `failed`, and `cancelled`. All skipped and blocked nodes have reason codes. A `completed` graph has exactly the declared final outputs; review/failure outcomes explicitly report any partial path without pretending it is complete.

## Calibration, budgets, and evidence

Any policy that can affect routing or stage admission is part of the selected structure and freezes before calibration: route predicates, route-driving Noul thresholds, ancestor `min_gate`, `force_review`, and `on_review`. Calibration may fit terminal review policies that do not change which nodes execute. If a later version fits a routing policy, it must run a separately budgeted end-to-end calibration under that policy; replaying only old parent answers cannot create previously unexecuted child answers. Selection uses validation; only train examples/traces reach DSPy or GEPA; calibration occurs after prompt selection; the complete artifact freezes before test. Derived stage inputs inherit their root example ID/group/split. Intermediate model predictions are not ground-truth labels.

One caller-owned attempt budget spans every expanded leaf, branch, cache miss, and authorized retry. Calls are reserved before dispatch; cache hits and skipped stages do not spend native attempts. A graph may have per-node caps tighter than its overall cap but cannot reset the shared cap. Accounting distinguishes graph executions, leaf evaluations, native attempts, cache hits, retries, unknown usage, and actual latency. Cache identity for a raw leaf response derives from provider/model/wire request; routing and final results are recomputed under the frozen graph and policy hash. A changed child payload cannot reuse an incompatible response. Distributed exactly-once delivery is not promised.

Durable replay/resume is opt-in because intermediate data are sensitive. The documented evidence mode must keep enough typed response validation evidence to reproduce normalization and routing, with stage/model/graph IDs, status, usage, attempted calls and checksum receipts. Normalized probabilities alone may lose the original rounding grid and are insufficient for strict replay. A run without retained evidence reports resume unavailable. The existing run owner, original limits, history and completed evidence survive recovery; uncertain in-flight attempts remain charged or reserved. Hashes detect changed content but are not authentication signatures.

## Package boundaries and migration

`models.py` will add source/frozen hierarchy models and explicit format dispatch; `architect.py` will propose bounded graph JSON; `compiler.py` will separate graph selection, calibration, freeze and test; `runtime.py` will execute a validated frozen graph through existing leaf `Runtime`; `backends.py` will own shared attempts and response identity; `metrics.py`/`reporting.py` will handle completion, probability scope and path metrics; CLI/schema/export will expose both formats. `gepa_adapter.py` will address text by stable qualified stage/question IDs and evaluate the whole graph. These changes do not alter `Program`, its v1 artifact hash, or flat `Runtime.run` semantics.

Authoring and runtime imports continue to work with the base package. DSPy/GEPA are compile-time extras; TypeSafe is loaded only when a live backend is explicitly selected. Public CLI defaults stay mock. Frozen hierarchy release status is earned by evaluating the **composition** and every executed leaf identity; a measured wrapper cannot launder synthetic/draft children. `deployment_approved` remains separate and false by default. A native playground export can materialize one resolved stage request, not a whole multi-call graph as one Jev request.

The fixtures accompanying this ADR cover a chain, conditional specialist branch, diamond join, nested reuse at two call sites, and invalid cycle/unbound/incomplete plans. They are synthetic contract inputs, not measured Jev predictions. Implementations in #8–#21 must make the valid fixtures executable by the public package and reject the invalid fixtures before the affected model call.
