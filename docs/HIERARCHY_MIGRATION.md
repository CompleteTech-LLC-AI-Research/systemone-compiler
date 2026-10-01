# Flat v1 to hierarchy v1

Flat `systemone-program/v1` artifacts and `Runtime.run` remain supported. A flat
artifact still makes one native request per uncached evaluation. Migration is
opt-in: use `typewright init NAME --starter hierarchy` to get an editable source graph
and four disjoint synthetic splits, or author a `systemone-hierarchy-source/v1`
document under the [contract](HIERARCHY_CONTRACT.md).

Compile with the existing `typewright compile` command and all four split paths. The
hierarchy source keeps the same public `UseCase` names, types, Choice labels,
Score scales, input allowlist, and target model. Leaf programs and conditional
routes are internal; they do not change that contract. The output is a separate
`systemone-hierarchy/v1` artifact. Load it with `HierarchyArtifact.load` and run
it with `HierarchyRuntime`, or use `typewright run` and `typewright evaluate`, which dispatch by
artifact format. Do not rename a flat artifact or edit its `format` field to
convert it.

Budget for selected leaves, not just root examples. A graph can stop with
`review_required`, `failed`, or `cancelled`; callers must handle those outcomes
without treating missing decisions as complete. Branch Choice probabilities
can be conditional on the chosen route, and review gates are not global
posteriors. A native playground export represents one resolved leaf only.

Keep flat v1 results as a baseline while validating the graph on independent
data. `typewright-study` provides a separate synthetic software check and a gated live
protocol. Synthetic output does not establish a quality gain or authorize live
inference, teacher sharing, deployment, or application side effects.
