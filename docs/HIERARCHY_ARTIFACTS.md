# Hierarchy source and frozen artifacts

Hierarchy v1 has separate authoring and frozen formats. A source document uses
`systemone-hierarchy-source/v1` and contains the public `UseCase`, bounded
`limits`, a root `graph`, and optional local `definitions`. Stages contain either
an embedded flat `Program` (`kind: leaf`) or a named local definition
(`kind: subgraph`). All inputs are typed root or stage-output references;
conditions and final mappings are declarative JSON. See
[`examples/hierarchy_contract/nested.json`](../examples/hierarchy_contract/nested.json)
for a reusable definition called twice.

`HierarchySource.load(path)` validates the source shape. `lower_hierarchy(source)`
expands local definitions into stable qualified instances such as `first/check`
and `second/check`, embeds each leaf program, records virtual subgraph exports,
and produces `systemone-hierarchy/v1`. Definitions are not fetched at load or
run time. The source-to-node map links each authoring leaf to all expanded
instances. The format limits depth, expanded stages and possible native calls;
the caller's request budget may be lower.

Save with `artifact.save(path)` and restore with `load_artifact(path)` or
`HierarchyArtifact.load(path)`. `load_artifact` explicitly dispatches between
flat `systemone-program/v1` and hierarchy v1 without changing flat hashes or
serialization. Unknown formats, duplicate JSON keys, extra fields, non-finite
numbers, inconsistent node IDs, missing local definitions, recursive expansion,
and unsafe path/URL-like identifiers are rejected. Loading reads the local file
only; it never fetches referenced content or calls a model. The root contract,
leaf prompts, bindings, routes, policies, target model and expanded edges all
contribute to `content_hash`. Graph and leaf provenance contribute to the
separate `provenance_hash`. The saved envelope's SHA-256 covers both and detects
accidental or uncoordinated changes; **it is not a signature or authenticity
proof**. Verify publisher identity separately if artifacts cross a trust boundary.

New artifacts start `draft`, with `composition_measured=false` and
`deployment_approved=false`. A measured hierarchy must declare measured
composition and contain only measured leaves; a measured child alone does not
establish measured graph quality. Deployment approval remains an independent
explicit decision. The composition report checksum is a required audit pointer,
not a cryptographic proof that the evaluation was honest. The
[static validator](HIERARCHY_VALIDATION.md) checks authored and frozen graph
dataflow. The [native hierarchy runtime](HIERARCHY_RUNTIME.md) executes frozen
graphs; shared recovery controls, compilation, and evaluation remain in
issues #11–#21.

Export the source and artifact JSON schemas with `s1 schema --out schemas`.
Schema generation, package import and frozen loading need only the base
dependencies; DSPy, GEPA and the live vendor SDK remain optional compile or
live-execution dependencies.
