# Hierarchy static validation

Call `validate_hierarchy_source(source)` for an authored or generated
`HierarchySource`, or call `lower_hierarchy(source)` to validate and produce a
frozen artifact in one step. Both paths use the same validator. It checks each
graph and reusable definition before any backend call, then orders stages with
a stable topological sort. Stages may refer to predecessors listed later in the
source document; list order does not determine execution. Errors name the
source graph, stage, and affected field where possible.

Validation rejects dependency cycles, recursive definitions, unknown or
unreachable stages, undeclared root fields, incompatible typed ports, unsafe
references, route predicates with invalid labels or literal types, and required
inputs that can read a skipped predecessor. A required input from a guarded
stage needs an equivalent or stronger guard on its consumer. Optional ports
may omit a value; typed defaults are allowed only on optional receiving ports.
An `after` edge orders stages but does not supply data. Final mappings must
cover every declared output, preserve Choice labels and Score scale lengths,
and have an explicit `review_required` fallback. Candidates for the same
output must be provably disjoint; ambiguous overlapping routes are rejected.
Only existing flat `Program` Score bindings can use positive weighted numeric
composition.

`validate_hierarchy_compile_inputs(source, splits)` also requires exactly
train, validation, calibration, and test and reuses the repository's duplicate
ID, projected-input, group, and label checks. Construct any paid backend or
teacher only after this preflight succeeds. The frozen artifact loader reruns
typed edge, route, terminal, and cycle checks, including when a file's checksum
was recomputed after an invalid edit. Checksums are integrity checks, not
publisher signatures.

Root inputs are selected by declared top-level field name. This limits what
enters a leaf but does **not** recursively redact nested arrays or objects, or
make retained intermediate values anonymous. Treat graph inputs, outputs, and
cache/evidence files as sensitive application data. Static validation does not
execute a graph; native runtime, compiler, and evaluation integration are
tracked in the dependent hierarchy issues.
