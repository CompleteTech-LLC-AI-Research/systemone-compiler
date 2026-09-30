# Authored hierarchy compilation

`HierarchyCompiler` compiles a validated `HierarchySource` without a teacher or
optional optimizer. Supply four disjoint root-example splits and one
`ManagedBackend`. Its owner request budget covers every flat and child native
call, including recalculation and held-out test calls. Configure `max_calls`
before compiling; budget failures are execution failures, not low-quality rows.

```python
from s1compiler import HierarchyCompiler, HierarchyCompileOptions

compiler = HierarchyCompiler(backend, options=HierarchyCompileOptions(
    min_calibration_samples=10, max_calibration_error=0.05))
artifact, report = compiler.compile(source, train=train, validation=validation,
                                    calibration=calibration, test=test)
artifact.save("frozen-graph.json")
```

The equivalent explicit sequence is `select`, `calibrate`, `freeze`, then
`test`. Selection runs registered train examples, evaluates the authored graph
and a fixed-contract flat template on validation, and never reads test results.
H09 does not perform structure or wording search; H10/H11 can use these phase
boundaries without invoking a complete compile per candidate.

Calibration fits **post-route final review gates** per public output and
selected origin. It uses only root labels; it never guesses an intermediate
node label. Unreached or insufficiently sampled origins become review-only.
Route-driving thresholds are frozen before calibration, so gates cannot alter
which child runs. Freeze embeds the final gates, child hashes, model, split
digests, implementation versions, owner budget, and provenance. It then
recomputes full calibration paths under the frozen graph. The first test call
occurs only afterward, and a session permits one held-out phase.

The report uses the same root-level objective for paired flat and hierarchy
comparison. Its synthetic status does not imply real Jev quality or deployment
approval. A measured artifact still requires independent acceptance before use.
