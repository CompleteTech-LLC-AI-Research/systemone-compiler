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
With `HierarchyCompileOptions(architect="dspy", structural_rounds=1..10)`, an
explicitly configured `DSPyTeacher` may propose bounded structure from train
examples and sealed train traces. Each complete graph is validated and scored on
validation before acceptance. Rejected JSON or graph data leaves the prior
candidate intact; provider and budget failures propagate. With
`optimizer="gepa"`, GEPA edits only qualified child question text after structure
selection. It evaluates each candidate through the complete graph on train or
validation roots. Its metric-call budget counts root evaluations; the shared
`ManagedBackend` separately counts every native stage call. The compiler
rechecks selected wording on validation, then calibrates and freezes it before
the single test phase. Teacher uploads require separate sharing consent, and
this offline source release makes none by default.

The proposal output contains only `definitions` and `graph`. The compiler copies
the root source contract, model and graph limits; generated text is never run as
code. The frozen artifact records a semantic review manifest with changed
routing goals, child prompts, composition, and its exact graph hash. A valid
schema does not certify semantic equivalence or deployment approval.
If a selected structure changes routing, prompts or composition, the artifact
remains draft for real-provider execution until semantic review is resolved;
the compiler does not self-approve it.

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

`components_from_hierarchy` and `hierarchy_from_components` address text by
qualified expanded stage and question ID. This lets two instances of a reused
definition receive different text without changing sibling stages. A changed
text candidate becomes a draft artifact requiring semantic review and final
policy recalibration. `optimize_hierarchy_gepa` uses installed GEPA with a
train-only reflective dataset and no checkpoint loading or tracking.
