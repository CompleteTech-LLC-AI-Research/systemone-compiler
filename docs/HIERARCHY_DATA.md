# Hierarchy data and teacher isolation

Hierarchy compilation requires nonempty `train`, `validation`, `calibration`,
and `test` splits. Call `validate_hierarchy_compile_inputs(source, splits)`
before constructing a provider or teacher. It rejects duplicate root IDs,
identical model-visible root inputs, cross-split groups, and identical static
leaf projections. The returned `split_guard` is pinned to that lowered graph.
Rebuild it for every changed candidate before evaluating the candidate.

For each row, bind its original split and identity before execution:

```python
plan = validate_hierarchy_compile_inputs(source, splits)
artifact = lower_hierarchy(source)
row = splits["train"][0]
bound = plan.split_guard.bind(artifact, "train", row)
result = HierarchyRuntime(artifact, backend).run(row.state, lineage=bound)
```

The runner checks every derived child state against its typed port contract
and the validated root or predecessor output before dispatch. The result's
`lineage` contains root ID, group, split, qualified stage IDs, input hashes,
and stage **predictions**. It does not promote model answers to gold labels.
Coarse generated values can repeat across distinct root examples; they remain
attached to the original root identity rather than becoming independent rows.
Stages mixing root and predecessor inputs have no complete static projection:
their derived values are unknown until execution. Equal mixed projections across
distinct registered roots are allowed under the same root-bound lineage rule;
they are not independent observations or permission to share evaluation traces.
Leaf programs require nonempty declared state before split projection runs.
`HierarchyTeacherInputs(guard).routed_train_subset(...)` deduplicates only
traces with the same training group (or root ID), stage-input hash, final
label, human intermediate labels and stage prediction. It rejects evaluation
rows instead of dropping them.

Optional `HierarchyExample.annotations` hold only explicitly human-sourced
intermediate labels, validated against the frozen stage's decision type.
Final `expected` labels remain separate. Hierarchy training does not require
intermediate annotations; unsupervised stages use end-to-end train feedback.
Use `read_hierarchy_jsonl(path, artifact)` for JSONL rows with annotations;
the ordinary flat dataset reader remains unchanged.

Teacher proposal examples, graph traces, and reflection datasets must be
constructed through `HierarchyTeacherInputs`. It accepts registered train rows
and sealed train results only. Validation, calibration, test, and failed-run
diagnostics are rejected before a teacher call. `guard.versions` records
dataset, static train-projection, and observed routed train-projection hashes;
the observed record stays incomplete until every train row has a valid terminal
run. It writes no raw prompt logs. GEPA's
flat adapter applies the same train-only rule to captured reflection traces;
ordinary validation scoring still runs without trace capture.

Train stage states are kept in memory for approved teacher feedback and are
never added to default results or evidence files. Sending that feedback to a
teacher still requires the separate paid-call and data-sharing consents.
