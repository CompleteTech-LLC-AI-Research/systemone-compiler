# Runnable hierarchy examples

These three editable sources use the public hierarchy CLI and the deterministic
lexical mock. Their labels, traces, numbers, and compiled reports are **synthetic
software fixtures, not Jev results or deployment approval**. The mock sees only
state and question criteria, never the `expected` labels. No example writes to a
PR, routes a ticket, or performs any application action.

From the repository root after installing the package, regenerate or verify the
checked-in sources, datasets, flat baselines, and expected traces:

```bash
python examples/hierarchy/generate_examples.py --check
```

For each of `support`, `pr_risk`, and `nested`, run an authored/no-optimizer compile
into a **new** directory (shown for `support`):

```bash
s1 compile examples/hierarchy/support/source.json \
  --train examples/hierarchy/support/train.jsonl \
  --validation examples/hierarchy/support/validation.jsonl \
  --calibration examples/hierarchy/support/calibration.jsonl \
  --test examples/hierarchy/support/test.jsonl \
  --min-calibration-samples 1 --out runs/support-hierarchy
s1 run runs/support-hierarchy/hierarchy.s1.json \
  --state examples/hierarchy/support/sample_state.json
s1 inspect runs/support-hierarchy/hierarchy.s1.json
```

The low calibration sample setting is only for these tiny fixtures; review gates
and mock quality metrics do not establish real reliability. Replace `support`
with the other directory names to exercise the other graphs. `flat_source.json`
has the same public contract and can be compiled with the same split files;
`flat_baseline.s1.json` is a draft flat artifact for side-by-side no-key runs.
The installed hierarchy starter is separately available through
`s1 demo --starter hierarchy --out runs/starter-demo`.

`support` routes billing, technical, sales and explicit other/review cases. The
technical route has a second triage step. Its `incorrect_router` fixture has
human-intended `bug_fix` gold, while the mock selects billing and outputs
`refund`; it illustrates error propagation, not measured error rate. A routed
Choice distribution is `branch_conditional`, so it is not a global posterior.

`pr_risk` evaluates independent breaking and operational signals, passes their
typed Noul/Score values into a later stage, then uses a Score-only weighted
mean of two normalized Score questions. The result is advisory: application
code can read `result["status"]` and `result["decisions"]["risk"]`, decide whether
to request human review, and separately own any side effect. It cannot merge,
comment on, or edit a PR.

`nested` invokes the same `check_note` definition at `first/check` and
`second/check`. It includes an absent optional input with a typed default, a
skipped second call, review-required behavior, a one-call budget failure, and
qualified provenance in `expected_traces.json`. Skipped outputs remain missing
and require review; the runtime does not invent a final value.

DSPy structure search and GEPA wording search are opt-in compile paths:
`--architect dspy --structural-rounds 1` or `--optimizer gepa`, together with a
configured `--teacher-model`. Those paths can send examples to a teacher and
incur charges; use `--allow-paid --share-feedback` only after separate paid-call
and data-sharing authorization. This repository does not run them as part of
the no-key examples and does not claim gains from them.

After that authorization, use the same source and four split paths from the
compile command above, a fresh `--out` directory, and one of these option sets:

```text
DSPy structure: --architect dspy --structural-rounds 1 --teacher-model MODEL --allow-paid --share-feedback --teacher-max-calls 20
GEPA wording:   --optimizer gepa --max-metric-calls 128 --teacher-model MODEL --allow-paid --share-feedback --teacher-max-calls 20
```
