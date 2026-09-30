# Hierarchy contract fixtures

These JSON files are the concrete source-format examples for [the hierarchy ADR](../../docs/HIERARCHY_CONTRACT.md). Each leaf embeds a valid flat `systemone-program/v1` program. The small [preview](preview.py) executes the valid graphs with the existing lexical mock backend to pin route order and output meaning before the production compiler and runtime work in issues #8–#21. It is synthetic and never calls Jev or a teacher.

From the repository root with the package installed:

```powershell
python examples/hierarchy_contract/preview.py examples/hierarchy_contract/chain.json
python examples/hierarchy_contract/preview.py examples/hierarchy_contract/conditional.json
python examples/hierarchy_contract/preview.py examples/hierarchy_contract/diamond.json
python examples/hierarchy_contract/preview.py examples/hierarchy_contract/nested.json
```

`chain.json` passes a typed Noul value to a later Score stage. `conditional.json` selects a billing or technical child and records that its reduced Choice distribution is branch conditional. An `other` route ends in review without fabricating a resolution. `diamond.json` runs two independent judgments before a later Score stage reads both. `nested.json` expands the same subgraph at two distinct qualified paths.

The three `invalid_*.json` fixtures fail before constructing a backend: cycle, undeclared root input, and missing public final mapping. `cases.json` holds only synthetic sample inputs and expected execution paths; it is separate from each strict source document. To regenerate the checked-in fixtures after changing the contract, run:

```powershell
python examples/hierarchy_contract/generate_fixtures.py
```

The preview is a reference interpreter for these fixtures, not a public hierarchy API or the final static validator. Future work must preserve the observed behavior while adding frozen artifacts, complete path validation, shared budgets, durable evidence, compilation and metrics. A mock prediction is never a real Jev result.
