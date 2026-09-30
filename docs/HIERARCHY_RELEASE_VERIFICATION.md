# Hierarchy v1 release verification

This is the software evidence map for epic [#6](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/6), current through 2026-09-30. It does not approve deployment or claim a measured Jev gain. Check the linked issues and PRs for current hosted status before release.

| Scope | Issue | Merged implementation |
|---|---|---|
| Typed contract and fixtures | [#7](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/7) | [#22](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/pull/22) |
| Versioned source and frozen schemas | [#8](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/8) | [#23](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/pull/23) |
| Dataflow and lowering | [#9](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/9) | [#24](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/pull/24) |
| Graph runtime | [#10](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/10) | [#25](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/pull/25) |
| Budgets and recovery | [#11](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/11) | [#26](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/pull/26) |
| Durable evidence | [#12](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/12) | [#27](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/pull/27) |
| Split and teacher-trace isolation | [#13](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/13) | [#28](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/pull/28) |
| Graph evaluation | [#14](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/14) | [#29](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/pull/29) |
| Authored compile/freeze/test | [#15](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/15) | [#30](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/pull/30) |
| DSPy structure search | [#16](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/16) | [#31](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/pull/31) |
| GEPA wording search | [#17](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/17) | [#32](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/pull/32) |
| Public CLI, API, and package | [#18](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/18) | [#33](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/pull/33) |
| Synthetic examples and traces | [#19](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/19) | [#34](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/pull/34) |
| Offline preregistered study runner | [#20](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/20), **open** | [#35](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/pull/35); live-manifest acceptance remains open |
| Integration and final release gate | [#21](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/issues/21), **open** | [#36](https://github.com/CompleteTech-LLC-AI-Research/systemone-compiler/pull/36), draft |

## Verification boundaries

`tests/test_hierarchy_validation.py` covers cycles, typed references, contract drift, and conditional dominance. `tests/test_hierarchy_artifact.py` covers size limits, path-like definition references, format dispatch, and flat hash compatibility. Runtime, budget, data, evidence, architect, GEPA, compiler, metrics, examples, and study test modules cover malformed responses, model drift, split leakage, small-branch review, cancellation, retries, lost attempt receipts, strict replay, and held-out freeze order. `tests/test_hierarchy_security.py` guards the hierarchy modules against dynamic execution and unsafe serialization/process imports. These are software tests; model prompt-injection resistance is not established by them.

The CI matrix runs pytest, Ruff, wheel build, installed-wheel flat and hierarchy no-key demos, and runtime import isolation on Python 3.11–3.13. Its optional job installs the pinned SDK/DSPy/GEPA extras and runs interface tests with fake providers. No CI job uses a live key. The exact results belong to the PR's check runs; a green configuration file alone proves nothing.

Issue #20 still needs an independent reviewed four-split dataset, prior-holdout exclusion fingerprints, verified price and external billing caps, and separate paid/teacher-sharing authorization to produce and review a complete live frozen manifest. Issue #21 and epic #6 must remain open until that criterion and their remaining checks are satisfied. The existing flat benchmark is not the new hierarchy holdout.
