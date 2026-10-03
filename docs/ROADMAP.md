# Follow-on work, not implemented features

The core compile/evaluate/runtime workflow is implemented, along with the
hierarchy v1 graph format, its preregistered study runner, and the flat benchmark
protocols. No live provider study has been completed. These are scoped extensions
rather than capabilities claimed by the current release.

1. **Verify and lock a real environment.** Install the optional SDKs, run their
   contracts and authorized inference smoke tests, generate a real dependency lock,
   and add sanitized regression fixtures from observed response schemas. This
   includes future SDK/DSPy releases beyond the verified profiles. DSPy 3.3.1
   and 3.4.0 teacher contracts are verified offline; native compilation requires
   the exact DSPy 3.4.0 / SDK 0.7.0 profile. Live provider qualification and a
   resolved transitive lock remain outstanding ([#90](https://github.com/Jev-Engineering/TypeWright/issues/90)).
2. **Establish useful benchmarks.** The BANKING77 and multi-benchmark protocols
   and the hierarchy study runner exist, but their live phases have not been
   executed and the hierarchy study still needs independently reviewed inputs.
   Independently label representative Choice, Noul, and Score tasks; compare
   template, DSPy-only, GEPA-only, and structural-plus-GEPA under matched budgets,
   several seeds, and sealed holdouts. Report failures and no-improvement results
   along with successful ones.
3. **Richer policy and calibration.** Add task-level utility matrices, statistically
   justified selective-risk control, optional probability recalibration, and
   supervised composition-weight fitting with carefully isolated data. Fitting
   Score cuts and Choice selection weights is now implemented natively from cached
   calibration predictions in flat programs, explicitly enabled with
   `compile --fit-decision-knobs`, with conservative grouped five-fold checks and an
   explicit unchanged result (#92; owner approved native fitting for #95).
   This is not posterior calibration or a measured gain.
4. **Scale and operations.** Async batched evaluation, explicitly authorized retry
   accounting, drift monitoring, signed artifact attestations, and a separately
   reviewed TypeScript runtime. Resumable non-pickle records already exist for
   hierarchy evidence (create, resume, replay); flat experiment runs do not have
   an equivalent.
5. **Beginner interface.** A local wizard for labels and rubrics, improved failure
   cluster visualization, provider-budget preview, and editor support from schemas.

6. **Maintain native integration qualification.** Compile-time native evaluation
   now uses the verified DSPy 3.4.0 `TypeSafe` wrapper; frozen execution and
   held-out evaluation retain the direct SDK and never import DSPy
   ([#91](https://github.com/Jev-Engineering/TypeWright/issues/91)). Recheck full
   payload/envelope preservation, identity, consent, zero retries and budgets
   against upstream changes before expanding its exact version guard. This
   offline software qualification does not establish live quality or justify
   replacing the approved native policy fitting with `ReAnchor`. A generative-LM
   baseline through DSPy's decision types remains outside the current plan.

Do not add a user interface ahead of measuring a real task. Do not make “improved prompts” a promise independent of data quality and
held-out results.
