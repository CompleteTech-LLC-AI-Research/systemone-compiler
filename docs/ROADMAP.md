# Follow-on work, not implemented features

The core compile/evaluate/runtime workflow is implemented. These are scoped
extensions rather than capabilities claimed by the current release.

1. **Verify and lock a real environment.** Install the optional SDKs, run their
   contracts and authorized inference smoke tests, generate a real dependency lock,
   and add sanitized regression fixtures from observed response schemas.
2. **Establish useful benchmarks.** Independently label representative Choice,
   Noul, and Score tasks; compare template, DSPy-only, GEPA-only, and structural-plus-
   GEPA under matched budgets, several seeds, and sealed holdouts. Report failures
   and no-improvement results along with successful ones.
3. **Richer policy and calibration.** Add task-level utility matrices, statistically
   justified selective-risk control, optional probability recalibration, and
   supervised composition-weight fitting with carefully isolated data.
4. **Scale and operations.** Async batched evaluation, explicitly authorized retry
   accounting, resumable non-pickle experiment records, drift monitoring, signed
   artifact attestations, and a separately reviewed TypeScript runtime.
5. **Beginner interface.** A local wizard for labels and rubrics, improved failure
   cluster visualization, provider-budget preview, and editor support from schemas.

Do not add a user interface ahead of verifying the native adapter and measuring a
real task. Do not make “improved prompts” a promise independent of data quality and
held-out results.
