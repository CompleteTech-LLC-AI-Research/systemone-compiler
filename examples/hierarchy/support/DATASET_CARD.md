# Support hierarchy dataset card

All rows are hand-written synthetic support tickets. Six cases per split cover
billing, technical bug and help paths, sales, other/review, and a deliberately
incorrect mock router outcome. IDs, groups, and root strings differ across the
four splits; source text is not customer data. Gold labels encode intended
advisory responses, independent of the lexical mock predictions. They are not
human-audited production labels and cannot support a Jev accuracy claim.

`cases.json` explains the intended routes, while `expected_traces.json` records
what the mock actually executes. The disagreement in `incorrect_router` is
intentional. Branch-specific Choice probabilities are conditional on the
selected specialist and must not be interpreted as full-contract posteriors.
