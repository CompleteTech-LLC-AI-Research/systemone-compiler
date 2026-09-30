# Advisory PR-risk dataset card

Three synthetic diff summaries per split represent documentation, localized
service, and broad interface/deployment changes. File paths are inert strings;
the example never opens a repository or performs a PR action. IDs, groups, and
diff strings are disjoint across train, validation, calibration, and test.

Gold Score targets are illustrative judgments, independent of the lexical
mock. The weighted Score composition is deterministic over two validated
question outputs, with each component normalized to its scale before rescaling
to the public three-level Score. Mock outputs and review flags are software
fixtures, not calibrated provider evidence or permission to automate a PR.
