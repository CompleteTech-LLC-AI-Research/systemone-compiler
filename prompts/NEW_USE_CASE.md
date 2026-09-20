# Prompt for a coding LLM: add a Jev use case

Use this repository's existing System One Compiler to implement the use case
below. Read README.md, AGENTS.md, and docs/ARCHITECTURE.md first. Do not build a
second framework or write a free-form prompt in place of a typed specification.

My use case: [describe the decision, input fields, consumers, and failure costs]
Available labeled data: [paths, format, ownership, and label provenance]
External actions: [none by default; runtime should return recommendations only]

Create a UseCase YAML with top-level typed state fields, atomic output goals,
Choice option definitions / Noul criteria / ordered Score levels as appropriate.
Keep the user-facing output schema stable. Use the smallest necessary input state.
Prefer understandable rubric definitions over hand-tuned prompt tricks.

Prepare train, validation, calibration, and test JSONL with unique IDs and group
isolation. Do not invent a real dataset or call synthetic labels human-verified.
If only a smoke example is possible, mark it synthetic and do not make accuracy
claims. Keep a dataset card describing source, label policy, exclusions, group
assignment, and intended use.

Add a template draft example, sample input, tests for schema and expected execution
shape, and CLI instructions. Run the no-key workflow first. Configure DSPy/GEPA as
optional compilation with explicit provider, privacy consent, and budgets; do not
run paid calls merely because this prompt mentions them. Explain what data would
be needed to verify improvement on the actual use case.

Inspect generated questions, review semantic drift, and use empirical review gates
without claiming guaranteed risk. No generated code, arbitrary file moves, or
external side effects in the artifact. Return exact changed files, test results,
and commands for the next authorized step.
