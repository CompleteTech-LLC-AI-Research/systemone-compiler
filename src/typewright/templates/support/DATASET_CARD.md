# Synthetic support-triage smoke fixtures

48 original, human-readable synthetic examples supplied with this package: 12 training,
12 selection/validation, 12 calibration, and 12 test. Every split has three examples
per department. These are illustrative labels, not adjudicated customer data, not a
statistically powered benchmark, and not evidence about Jev. They contain no real
customer records. No downloads are needed. License: MIT, like this project.

The mock backend is a lexical-overlap fixture, not a substitute for Jev. It is useful
for checking shapes, metrics, caching, CLI flow, and serialization. Do not cite its
metrics as real model accuracy or as proof that optimization works.

For a real deployment, replace these examples with separately labeled, representative
records and substantially larger disjoint splits. Keep all rows from one customer,
conversation, source document, or correlated case in the same group and split.
Noul labels must be JSON true/false, not strings. Score labels range from 0 to 2 here.

The framework checks exact projected-state duplicates and cross-split groups, not
semantic near-duplicates. You remain responsible for temporal and domain leakage.
