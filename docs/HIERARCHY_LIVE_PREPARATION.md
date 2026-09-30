# Human-annotated hierarchy study preparation

The proposed study uses **MASSIVE 1.0 fr-FR**, with its full 60-intent public
contract. The [authors' repository](https://github.com/alexa/massive) documents
the human judgments, aligned English originals, official partitions, and
CC BY 4.0 data license. This preparation makes no provider calls. Upstream
human judgments establish label provenance; study independence and graph
semantics still require human review before live registration.

Download the official archive to an ignored local directory:

```powershell
Invoke-WebRequest https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz -OutFile runs/amazon-massive-dataset-1.0.tar.gz
python -m s1compiler.hierarchy_research_data --archive runs/amazon-massive-dataset-1.0.tar.gz --prior-text-exclusions runs/prior-flat-test-text-exclusions.json --out runs/massive-fr-study-inputs
```

Use the verified prior-flat-test text exclusion export described in
[the study guide](HIERARCHY_STUDY.md). The preparer verifies pinned archive and
member bytes, reads only allowlisted regular members without filesystem
extraction, and requires aligned source identities/labels/partitions.

Retain a row only when at least two distinct human raters marked its intent
correct and language on target. Exclude prior holdout text in **both French
and aligned English**, and discard every occurrence of duplicate French or
English source text. Keep the official train and test partitions; assign
official dev rows to validation or calibration by a fixed English text hash.
No prediction or measured task quality influences this procedure.

Outputs include four JSONL splits, a fixed source contract, authored hierarchy
draft, flat baseline, source license, aggregate provenance/coverage report,
request estimates, and an explicitly **pending** human attestation. All original
60 labels remain supported; filtering can leave rare labels absent from an
individual split. Groups cover identities and bilingual text, not original
speaker membership. Public-corpus pretraining contamination and semantic
near-duplicates remain limitations. Hashes are sensitive and do not anonymize
data; keep generated inputs and exclusion lists off Git.

The authored graph routes among four scenario families and then classifies a
branch intent; intermediate labels are derived ontology mappings, not additional
human gold. Its conditional probabilities remain branch conditional. The
report proposes one DSPy structural round, a GEPA metric limit of
`5 * (validation_rows + 3)`, eight teacher
signature calls, 4,096 teacher output tokens, and a five-node/call graph bound.
These are proposed limits, not approval or measured results.

Before registration, a human must review the source/taxonomy, provenance,
independence limitations, and data-sharing scope. Specify the teacher model,
verified per-call price bounds, and an externally enforced billing cap. Paid
selection and train-only teacher sharing need separate consent. Selection
produces calibrated frozen graph digests for the final live manifest; its exact
frozen identity and semantic changes need review before held-out execution.
Do not relabel the pending attestation or draft as approved, frozen, or measured.

After registration, `s1-study manifest` writes the complete pre-spend live
manifest, including the teacher endpoint resolved from `S1_TEACHER_API_BASE`
at that moment, with no provider calls. Complete its `review_template` with the
rechecked quotes, the enforced cap, and the consents, then bind it with
`s1-study review`. Live selection refuses to start unless `--reviewed-manifest`
matches the protocol, the current software parameters, and the same teacher
endpoint. See [the study guide](HIERARCHY_STUDY.md) for the exact commands.

## Prepared proposal for issue #20

The 2026-09-30 local preparation retained 14,960 rows: 10,451 train, 917
validation, 902 calibration, and 2,690 test. It excluded 484 rows without the
required human agreement, 50 prior-holdout overlaps, and 1,027 duplicate rows.
All 60 intents remain in the contract; train covers 60, validation 59,
calibration 58, and test 59. Results cannot establish quality on absent labels.
The preparation digest is
`0febe4a8d584d75dd97c2e6c2acdff6ee366860f11255445afe4c911f4f49d8d`.

The concrete GEPA limit is 4,600 metric calls, exceeding the mandatory initial
validation evaluation. Registration rejects plans below that minimum before
any provider is constructed. The conservative allocation is 610,371 native selection/calibration attempts,
40,350 native held-out attempts, and 16 teacher provider attempts (eight
signatures). These are hard allocation ceilings, not expected realized use.
The five-node bound includes branches that are mutually exclusive in the
authored graph; its ordinary route uses two requests per completed root.

The proposed native model is `jev-1.13.0`. [TypeSafe's model documentation](https://docs.typesafe.ai/models)
lists $0.042 per million input tokens with output free. A proposed native
per-attempt bound of $0.0028 conservatively covers a 65,536-token request at that
direct price. The proposed teacher is DeepSeek Flash via the direct API,
`openai/deepseek-flash` with `S1_TEACHER_API_BASE=https://api.deepseek.com`.
Configure a direct DeepSeek credential in `S1_TEACHER_API_KEY`; an existing
gateway credential does not authorize this endpoint.
[DeepSeek's pricing documentation](https://api-docs.deepseek.com/quick_start/pricing/)
currently maps that name to V4.1 Flash and lists peak prices of $0.30 per million
uncached input tokens and $1.20 per million output tokens. The proposed $0.65
teacher-attempt bound exceeds a two-million-input-token allowance plus the
4,096-output-token limit. Both quotes were checked on 2026-09-30; verify them
again before approval and account for any endpoint markup or billing changes.

Those proposed price bounds yield $1,719.4388 for selection/calibration and
$112.98 for the held-out phase: $1,832.4188 in total. A proposed **$1,850 external
cap** has not been configured or verified. No spending or teacher sharing is
approved. Only train examples and train traces would be eligible for teacher
sharing; validation, calibration, and test remain excluded. Selection approval
does not authorize the held-out phase. The teacher's moving alias must be
rechecked and recorded before selection; native Jev remains pinned.
