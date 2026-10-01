# Changelog

## Unreleased: rename to Typewright

- Frozen hierarchy artifacts are now checked for lexical scope on load: node origins must match
  their export chain, references, `after` edges, and finals may not cross subgraph scopes (except
  through the inputs an export passes down), and exports must have descendant nodes (#85).
- Renamed the distribution and import package from `systemone-compiler` /
  `s1compiler` to `typewright`. The primary commands are `typewright` and
  `typewright-study`; `s1` and `s1-study` remain as deprecated aliases. The old
  `s1compiler` import path is removed.
- Unchanged by design: the `.s1.json` artifact extension, `systemone-*` format
  identifiers, the artifact provenance key, and the `S1_TEACHER_*` environment
  variables, so existing frozen artifacts and checksums stay valid.
- Historical verification reports, `reports/`, and the shipped manifests still
  record the old name because they describe what was run at the time.

## Unreleased: hierarchy v1

- Added a separate versioned hierarchy source and frozen graph artifact with
  validated typed references, conditional routes, nested reuse, and serial
  native leaf execution under one caller budget. Flat v1 remains supported.
- Added bounded DSPy structure and GEPA wording selection, four-split
  calibration/freeze flow, graph metrics, and durable attempt evidence for
  strict replay and resume.
- Added hierarchy CLI/schema/playground dispatch, packaged starter data, and
  synthetic support, PR-risk, and nested-reuse examples.
- Added a preregistered three-arm study runner and offline synthetic smoke.
  Its live protocol requires independent reviewed data and separate paid and
  teacher-sharing approvals; no measured hierarchy gain is claimed.
- Added `s1-study manifest` and `s1-study review`: a complete pre-spend live
  manifest with every run parameter, a digest-bound human review of required
  attestations, and a live `select --reviewed-manifest` gate that refuses any
  parameter drift before building a provider or teacher. No live study has
  been registered, reviewed, or executed.
- Rebuilt the tracked `wheels/` wheel from current main so it includes the hierarchy modules and
  the `s1-study` entry point, and refreshed the hashes in `MANIFEST.sha256` for files changed since
  it was last updated. `MANIFEST.sha256.as-shipped` is unchanged. Release wheels remain unsigned
  local builds, and the manifest is an integrity check, not a signature.

See [migration](docs/HIERARCHY_MIGRATION.md) for format and caller changes.
