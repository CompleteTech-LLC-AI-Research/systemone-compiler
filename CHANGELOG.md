# Changelog

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
