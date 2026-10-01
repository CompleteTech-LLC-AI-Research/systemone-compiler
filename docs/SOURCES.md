# Primary sources used for implementation

Documentation inspection date: **September 19, 2026**. Versioned integration targets
are recorded in `INTEGRATIONS.md`; a source link is not a live execution result.

## TypeSafe AI

- Primitives, batching, native response semantics:
  https://docs.typesafe.ai/primitives
- Structured instruction/criteria entry formats:
  https://docs.typesafe.ai/primitives/advanced
- Probability versus confidence:
  https://docs.typesafe.ai/confidence
- Score ranges, ordinal levels, expected score:
  https://docs.typesafe.ai/primitives/score
- Python SDK usage, retry policy, raw question dictionaries:
  https://docs.typesafe.ai/sdk/python/usage
- Response schema:
  https://docs.typesafe.ai/sdk/python/api/types/responses
- Versioned model IDs and moving aliases:
  https://docs.typesafe.ai/models
- SDK release metadata:
  https://pypi.org/project/typesafe-sdk/0.7.0/
- Related question-discovery precedent (not copied as an implementation):
  https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery

## DSPy

- Official documentation:
  https://dspy.ai/
- Official repository:
  https://github.com/stanfordnlp/dspy
- Stable integration-target metadata:
  https://pypi.org/project/dspy/3.3.1/
- Second verified target (release notes, not re-read for the inspection date):
  https://github.com/stanfordnlp/dspy/releases/tag/3.4.0
  https://pypi.org/project/dspy/3.4.0/

## GEPA

- Official documentation:
  https://gepa-ai.github.io/gepa/
- Standalone optimize API:
  https://gepa-ai.github.io/gepa/api/core/optimize/
- Adapter protocol, reflective data, custom proposer:
  https://gepa-ai.github.io/gepa/api/core/GEPAAdapter/
- Official repository:
  https://github.com/gepa-ai/gepa
- Integration-target metadata:
  https://pypi.org/project/gepa/0.1.4/

The source release intentionally uses a standalone GEPA adapter rather than
pretending Jev provides a general text-generation API. No upstream performance
claim is asserted for this framework. Metrics from the bundled fixture are
synthetic software test output only.
