# Prompt Changelog

All changes to prompts in this directory are logged here with rationale and versions.

## [1.0.0] - 2026-10-08

### Added
- `classify_v1.txt`: Baseline zero-shot product classification prompt with 10 consolidated categories and XML delimiter tag `<complaint>` for untrusted input injection defense.
- `extract_v1.txt`: Baseline structured extraction prompt (Task B) extracting issue_category, root_cause, urgency, sentiment, requested_resolution, and evidence_quote with exact-substring constraints.
