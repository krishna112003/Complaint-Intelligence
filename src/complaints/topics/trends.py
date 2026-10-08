"""Topic trend queries and emerging-theme detection via DuckDB window functions. Phase 8 implements this."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_trends() -> None:
    """Compute topic trends over time (Phase 8)."""
    raise NotImplementedError("Phase 8: topic trends not yet implemented.")


if __name__ == "__main__":
    from complaints.config import setup_logging

    setup_logging()
    compute_trends()
