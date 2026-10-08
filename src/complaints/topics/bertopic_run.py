"""BERTopic modeling on customer complaints. Phase 8 implements this."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_bertopic() -> None:
    """Run BERTopic modeling (Phase 8)."""
    raise NotImplementedError("Phase 8: BERTopic run not yet implemented.")


if __name__ == "__main__":
    from complaints.config import setup_logging

    setup_logging()
    run_bertopic()
