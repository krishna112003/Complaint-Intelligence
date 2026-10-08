"""Prompt Registry: loads versioned prompt templates from prompts/ directory.

All prompt files follow the naming convention: <name>_v<version>.txt.
Returns both the prompt text and the explicit version tag for tracking.
"""

from __future__ import annotations

import logging

from complaints.config import get_project_root

logger = logging.getLogger(__name__)


def load_prompt(name: str, version: str = "v1") -> tuple[str, str]:
    """Load a versioned prompt template from prompts/.

    Parameters
    ----------
    name : str
        Base prompt identifier, e.g. "classify" or "extract"
    version : str
        Version identifier, e.g. "v1"

    Returns
    -------
    tuple[str, str]
        (prompt_text, version_tag)
    """
    root = get_project_root()
    prompt_file = root / "prompts" / f"{name}_{version}.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file '{prompt_file}' does not exist in prompts/ directory."
        )

    with open(prompt_file, encoding="utf-8") as f:
        text = f.read()

    version_tag = f"{name}_{version}"
    logger.debug("Loaded prompt '%s' (%d characters)", version_tag, len(text))
    return text, version_tag


def format_untrusted_prompt(
    template: str,
    narrative: str,
    **kwargs: str,
) -> str:
    """Format template securely treating narrative as untrusted input.

    Uses safe substring substitution so prompt JSON schemas with curly braces do not error.
    """
    clean_narrative = narrative.strip()
    result = template.replace("{narrative}", clean_narrative)
    for k, v in kwargs.items():
        result = result.replace(f"{{{k}}}", str(v))
    return result
