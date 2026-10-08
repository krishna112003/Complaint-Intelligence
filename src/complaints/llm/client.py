"""Provider-agnostic LLM client with budget cap, retries and cost tracking.

Defines the abstract LLMClient interface and concrete adapters.
Phase 4 implements the real adapters; FakeLLMClient is available now for tests.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """Standardised response from any LLM provider."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    raw: dict[str, Any] = field(default_factory=dict)
    cached: bool = False


class LLMClient(abc.ABC):
    """Abstract LLM client. All providers implement this interface."""

    @abc.abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a completion request and return a standardised response."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...


class FakeLLMClient(LLMClient):
    """Deterministic fake client for unit tests. No network calls."""

    def __init__(
        self,
        default_response: str = '{"issue_category": "test", "urgency": "low"}',
        latency_ms: float = 10.0,
    ) -> None:
        self._default_response = default_response
        self._latency_ms = latency_ms
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return "fake-model-v1"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Return a canned response after a simulated delay."""
        self._call_count += 1
        time.sleep(self._latency_ms / 1000)

        # Estimate token counts from message lengths
        input_text = " ".join(m.get("content", "") for m in messages)
        input_tokens = len(input_text.split())
        output_tokens = len(self._default_response.split())

        return LLMResponse(
            text=self._default_response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=self._latency_ms,
            cost_usd=0.0,
            raw={"fake": True, "call_number": self._call_count},
            cached=False,
        )

    @property
    def call_count(self) -> int:
        return self._call_count
