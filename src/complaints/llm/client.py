"""Unified, provider-agnostic LLM client with budget cap and caching.

Implements:
1. LLMClient base interface
2. GeminiClient concrete adapter with google-genai SDK
3. FakeLLMClient for zero-cost deterministic unit tests and dry runs
4. ReplayClient for frozen CI regression runs using SQLite cache
5. Exponential backoff retries on rate limits (429) and server overloads (503)
6. Hard budget cap enforcement per run
"""

from __future__ import annotations

import abc
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from complaints.config import get, get_env
from complaints.llm.cache import LLMCache, compute_cache_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """Standardized response schema from any LLM provider."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    raw: dict[str, Any] = field(default_factory=dict)
    cached: bool = False


class BudgetExceededError(RuntimeError):
    """Raised when cumulative run spend exceeds budget_cap_usd."""


class LLMClient(abc.ABC):
    """Abstract LLM client interface."""

    def __init__(self, budget_cap_usd: float | None = None) -> None:
        self.budget_cap_usd = budget_cap_usd or float(get("llm", "budget_cap_usd", 5.0))
        self.cumulative_cost_usd: float = 0.0
        self.cache: LLMCache = LLMCache()

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Return provider model identifier."""
        ...

    @property
    @abc.abstractmethod
    def input_price_per_million(self) -> float:
        """Cost in USD per 1M input tokens."""
        ...

    @property
    @abc.abstractmethod
    def output_price_per_million(self) -> float:
        """Cost in USD per 1M output tokens."""
        ...

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate USD cost for token usage."""
        cost_in = (input_tokens / 1_000_000.0) * self.input_price_per_million
        cost_out = (output_tokens / 1_000_000.0) * self.output_price_per_million
        return float(cost_in + cost_out)

    def _check_budget(self, additional_cost: float) -> None:
        """Enforce strict per-run expenditure ceiling."""
        if (self.cumulative_cost_usd + additional_cost) > self.budget_cap_usd:
            raise BudgetExceededError(
                f"LLM budget cap exceeded! Attempted spend (${self.cumulative_cost_usd + additional_cost:.4f}) "
                f"exceeds limit (${self.budget_cap_usd:.2f}). Aborting."
            )

    @abc.abstractmethod
    def _execute_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, int, int, dict[str, Any]]:
        """Provider-specific completion call returning (text, in_tokens, out_tokens, raw_dict)."""
        ...

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        prompt_version: str = "v1",
        use_cache: bool = True,
    ) -> LLMResponse:
        """Complete prompt with caching, retries, and budget safety."""
        cache_key = compute_cache_key(self.model_name, prompt_version, messages, temperature)

        # 1. Check Cache
        if use_cache:
            cached_item = self.cache.get(cache_key)
            if cached_item:
                return LLMResponse(
                    text=cached_item["text"],
                    input_tokens=cached_item["input_tokens"],
                    output_tokens=cached_item["output_tokens"],
                    latency_ms=cached_item["latency_ms"],
                    cost_usd=0.0,  # Cache hits cost $0.00
                    raw=cached_item["raw"],
                    cached=True,
                )

        # 2. Call Provider with Latency Timing
        t0 = time.perf_counter()
        text, in_tokens, out_tokens, raw = self._execute_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Calculate Cost & Enforce Budget
        cost = self.calculate_cost(in_tokens, out_tokens)
        self._check_budget(cost)
        self.cumulative_cost_usd += cost

        # 4. Save to Cache
        if use_cache:
            self.cache.put(cache_key, text, in_tokens, out_tokens, latency_ms, raw)

        return LLMResponse(
            text=text,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            raw=raw,
            cached=False,
        )


class FakeLLMClient(LLMClient):
    """Deterministic fake client for offline tests and dry runs."""

    def __init__(
        self,
        default_response: str | None = None,
        latency_ms: float = 5.0,
        budget_cap_usd: float = 10.0,
    ) -> None:
        super().__init__(budget_cap_usd=budget_cap_usd)
        self._default_response = default_response
        self._latency_ms = latency_ms
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "fake-model-v1"

    @property
    def input_price_per_million(self) -> float:
        return 0.0

    @property
    def output_price_per_million(self) -> float:
        return 0.0

    def _execute_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, int, int, dict[str, Any]]:
        self.call_count += 1
        time.sleep(self._latency_ms / 1000.0)

        # Detect task from user message
        content = " ".join(m.get("content", "") for m in messages)

        if self._default_response is not None:
            resp_text = self._default_response
        elif "issue_category" in content or "evidence_quote" in content:
            # Task B extraction canned valid output
            # Extract an exact quote from complaint if possible
            quote = "inaccurate charge" if "inaccurate charge" in content else "credit report"
            if "<complaint>" in content:
                snippet = content.split("<complaint>")[1].split("</complaint>")[0].strip()
                words = snippet.split()
                quote = " ".join(words[:5]) if len(words) >= 5 else "dispute"

            resp_text = json.dumps(
                {
                    "issue_category": "Reporting Inaccuracy",
                    "root_cause": "Disputed incorrect balance on consumer report",
                    "urgency": "medium",
                    "sentiment": "negative",
                    "requested_resolution": "Correction of report",
                    "evidence_quote": quote,
                }
            )
        else:
            # Task A classification canned output
            resp_text = json.dumps({"product": "Credit Reporting"})

        in_tokens = len(content.split())
        out_tokens = len(resp_text.split())
        return resp_text, in_tokens, out_tokens, {"fake": True, "call_count": self.call_count}


class GeminiClient(LLMClient):
    """Google Gemini adapter using the google-genai SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        self._api_key = api_key or get_env("GEMINI_API_KEY", required=True)
        self._model = str(
            get("llm", "providers", {}).get("gemini", {}).get("model", "gemini-2.0-flash")
        )
        self._in_price = float(
            get("llm", "providers", {}).get("gemini", {}).get("input_price_per_million", 0.075)
        )
        self._out_price = float(
            get("llm", "providers", {}).get("gemini", {}).get("output_price_per_million", 0.30)
        )

        # Lazy initialize client
        from google import genai

        self.client = genai.Client(api_key=self._api_key)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def input_price_per_million(self) -> float:
        return self._in_price

    @property
    def output_price_per_million(self) -> float:
        return self._out_price

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _execute_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, int, int, dict[str, Any]]:
        # Format messages into text contents for Gemini SDK
        prompt_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        response = self.client.models.generate_content(
            model=self._model,
            contents=prompt_text,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json" if response_format else None,
            },
        )
        text = response.text or "{}"
        in_tokens = int(
            getattr(response.usage_metadata, "prompt_token_count", len(prompt_text.split()))
        )
        out_tokens = int(
            getattr(response.usage_metadata, "candidates_token_count", len(text.split()))
        )

        return text, in_tokens, out_tokens, {"provider": "gemini", "model": self._model}


class ReplayClient(LLMClient):
    """Replay-only client serving frozen responses strictly from SQLite cache.

    Guarantees zero network calls and zero cost in CI environments.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def model_name(self) -> str:
        return "replay-client"

    @property
    def input_price_per_million(self) -> float:
        return 0.0

    @property
    def output_price_per_million(self) -> float:
        return 0.0

    def _execute_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, int, int, dict[str, Any]]:
        raise KeyError("Cache miss in ReplayClient! CI evaluation requires pre-cached responses.")


def get_llm_client() -> LLMClient:
    """Factory creating configured LLM client based on configs/llm.yaml and environment."""
    provider = str(get("llm", "active_provider", "fake")).lower()

    if provider == "gemini":
        gemini_key = get_env("GEMINI_API_KEY")
        if gemini_key and gemini_key != "your-gemini-api-key-here":
            return GeminiClient(api_key=gemini_key)
        logger.warning("GEMINI_API_KEY missing or placeholder. Falling back to FakeLLMClient.")
        return FakeLLMClient()

    if provider == "replay":
        return ReplayClient()

    return FakeLLMClient()
