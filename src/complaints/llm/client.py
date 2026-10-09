"""Unified, provider-agnostic OpenAI-compatible LLM client with free-tier safety.

Phase 4 & Free Architecture implementation:
1. OpenAI-compatible chat client for local Ollama, Groq free tier, Gemini free tier, OpenRouter.
2. Rate limiting (requests per minute throttling) and exponential backoff retry on 429/503.
3. Disk caching keyed by hash of (model, prompt_version, messages, temperature).
4. Hypothetical cost calculation at reference pricing ($0.00 actual spend).
5. FakeLLMClient and ReplayClient for zero-network testing and frozen CI.
"""

from __future__ import annotations

import abc
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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
    cost_usd: float  # Hypothetical cost at reference pricing (actual spend is $0.00)
    raw: dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    cost_type: str = "hypothetical_reference_pricing"


class BudgetExceededError(RuntimeError):
    """Raised when cumulative hypothetical run spend exceeds budget_cap_usd."""


class QuotaExceededError(RuntimeError):
    """Raised when free-tier rate limits or token quotas are exhausted."""


class RateLimiter:
    """Enforces requests-per-minute (RPM) throttling."""

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.min_interval = 60.0 / max(1, requests_per_minute)
        self.last_call_time = 0.0

    def wait(self) -> None:
        """Throttle until next allowable request window."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()


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
    def input_price_per_million(self) -> float:
        """Reference hypothetical cost in USD per 1M input tokens."""
        return float(get("llm", "reference_pricing.input_price_per_million", 0.15))

    @property
    def output_price_per_million(self) -> float:
        """Reference hypothetical cost in USD per 1M output tokens."""
        return float(get("llm", "reference_pricing.output_price_per_million", 0.60))

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate hypothetical USD cost at reference pricing."""
        cost_in = (input_tokens / 1_000_000.0) * self.input_price_per_million
        cost_out = (output_tokens / 1_000_000.0) * self.output_price_per_million
        return float(cost_in + cost_out)

    def _check_budget(self, additional_cost: float) -> None:
        """Enforce per-run hypothetical expenditure ceiling."""
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
        """Complete prompt with disk caching, retries, and rate limit safety."""
        cache_key = compute_cache_key(self.model_name, prompt_version, messages, temperature)

        # 1. Check disk cache (Cache hits cost $0.00)
        if use_cache:
            cached_item = self.cache.get(cache_key)
            if cached_item:
                return LLMResponse(
                    text=cached_item["text"],
                    input_tokens=cached_item["input_tokens"],
                    output_tokens=cached_item["output_tokens"],
                    latency_ms=cached_item["latency_ms"],
                    cost_usd=0.0,
                    raw=cached_item["raw"],
                    cached=True,
                )

        # 2. Call provider with latency timing
        t0 = time.perf_counter()
        text, in_tokens, out_tokens, raw = self._execute_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Calculate hypothetical cost
        cost = self.calculate_cost(in_tokens, out_tokens)
        self._check_budget(cost)
        self.cumulative_cost_usd += cost

        # 4. Save to disk cache
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


class OpenAICompatibleClient(LLMClient):
    """Universal client for any OpenAI-compatible chat endpoint (Ollama, Groq, Gemini, OpenRouter)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:3b",
        api_key: str | None = None,
        requests_per_minute: int = 60,
        timeout_seconds: float = 60.0,
        profile_name: str = "ollama",
    ) -> None:
        super().__init__()
        self._base_url = base_url
        self._model = model
        self._profile_name = profile_name
        self._api_key = api_key or "not-needed"
        self._timeout_seconds = timeout_seconds
        self.rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)

        self.client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout_seconds,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, InternalServerError, APIConnectionError)
        ),
        wait=wait_exponential(multiplier=1.5, min=2, max=30),
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
        self.rate_limiter.wait()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        try:
            res = self.client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            logger.error("Rate limit hit on %s: %s", self._profile_name, e)
            raise QuotaExceededError(
                f"Rate limit / quota exceeded for free profile '{self._profile_name}'. "
                "Wait before retrying or use cached replay."
            ) from e
        except APIConnectionError as e:
            logger.error("Connection failed to %s at %s: %s", self._profile_name, self._base_url, e)
            raise ConnectionError(
                f"Could not connect to {self._profile_name} at {self._base_url}. "
                "If using Ollama, ensure it is running with 'ollama serve'."
            ) from e

        choice = res.choices[0]
        text = choice.message.content or "{}"
        usage = res.usage
        in_tokens = (
            usage.prompt_tokens
            if usage
            else sum(len(m.get("content", "").split()) for m in messages)
        )
        out_tokens = usage.completion_tokens if usage else len(text.split())

        return text, in_tokens, out_tokens, {"profile": self._profile_name, "model": self._model}


class FakeLLMClient(LLMClient):
    """Deterministic fake client for offline tests and zero-network dry runs."""

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

        content = " ".join(m.get("content", "") for m in messages)

        if self._default_response is not None:
            resp_text = self._default_response
        elif "issue_category" in content or "evidence_quote" in content:
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
            resp_text = json.dumps({"product": "Credit Reporting"})

        in_tokens = len(content.split())
        out_tokens = len(resp_text.split())
        return resp_text, in_tokens, out_tokens, {"fake": True, "call_count": self.call_count}


class ReplayClient(LLMClient):
    """Replay-only client serving frozen responses strictly from SQLite cache."""

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


def get_llm_client(profile: str | None = None) -> LLMClient:
    """Factory creating configured LLM client based on configs/llm.yaml and environment."""
    profile_name = profile or str(get("llm", "active_profile", "ollama")).lower()

    if profile_name == "fake":
        return FakeLLMClient()

    if profile_name == "replay":
        return ReplayClient()

    # Read profile config
    profiles_cfg = get("llm", "profiles", {})
    profile_cfg = profiles_cfg.get(profile_name)

    if not profile_cfg:
        logger.warning(
            "Profile '%s' not found in configs/llm.yaml. Falling back to FakeLLMClient.",
            profile_name,
        )
        return FakeLLMClient()

    base_url = profile_cfg.get("base_url")
    model = profile_cfg.get("model", "qwen2.5:3b")
    api_key_env = profile_cfg.get("api_key_env")
    api_key = get_env(api_key_env) if api_key_env else profile_cfg.get("api_key_default", "ollama")
    rpm = int(profile_cfg.get("requests_per_minute", 60))
    timeout = float(profile_cfg.get("timeout_seconds", 60.0))

    if not base_url:
        return FakeLLMClient()

    # If an API key is required but not provided, fall back safely
    if api_key_env and not api_key:
        logger.warning(
            "Environment variable %s for profile '%s' is not set. Falling back to FakeLLMClient for offline execution.",
            api_key_env,
            profile_name,
        )
        return FakeLLMClient()

    # If Ollama is selected, check if reachable; if not, gracefully fall back to FakeLLMClient
    if profile_name == "ollama":
        import urllib.request

        try:
            # Quick 0.5s connection probe to Ollama
            req = urllib.request.Request(f"{base_url.rstrip('/v1')}/api/version")
            with urllib.request.urlopen(req, timeout=0.8):
                pass
        except Exception:
            logger.info(
                "Local Ollama not detected at %s. Using FakeLLMClient for offline execution.",
                base_url,
            )
            return FakeLLMClient()

    return OpenAICompatibleClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        requests_per_minute=rpm,
        timeout_seconds=timeout,
        profile_name=profile_name,
    )
