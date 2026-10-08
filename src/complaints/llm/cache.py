"""SQLite-backed response caching for LLM requests.

Keyed by deterministic SHA-256 hash of:
(model_name, prompt_version, messages, temperature, schema_version)

Cache hits:
- Set cached = True
- Cost = $0.00
- Ensure deterministic and free CI test execution
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from complaints.config import get, get_project_root

logger = logging.getLogger(__name__)


def compute_cache_key(
    model_name: str,
    prompt_version: str,
    messages: list[dict[str, str]],
    temperature: float,
    schema_version: str = "v1",
) -> str:
    """Compute a deterministic SHA-256 hash for an LLM query."""
    payload = {
        "model": model_name,
        "prompt_version": prompt_version,
        "messages": messages,
        "temperature": round(temperature, 4),
        "schema_version": schema_version,
    }
    dumped = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


class LLMCache:
    """Persistent SQLite cache for model responses."""

    def __init__(self, db_path: Path | None = None) -> None:
        root = get_project_root()
        cache_dir = root / get("llm", "cache", {}).get("cache_dir", "data/cache/llm")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (cache_dir / "llm_responses.db")
        self._init_db()

    def _init_db(self) -> None:
        """Create sqlite table if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    cache_key TEXT PRIMARY KEY,
                    response_text TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    raw_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        """Retrieve cached response by cache_key, or return None on cache miss."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT response_text, input_tokens, output_tokens, latency_ms, raw_json
                FROM responses WHERE cache_key = ?
                """,
                (cache_key,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "text": row[0],
                    "input_tokens": row[1],
                    "output_tokens": row[2],
                    "latency_ms": row[3],
                    "raw": json.loads(row[4]),
                    "cached": True,
                }
        return None

    def put(
        self,
        cache_key: str,
        text: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        raw: dict[str, Any] | None = None,
    ) -> None:
        """Store response into SQLite cache."""
        raw_str = json.dumps(raw or {})
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO responses
                (cache_key, response_text, input_tokens, output_tokens, latency_ms, raw_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cache_key, text, input_tokens, output_tokens, latency_ms, raw_str),
            )
            conn.commit()
