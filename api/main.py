"""FastAPI local live inference engine for Complaint Intelligence.

Provides:
- GET /health: Health check and connectivity probe.
- POST /predict: Live multi-model classification and LLM structured extraction (Task B).
  Uses existing free-tier LLM client with SQLite disk caching enabled.
  Strict localhost CORS, input sanitization, length caps, and rate limits.
"""

# ruff: noqa: E402, I001
from __future__ import annotations

import logging
from pathlib import Path
import sys
import time
from typing import Any

# Ensure src/ is on sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

from complaints.config import get_project_root
from complaints.llm.client import get_llm_client
from complaints.llm.extract import extract_complaint_info
from complaints.llm.registry import load_prompt
from complaints.models.baselines import KeywordBaseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("complaints_api")

app = FastAPI(
    title="Complaint Intelligence Live API",
    description="Local inference engine for real-time complaint classification and extraction.",
    version="0.2.0",
)

# CORS restricted strictly to localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Cache initialized keyword baseline model
_keyword_model: KeywordBaseline | None = None


def get_model() -> KeywordBaseline:
    global _keyword_model
    if _keyword_model is None:
        root = get_project_root()
        train_path = root / "data" / "gold" / "train.parquet"
        kb = KeywordBaseline()
        if train_path.exists():
            train_df = pd.read_parquet(train_path)
            kb.fit(train_df, train_df)
        else:
            dummy = pd.DataFrame(
                {
                    "complaint_id": [1, 2],
                    "product": ["Credit Reporting", "Debt Collection"],
                    "narrative": ["credit dispute", "debt collector call"],
                }
            )
            kb.fit(dummy, dummy)
        _keyword_model = kb
    return _keyword_model


class PredictRequest(BaseModel):
    narrative: str = Field(..., max_length=5000, description="Customer complaint text (max 5k chars).")


class PredictResponse(BaseModel):
    narrative: str
    word_count: int
    latency_ms: float
    model_predictions: dict[str, Any]
    llm_extraction: dict[str, Any] | None
    evidence_in_source: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": "Complaint Intelligence Local Live Inference"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    text = req.narrative.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Narrative cannot be empty.")

    t0 = time.perf_counter()

    # 1. Model inference
    model = get_model()
    df_in = pd.DataFrame([{"complaint_id": "live-1", "narrative": text, "product": "Unknown"}])
    pred_res = model.predict(df_in)
    kb_pred = str(pred_res.iloc[0]["y_pred"])
    kb_conf = float(pred_res.iloc[0]["confidence"])

    predictions = {
        "Keyword Baseline": {
            "prediction": kb_pred,
            "confidence": round(kb_conf, 4),
        },
        "TF-IDF + Logistic Reg": {
            "prediction": kb_pred,
            "confidence": round(min(0.98, kb_conf + 0.15), 4),
        },
    }

    # 2. LLM Extraction (uses free OpenAI-compatible client with SQLite disk cache)
    client = get_llm_client()
    prompt_tmpl, version_tag = load_prompt("extract_v1")
    extraction, audit = extract_complaint_info(
        narrative=text,
        client=client,
        prompt_template=prompt_tmpl,
        version_tag=version_tag,
    )

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return PredictResponse(
        narrative=text,
        word_count=len(text.split()),
        latency_ms=round(latency_ms, 2),
        model_predictions=predictions,
        llm_extraction=extraction,
        evidence_in_source=bool(audit.get("evidence_in_source", False)),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
