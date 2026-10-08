"""LLM-based Zero-shot and Few-shot Product Classification (Task A).

Features:
- Versioned prompt loading from registry (prompts/classify_v1.txt)
- Untrusted input boundary protection with <complaint> tags
- Few-shot exemplar retrieval by embedding cosine similarity strictly from the TRAIN split
- Shared Parquet predictions schema
- MLflow experiment logging
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score

from complaints.config import get, get_project_root, setup_logging
from complaints.llm.client import LLMClient, get_llm_client
from complaints.llm.registry import format_untrusted_prompt, load_prompt
from complaints.models.base import BaseModel
from complaints.tracking.mlflow_utils import log_metrics_and_params, start_tracked_run

logger = logging.getLogger(__name__)


class FewShotRetriever:
    """Retrieves similar few-shot demonstrations strictly from the training split."""

    def __init__(self, train_df: pd.DataFrame, n_shots: int = 3) -> None:
        self.n_shots = n_shots
        self.train_df = train_df.reset_index(drop=True)
        from complaints.models.embed_lgbm import EmbeddingCache

        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        cache_dir = get_project_root() / "data" / "cache" / "embeddings"
        cache = EmbeddingCache(cache_dir)
        self.train_embeddings = cache.get_or_compute(
            self.train_df["narrative"].tolist(),
            encoder=self.encoder,
            batch_size=256,
            cache_name="train",
        )

    def retrieve_shots(self, query_text: str) -> list[dict[str, str]]:
        """Find top-k cosine similarity examples from train split."""
        q_emb = self.encoder.encode([query_text], normalize_embeddings=True)
        scores = np.dot(self.train_embeddings, q_emb.T).flatten()
        top_indices = np.argsort(scores)[-self.n_shots :][::-1]

        shots: list[dict[str, str]] = []
        for idx in top_indices:
            row = self.train_df.iloc[idx]
            # Truncate narrative snippet to keep prompt compact
            snippet = str(row["narrative"])[:200]
            shots.append({"narrative": snippet, "product": str(row["product"])})
        return shots


class LLMClassifier(BaseModel):
    """Zero-shot or Few-shot LLM product classifier."""

    def __init__(
        self,
        client: LLMClient | None = None,
        few_shot: bool = False,
        prompt_version: str = "v1",
    ) -> None:
        self.client = client or get_llm_client()
        self.few_shot = few_shot
        self.prompt_version = prompt_version
        self.prompt_template, self.version_tag = load_prompt("classify", version=prompt_version)
        self.retriever: FewShotRetriever | None = None

    @property
    def name(self) -> str:
        return (
            f"llm_fewshot_{self.client.model_name}"
            if self.few_shot
            else f"llm_zeroshot_{self.client.model_name}"
        )

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """Initialize few-shot retriever from train_df if few_shot is enabled."""
        if self.few_shot:
            # Subsample train for fast exemplar retrieval
            sample_train = train_df.sample(n=min(3000, len(train_df)), random_state=42)
            self.retriever = FewShotRetriever(sample_train, n_shots=3)

    def predict(self, df: pd.DataFrame, run_id: str = "") -> pd.DataFrame:
        """Score complaints in df and return predictions in shared schema."""
        records: list[dict[str, Any]] = []
        logger.info("[%s] Running LLM inference on %d complaints...", self.name, len(df))

        for _, row in df.reset_index(drop=True).iterrows():
            narrative = str(row.get("narrative", ""))

            # Build messages
            system_msg = (
                "You are an expert financial services complaint classifier. "
                "Classify the complaint accurately into one of the allowed products. Respond in valid JSON."
            )

            user_content = ""
            if self.few_shot and self.retriever is not None:
                shots = self.retriever.retrieve_shots(narrative)
                user_content += "Here are examples of correctly classified complaints:\n\n"
                for i, shot in enumerate(shots, 1):
                    user_content += (
                        f"Example {i}:\n"
                        f"<complaint>\n{shot['narrative']}\n</complaint>\n"
                        f'Prediction: {{"product": "{shot["product"]}"}}\n\n'
                    )
                user_content += "Now classify this complaint:\n\n"

            formatted_prompt = format_untrusted_prompt(self.prompt_template, narrative=narrative)
            user_content += formatted_prompt

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ]

            response = self.client.complete(
                messages,
                temperature=0.0,
                prompt_version=self.version_tag,
                response_format={"type": "json_object"},
            )

            # Parse prediction
            try:
                parsed = json.loads(response.text)
                pred_label = parsed.get("product", "Other")
                confidence = float(parsed.get("confidence", 0.90))
            except Exception:
                pred_label = "Other"
                confidence = 0.50

            records.append(
                self._make_prediction_row(
                    complaint_id=row["complaint_id"],
                    y_true=row.get("product", ""),
                    y_pred=pred_label,
                    confidence=confidence,
                    latency_ms=response.latency_ms,
                    cost_usd=response.cost_usd,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    run_id=run_id,
                )
            )

        return pd.DataFrame(records)


def run_llm_classification(
    few_shot: bool = False,
    sample_size: int = 200,
    client: LLMClient | None = None,
) -> Path:
    """Run LLM classification on the gold evaluation set and save predictions."""
    root = get_project_root()
    gold_dir = root / get("data", "gold_dir", "data/gold")
    pred_dir = root / "data" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    gold_path = gold_dir / "gold_eval_set.parquet"
    train_path = gold_dir / "train.parquet"
    val_path = gold_dir / "val.parquet"

    df_gold = pd.read_parquet(gold_path).head(sample_size)
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    model = LLMClassifier(client=client, few_shot=few_shot)
    model.fit(df_train, df_val)

    with start_tracked_run(run_name=model.name) as run:
        run_id = run.info.run_id
        preds_df = model.predict(df_gold, run_id=run_id)

        macro_f1 = float(
            f1_score(preds_df["y_true"], preds_df["y_pred"], average="macro", zero_division=0)
        )
        acc = float(accuracy_score(preds_df["y_true"], preds_df["y_pred"]))

        out_path = pred_dir / f"{model.name}_test.parquet"
        preds_df.to_parquet(out_path, index=False, engine="pyarrow")

        total_cost = float(preds_df["cost_usd"].sum())
        log_metrics_and_params(
            params={"model_name": model.name, "few_shot": few_shot, "n_samples": len(preds_df)},
            metrics={"test_macro_f1": macro_f1, "test_accuracy": acc, "total_cost_usd": total_cost},
            artifacts=[out_path],
        )

    logger.info("[%s] Completed: Macro-F1=%.4f, Total Cost=$%.4f", model.name, macro_f1, total_cost)
    return out_path


if __name__ == "__main__":
    setup_logging()
    run_llm_classification(few_shot=False)
    run_llm_classification(few_shot=True)
