"""BERTopic modeling on customer complaints with precomputed embeddings and reproducibility.

Phase 8 implementation:
- BERTopic clustering using cached sentence embeddings (zero re-encoding).
- Seed stability analysis across multiple seeds (Jaccard similarity).
- Topic labeling and serialization to configs/topic_labels.json.
- Topic assignments cached to data/silver/complaint_topics.parquet.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from complaints.config import get_project_root, get_seed

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 8000
DEFAULT_NR_TOPICS = 18
DEFAULT_MIN_CLUSTER_SIZE = 25


def create_bertopic_model(
    seed: int = 42,
    nr_topics: int = DEFAULT_NR_TOPICS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> Any:
    """Create a configured BERTopic pipeline instance."""
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=5,
        ngram_range=(1, 2),
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=False,
    )


def generate_topic_labels(topic_model: Any) -> dict[int, str]:
    """Generate human-readable labels from top c-TF-IDF terms per topic."""
    topic_info = topic_model.get_topic_info()
    labels: dict[int, str] = {}
    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            labels[tid] = "Outlier / Miscellaneous"
            continue
        words = topic_model.get_topic(tid)
        if words:
            top_words = [w[0] for w in words[:3]]
            labels[tid] = f"Topic {tid}: " + " / ".join(top_words)
        else:
            labels[tid] = f"Topic {tid}"
    return labels


def check_topic_stability(
    texts: list[str],
    embeddings: np.ndarray,
    seeds: tuple[int, ...] = (42, 43, 44),
    sample_n: int = 1500,
    top_k_words: int = 10,
) -> float:
    """Evaluate topic keyword stability across multiple seeds using Jaccard similarity.

    Returns the mean pairwise Jaccard similarity across the provided seeds.
    """
    if len(texts) > sample_n:
        texts = texts[:sample_n]
        embeddings = embeddings[:sample_n]

    seed_topic_words: list[list[set[str]]] = []
    for s in seeds:
        model = create_bertopic_model(seed=s, nr_topics=10, min_cluster_size=20)
        model.fit_transform(texts, embeddings)
        topic_info = model.get_topic_info()
        valid_topics = topic_info[topic_info["Topic"] >= 0]["Topic"].tolist()
        words_sets = []
        for tid in valid_topics[:5]:
            topic_terms = model.get_topic(tid)
            if topic_terms:
                words_sets.append({w[0] for w in topic_terms[:top_k_words]})
        seed_topic_words.append(words_sets)

    jaccard_scores: list[float] = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            set_a = seed_topic_words[i]
            set_b = seed_topic_words[j]
            pair_sims: list[float] = []
            for t_a in set_a:
                best_sim = 0.0
                for t_b in set_b:
                    if not t_a or not t_b:
                        continue
                    sim = len(t_a & t_b) / len(t_a | t_b)
                    if sim > best_sim:
                        best_sim = sim
                pair_sims.append(best_sim)
            if pair_sims:
                jaccard_scores.append(float(np.mean(pair_sims)))

    return float(np.mean(jaccard_scores)) if jaccard_scores else 0.0


def run_bertopic(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int | None = None,
    nr_topics: int = DEFAULT_NR_TOPICS,
    force_retrain: bool = False,
) -> tuple[Any, pd.DataFrame, dict[int, str]]:
    """Run BERTopic modeling or load cached topic assignments.

    Returns
    -------
    (topic_model, doc_topics_df, topic_labels)
    """
    root = get_project_root()
    silver_dir = root / "data/silver"
    silver_dir.mkdir(parents=True, exist_ok=True)
    cached_topics_path = silver_dir / "complaint_topics.parquet"
    labels_path = root / "configs/topic_labels.json"
    report_labels_path = root / "reports/topic_labels.json"

    if not force_retrain and cached_topics_path.exists() and labels_path.exists():
        logger.info("Loading cached topic assignments from %s", cached_topics_path)
        doc_df = pd.read_parquet(cached_topics_path)
        with open(labels_path) as f:
            raw_labels = json.load(f)
            labels = {int(k): str(v) for k, v in raw_labels.items()}
        return None, doc_df, labels

    if seed is None:
        seed = get_seed()

    train_path = root / "data/gold/train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training data: {train_path}")

    logger.info("Loading %d complaints from %s", sample_size, train_path)
    df = pd.read_parquet(train_path).head(sample_size).copy()

    # Reuse cached embeddings to avoid costly re-encoding
    cache_file = root / "data/cache/embeddings/train_34850.npy"
    if not cache_file.exists():
        raise RuntimeError(f"Cached embeddings not found at {cache_file}")

    logger.info("Loading cached sentence embeddings from %s", cache_file)
    all_emb = np.load(cache_file)
    if len(all_emb) < sample_size:
        raise RuntimeError(
            f"Cached embeddings ({len(all_emb)}) smaller than sample_size ({sample_size})"
        )

    emb = all_emb[:sample_size]

    logger.info("Fitting BERTopic with seed=%d, nr_topics=%d", seed, nr_topics)
    model = create_bertopic_model(seed=seed, nr_topics=nr_topics)
    topics, _ = model.fit_transform(df["narrative"].tolist(), emb)

    labels = generate_topic_labels(model)

    df["topic_id"] = topics
    df["topic_label"] = df["topic_id"].map(labels).fillna("Unknown Topic")

    # Save assignments
    save_cols = [
        c
        for c in [
            "complaint_id",
            "date_received",
            "product",
            "sub_product",
            "topic_id",
            "topic_label",
            "narrative",
        ]
        if c in df.columns
    ]
    df[save_cols].to_parquet(cached_topics_path, index=False)
    logger.info("Saved complaint topic assignments to %s", cached_topics_path)

    # Save topic labels
    labels_serializable = {str(k): v for k, v in labels.items()}
    with open(labels_path, "w") as f:
        json.dump(labels_serializable, f, indent=2)

    report_labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_labels_path, "w") as f:
        json.dump(labels_serializable, f, indent=2)

    logger.info("Saved topic labels to %s", labels_path)
    return model, df[save_cols], labels


if __name__ == "__main__":
    from complaints.config import setup_logging

    setup_logging()
    run_bertopic()
