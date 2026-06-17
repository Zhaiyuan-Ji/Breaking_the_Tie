from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


def choose_cluster_count(num_samples: int, min_clusters: int = 3, max_clusters: int = 20) -> int:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    return min(max_clusters, max(min_clusters, int(np.sqrt(num_samples))))


def assign_kmeans_clusters(
    embeddings: np.ndarray,
    n_clusters: int | str = "auto",
    seed: int = 42,
    n_init: int = 15,
) -> tuple[np.ndarray, KMeans]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [num_samples, dim]")
    if n_clusters == "auto":
        k = choose_cluster_count(len(embeddings))
    else:
        k = int(n_clusters)
    k = min(k, len(embeddings))
    model = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    cluster_ids = model.fit_predict(embeddings)
    return cluster_ids.astype(np.int64), model


def save_cluster_cache(path: str | Path, questions: list[str], cluster_ids: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, questions=np.asarray(questions, dtype=object), cluster_ids=np.asarray(cluster_ids))


def load_cluster_cache(path: str | Path, questions: list[str]) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    cache = np.load(path, allow_pickle=True)
    cached_questions = cache["questions"].tolist()
    if cached_questions != list(questions):
        return None
    return cache["cluster_ids"]
