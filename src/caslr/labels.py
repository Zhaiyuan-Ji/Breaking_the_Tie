from __future__ import annotations

import numpy as np


def compute_cluster_utilities(success: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    success = np.asarray(success, dtype=np.int64)
    cluster_ids = np.asarray(cluster_ids)
    if success.ndim != 2:
        raise ValueError("success must have shape [num_samples, num_experts]")
    if len(cluster_ids) != success.shape[0]:
        raise ValueError("cluster_ids length must match number of samples")

    unique_clusters = sorted(np.unique(cluster_ids).tolist())
    cluster_to_row = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
    utilities = np.zeros((len(unique_clusters), success.shape[1]), dtype=np.int64)
    for sample_id, cluster in enumerate(cluster_ids):
        utilities[cluster_to_row[cluster]] += success[sample_id]
    return utilities


def masked_softmax(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if scores.shape != mask.shape:
        raise ValueError("scores and mask must have the same shape")
    out = np.zeros_like(scores, dtype=np.float64)
    if not np.any(mask):
        return out
    masked_scores = scores[mask]
    shifted = masked_scores - np.max(masked_scores)
    probs = np.exp(shifted)
    probs = probs / probs.sum()
    out[mask] = probs
    return out.astype(np.float32)


def _cluster_index(cluster_ids: np.ndarray) -> dict[int, int]:
    return {cluster: idx for idx, cluster in enumerate(sorted(np.unique(cluster_ids).tolist()))}


def build_soft_labels(success: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    success = np.asarray(success, dtype=np.int64)
    cluster_ids = np.asarray(cluster_ids)
    utilities = compute_cluster_utilities(success, cluster_ids)
    cluster_to_row = _cluster_index(cluster_ids)
    labels = np.zeros_like(success, dtype=np.float32)
    for sample_id, cluster in enumerate(cluster_ids):
        mask = success[sample_id].astype(bool)
        scores = utilities[cluster_to_row[int(cluster)]].astype(np.float64)
        labels[sample_id] = masked_softmax(scores, mask)
    return labels


def build_hard_labels(success: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    success = np.asarray(success, dtype=np.int64)
    cluster_ids = np.asarray(cluster_ids)
    utilities = compute_cluster_utilities(success, cluster_ids)
    cluster_to_row = _cluster_index(cluster_ids)
    labels = np.zeros_like(success, dtype=np.float32)
    for sample_id, cluster in enumerate(cluster_ids):
        successful = np.flatnonzero(success[sample_id] == 1)
        if len(successful) == 0:
            continue
        scores = utilities[cluster_to_row[int(cluster)], successful]
        chosen = successful[int(np.argmax(scores))]
        labels[sample_id, chosen] = 1.0
    return labels


def build_random_success_labels(success: np.ndarray, seed: int = 42) -> np.ndarray:
    success = np.asarray(success, dtype=np.int64)
    rng = np.random.default_rng(seed)
    labels = np.zeros_like(success, dtype=np.float32)
    for sample_id, row in enumerate(success):
        successful = np.flatnonzero(row == 1)
        if len(successful) == 0:
            continue
        chosen = int(rng.choice(successful))
        labels[sample_id, chosen] = 1.0
    return labels


def filter_solvable(success: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    success = np.asarray(success)
    keep = success.sum(axis=1) > 0
    return (success[keep],) + tuple(np.asarray(arr)[keep] for arr in arrays)
