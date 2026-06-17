from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caslr.labels import compute_cluster_utilities


@dataclass(frozen=True)
class RoutingNoiseSummary:
    num_samples: int
    multi_success_samples: int
    unsolved_samples: int
    average_successful_experts: float


def summarize_routing_noise(success: np.ndarray) -> RoutingNoiseSummary:
    success = np.asarray(success, dtype=np.int64)
    successful_counts = success.sum(axis=1)
    return RoutingNoiseSummary(
        num_samples=int(success.shape[0]),
        multi_success_samples=int(np.sum(successful_counts > 1)),
        unsolved_samples=int(np.sum(successful_counts == 0)),
        average_successful_experts=float(successful_counts.mean()) if len(successful_counts) else 0.0,
    )


def estimate_routing_collapse_gap(
    success: np.ndarray,
    cluster_ids: np.ndarray,
    predicted_expert_ids: np.ndarray,
) -> float:
    """Estimate the utility gap between cluster-best successful experts and predictions.

    This operationalizes the paper's routing-collapse discussion for datasets where
    only binary correctness is available. For each sample, the domain utility is the
    cluster-level success count of the selected expert. The gap compares the best
    successful expert on that sample against the router's prediction.
    """

    success = np.asarray(success, dtype=np.int64)
    cluster_ids = np.asarray(cluster_ids)
    predicted_expert_ids = np.asarray(predicted_expert_ids, dtype=np.int64)
    utilities = compute_cluster_utilities(success, cluster_ids)
    clusters = {cluster: idx for idx, cluster in enumerate(sorted(np.unique(cluster_ids).tolist()))}

    gaps: list[float] = []
    for sample_id, cluster in enumerate(cluster_ids):
        row = utilities[clusters[int(cluster)]]
        successful = np.flatnonzero(success[sample_id] == 1)
        if len(successful) == 0:
            continue
        best_utility = float(np.max(row[successful]))
        predicted_utility = float(row[predicted_expert_ids[sample_id]])
        gaps.append(max(0.0, best_utility - predicted_utility))
    return float(np.mean(gaps)) if gaps else 0.0
