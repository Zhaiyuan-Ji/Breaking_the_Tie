import numpy as np

from caslr.clustering import assign_kmeans_clusters, choose_cluster_count


def test_choose_cluster_count_matches_paper_implementation_heuristic():
    assert choose_cluster_count(4) == 3
    assert choose_cluster_count(100) == 10
    assert choose_cluster_count(10_000) == 20


def test_assign_kmeans_clusters_returns_one_cluster_id_per_embedding():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ],
        dtype=np.float32,
    )

    cluster_ids, model = assign_kmeans_clusters(embeddings, n_clusters=2, seed=42)

    assert cluster_ids.shape == (4,)
    assert len(set(cluster_ids.tolist())) == 2
    assert hasattr(model, "cluster_centers_")
