import numpy as np

from caslr.labels import (
    build_hard_labels,
    build_soft_labels,
    compute_cluster_utilities,
    masked_softmax,
)


def test_compute_cluster_utilities_counts_successes_per_cluster_and_expert():
    success = np.array(
        [
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=np.int64,
    )
    clusters = np.array([0, 0, 1, 1])

    utilities = compute_cluster_utilities(success, clusters)

    assert utilities.shape == (2, 3)
    np.testing.assert_array_equal(utilities[0], np.array([2, 1, 1]))
    np.testing.assert_array_equal(utilities[1], np.array([0, 1, 2]))


def test_masked_softmax_assigns_zero_to_failed_experts_and_normalizes_successes():
    scores = np.array([2.0, 1.0, 5.0])
    mask = np.array([1, 1, 0], dtype=bool)

    labels = masked_softmax(scores, mask)

    assert labels[2] == 0.0
    assert labels[0] > labels[1]
    np.testing.assert_allclose(labels.sum(), 1.0)


def test_build_soft_labels_uses_cluster_utility_only_for_successful_experts():
    success = np.array(
        [
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=np.int64,
    )
    clusters = np.array([0, 0, 1])

    labels = build_soft_labels(success, clusters)

    assert labels.shape == success.shape
    assert labels[0, 1] == 0.0
    assert labels[1, 2] == 0.0
    np.testing.assert_allclose(labels.sum(axis=1), np.ones(3))
    assert labels[0, 0] > labels[0, 2]


def test_build_hard_labels_selects_best_successful_cluster_expert():
    success = np.array(
        [
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=np.int64,
    )
    clusters = np.array([0, 0, 1])

    labels = build_hard_labels(success, clusters)

    np.testing.assert_array_equal(labels[0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(labels[2], np.array([0.0, 0.0, 1.0]))
