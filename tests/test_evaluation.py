import numpy as np

from caslr.evaluation import (
    compute_expert_accuracy,
    compute_router_accuracy,
    compute_selection_ratios,
)


def test_compute_expert_accuracy_returns_per_expert_rates():
    success = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int64)
    experts = ["A", "B"]

    accuracy = compute_expert_accuracy(success, experts)

    assert accuracy == {"A": 2 / 3, "B": 2 / 3}


def test_compute_router_accuracy_counts_prediction_success():
    success = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int64)
    predicted = np.array([0, 0, 1], dtype=np.int64)

    accuracy = compute_router_accuracy(success, predicted)

    assert accuracy == 2 / 3


def test_compute_selection_ratios_uses_all_experts():
    predicted = np.array([0, 0, 2, 2], dtype=np.int64)
    experts = ["A", "B", "C"]

    ratios = compute_selection_ratios(predicted, experts)

    assert ratios == {"A": 0.5, "B": 0.0, "C": 0.5}
