from __future__ import annotations

import numpy as np


def compute_expert_accuracy(success: np.ndarray, experts: list[str]) -> dict[str, float]:
    success = np.asarray(success, dtype=np.int64)
    if success.shape[1] != len(experts):
        raise ValueError("expert count must match success matrix width")
    return {expert: float(success[:, idx].mean()) for idx, expert in enumerate(experts)}


def compute_router_accuracy(success: np.ndarray, predicted_expert_ids: np.ndarray) -> float:
    success = np.asarray(success, dtype=np.int64)
    predicted_expert_ids = np.asarray(predicted_expert_ids, dtype=np.int64)
    if len(predicted_expert_ids) != success.shape[0]:
        raise ValueError("predicted_expert_ids length must match number of samples")
    correct = success[np.arange(success.shape[0]), predicted_expert_ids] == 1
    return float(correct.mean()) if len(correct) else 0.0


def compute_selection_ratios(predicted_expert_ids: np.ndarray, experts: list[str]) -> dict[str, float]:
    predicted_expert_ids = np.asarray(predicted_expert_ids, dtype=np.int64)
    total = len(predicted_expert_ids)
    if total == 0:
        return {expert: 0.0 for expert in experts}
    return {
        expert: float(np.sum(predicted_expert_ids == idx) / total)
        for idx, expert in enumerate(experts)
    }


def oracle_accuracy(success: np.ndarray) -> float:
    success = np.asarray(success, dtype=np.int64)
    return float((success.sum(axis=1) > 0).mean()) if len(success) else 0.0


def random_router_accuracy(success: np.ndarray) -> float:
    success = np.asarray(success, dtype=np.int64)
    if success.size == 0:
        return 0.0
    return float(success.mean(axis=1).mean())
