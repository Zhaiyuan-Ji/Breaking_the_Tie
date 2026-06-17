from __future__ import annotations

import argparse
import json

import numpy as np

from caslr.config import load_config
from caslr.evaluation import compute_router_accuracy, compute_selection_ratios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate router predictions against correctness columns.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--prepared", default=None)
    parser.add_argument("--predictions", default=None, help="Optional .npy file of predicted expert indices.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, sample=args.sample)
    prepared = args.prepared or str(cfg.outputs.prepared_dir / "soft_labels.npz")
    data = np.load(prepared, allow_pickle=True)
    success = data["success"]
    experts = data["experts"].tolist()
    if args.predictions:
        predicted = np.load(args.predictions)
    else:
        predicted = np.argmax(data["labels"], axis=1)
    results = {
        "router_accuracy": compute_router_accuracy(success, predicted),
        "selection_ratios": compute_selection_ratios(predicted, experts),
    }
    cfg.outputs.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.outputs.results_dir / "router_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
