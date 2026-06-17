from __future__ import annotations

import argparse
import json

from caslr.config import load_config
from caslr.data import load_many_router_tables, records_to_arrays
from caslr.evaluation import compute_expert_accuracy, oracle_accuracy, random_router_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base experts from router correctness columns.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, sample=args.sample)
    records = load_many_router_tables(cfg.data.files, experts=cfg.experts)
    _, _, success = records_to_arrays(records, cfg.experts)
    results = {
        "expert_accuracy": compute_expert_accuracy(success, cfg.experts),
        "oracle_accuracy": oracle_accuracy(success),
        "random_router_expected_accuracy": random_router_accuracy(success),
    }
    cfg.outputs.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.outputs.results_dir / "expert_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
