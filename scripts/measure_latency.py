from __future__ import annotations

import argparse
import json
import time

from caslr.config import load_config
from caslr.inference import BertRouter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure router latency.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--query", default="Solve 2+2.")
    parser.add_argument("--dataset", default="Math")
    parser.add_argument("--repeat", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, sample=args.sample)
    router = BertRouter(cfg.router.output_dir, device=cfg.device)
    start = time.perf_counter()
    for _ in range(args.repeat):
        router.route(args.query, dataset=args.dataset)
    elapsed = time.perf_counter() - start
    result = {"repeat": args.repeat, "total_seconds": elapsed, "seconds_per_query": elapsed / args.repeat}
    cfg.outputs.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.outputs.results_dir / "latency.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
