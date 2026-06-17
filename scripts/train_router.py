from __future__ import annotations

import argparse

import numpy as np

from caslr.config import load_config
from caslr.training import RouterTrainingBundle, train_bert_router


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BERT router with CASLR labels.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--prepared", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, sample=args.sample)
    prepared = args.prepared or str(cfg.outputs.prepared_dir / "soft_labels.npz")
    data = np.load(prepared, allow_pickle=True)
    experts = data["experts"].tolist()
    bundle = RouterTrainingBundle(
        questions=data["questions"].tolist(),
        datasets=data["datasets"].tolist(),
        labels=data["labels"],
        experts=experts,
    )
    train_bert_router(
        bundle,
        base_model_name_or_path=cfg.router.base_model_name_or_path,
        output_dir=cfg.router.output_dir,
        max_length=cfg.router.max_length,
        learning_rate=cfg.router.learning_rate,
        epochs=cfg.router.epochs,
        per_device_batch_size=cfg.router.per_device_batch_size,
        max_steps=args.max_steps,
        bf16=cfg.router.bf16,
    )


if __name__ == "__main__":
    main()
