from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from caslr.clustering import assign_kmeans_clusters
from caslr.config import load_config
from caslr.data import load_many_router_tables, records_to_arrays
from caslr.embeddings import encode_texts
from caslr.labels import build_hard_labels, build_soft_labels, filter_solvable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CASLR cluster-aware labels.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample", action="store_true", help="Use synthetic sample data.")
    parser.add_argument("--label-type", choices=["soft", "hard"], default="soft")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, sample=args.sample)
    records = load_many_router_tables(
        cfg.data.files,
        question_column=cfg.data.question_column,
        dataset_column=cfg.data.dataset_column,
        router_prefix=cfg.data.router_prefix,
        experts=cfg.experts,
    )
    questions, datasets, success = records_to_arrays(records, cfg.experts)
    success, questions, datasets = filter_solvable(success, np.asarray(questions, dtype=object), np.asarray(datasets, dtype=object))
    questions = questions.tolist()
    datasets = datasets.tolist()

    cache_path = cfg.embedding.cache_dir / "embeddings.npz"
    embeddings = encode_texts(
        questions,
        provider=cfg.embedding.provider,
        model_name_or_path=cfg.embedding.model_name_or_path,
        batch_size=cfg.embedding.batch_size,
        device=cfg.device,
        normalize=cfg.embedding.normalize,
        cache_path=cache_path,
    )
    cluster_ids, _ = assign_kmeans_clusters(embeddings, n_clusters=cfg.clustering.n_clusters, seed=cfg.seed)
    labels = build_soft_labels(success, cluster_ids) if args.label_type == "soft" else build_hard_labels(success, cluster_ids)

    cfg.outputs.prepared_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cfg.outputs.prepared_dir / f"{args.label_type}_labels.npz",
        questions=np.asarray(questions, dtype=object),
        datasets=np.asarray(datasets, dtype=object),
        success=success,
        cluster_ids=cluster_ids,
        labels=labels,
        experts=np.asarray(cfg.experts, dtype=object),
    )
    pd.DataFrame({"question": questions, "dataset": datasets, "cluster_id": cluster_ids}).to_csv(
        cfg.outputs.prepared_dir / "prepared_index.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
