# Release Scope and Legacy Experiment Mapping

The original local `code/` directory contains historical experiment scripts and dated ablation snapshots. The public release surface is rewritten under `src/caslr/` and `scripts/`; scratch scripts are not part of the supported API.

## Mapping

| Legacy path pattern | Observed purpose | Public destination |
|---|---|---|
| `code/train_and_save.py`, `code/1.py`, `code/2026_05_22/train_and_save.py` | Main BERT router training with BGE embeddings, KMeans, soft labels, and hard-coded paths | `src/caslr/embeddings.py`, `src/caslr/clustering.py`, `src/caslr/labels.py`, `src/caslr/training.py`, `scripts/prepare_dataset.py`, `scripts/train_router.py` |
| `code/evaluate.py`, `code/2026_05_22/evaluate.py` | Router evaluation and expert selection ratios | `src/caslr/evaluation.py`, `scripts/evaluate_router.py` |
| `code/evaluate_experts_only.py` | Base expert accuracy table | `scripts/evaluate_experts.py` |
| `code/SOFT LABELS/*` | Ablations for random, hard-label, best-per-cluster, and collapse analysis | `src/caslr/labels.py`, `src/caslr/analysis.py`, `scripts/run_ablation_hard_label.py`, `docs/reproduction.md` |
| `code/train_and_save/Bert_cluster/*` | C-BERT style clustering baseline | Documented as related baseline; full baseline reproduction depends on original setup |
| `code/train_and_save/Bert_MLC/*` | Multi-label classification baseline | Documented as related baseline; core CASLR release focuses on paper method |
| `code/train_and_save/GM/*`, `code/train_and_save/GMM/*` | Gaussian and GMM router variants | Not part of the main CASLR method; kept as historical ablation source |
| `code/train_and_save/Graph_router/*` | GraphRouter-style experiment | External baseline; not vendored as the primary public API |
| `code/inference.py`, `code/inference/*` | Router inference experiments | `src/caslr/inference.py` |
| `code/Cluster_Visualization.py`, `code/2026_05_22/plot_routing_collapse.py` | Cluster and routing-collapse visualization | `src/caslr/plotting.py`, `docs/assets/` |
| `code/tempory/*`, dated folders, numbered folders | Scratch or dated experiment snapshots | Not used as public API |

## Release Policy

The legacy directory is kept only in the local workspace as an experimental archive. Public users should rely on the normalized modules and scripts instead.
