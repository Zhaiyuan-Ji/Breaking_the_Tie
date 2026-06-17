# Reproduction Guide

This guide describes how to reproduce the paper pipeline with the organized code.

## Required Inputs

Full reproduction requires:

1. Benchmark files for Math, GSM-Symbolic, HumanEval, MBPP, MMLU, AIME1983-2025, and HellaSwag.
2. One binary correctness column per candidate expert, named `router_<expert_name>`.
3. Access to the embedding model, such as `BAAI/bge-large-en-v1.5`.
4. Access to a BERT base model, such as `bert-base-uncased`.
5. Hardware suitable for BERT fine-tuning.

The repository includes a synthetic sample file for code-path inspection only.

## Commands

Prepare soft labels:

```bash
python scripts/prepare_dataset.py --config configs/five_expert.yaml --label-type soft
```

Prepare hard-label ablation:

```bash
python scripts/prepare_dataset.py --config configs/five_expert.yaml --label-type hard
```

Evaluate base experts:

```bash
python scripts/evaluate_experts.py --config configs/five_expert.yaml
```

Train the router:

```bash
python scripts/train_router.py --config configs/five_expert.yaml
```

Evaluate predictions:

```bash
python scripts/evaluate_router.py --config configs/five_expert.yaml --predictions outputs/results/predictions.npy
```

Measure latency:

```bash
python scripts/measure_latency.py --config configs/five_expert.yaml --repeat 100
```

## Sample Mode

Sample mode uses `data/sample/sample_router_data.csv` and deterministic embeddings:

```bash
python scripts/prepare_dataset.py --config configs/default.yaml --sample
python scripts/evaluate_experts.py --config configs/default.yaml --sample
python scripts/evaluate_router.py --config configs/default.yaml --sample
```

## Known Release Limits

- Raw benchmark files are not included in this cleanup because their redistribution status must be confirmed.
- Trained router checkpoints are not included.
- The official paper URL is not yet included.
- License selection is pending owner confirmation.
- Several external baselines named in the paper, such as GraphRouter and RouterEval variants, may require their original repositories or unreleased preprocessing. This repository documents the interfaces and includes code paths for available local variants, but does not claim to vendor every external baseline.
- The SVG figures in `docs/assets/` are recreated schematics based on the paper draft. Replace them with final exported paper figures before public release if the journal/conference format allows redistribution.

## No Private Paths

Public configs use repository-relative paths. The old experimental code under `code/` may still contain historical machine paths, but public entrypoints in `src/`, `scripts/`, `configs/`, and `docs/` should not rely on them.
