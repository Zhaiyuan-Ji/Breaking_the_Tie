# Breaking the Tie: A Cluster-Aware Routing Framework for Large Language Models

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Method](https://img.shields.io/badge/method-CASLR-green)](#method)
[![Status](https://img.shields.io/badge/status-research--code-orange)](#reproduction)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

This repository contains the organized research code for **Breaking the Tie: A Cluster-Aware Routing Framework for Large Language Models**. The method introduced in the paper is **CASLR**: Cluster-Aware Soft-Labeling Routing.

## Overview

LLM routing selects one expert model from a candidate pool for each input query. Existing routers often reduce this to ordinary classification, but this creates a failure mode when multiple experts answer the same query correctly. The paper formalizes this as **routing noise** and shows how it can lead to **routing collapse** on unseen tasks.

CASLR resolves the tie by moving from single-query success to **cluster-level domain consensus**:

1. Encode queries into a semantic space.
2. Cluster related queries with KMeans.
3. Compute each expert's cluster-level utility.
4. Assign zero probability to experts that fail the current query.
5. Apply masked softmax over successful experts using cluster utility scores.
6. Train a lightweight BERT router with the refined soft labels.

<p align="center">
  <img src="docs/assets/paper/paper_figure1_routing_noise_collapse.png" width="86%" alt="Routing noise and routing collapse">
</p>

## Method

<p align="center">
  <img src="docs/assets/paper/paper_figure2_caslr_framework.png" width="92%" alt="Overview of the CASLR framework">
</p>

The code mirrors the paper pipeline:

- `src/caslr/data.py`: loads benchmark files with `question` and `router_<expert>` columns.
- `src/caslr/embeddings.py`: supports deterministic sample embeddings and Hugging Face/BGE encoders.
- `src/caslr/clustering.py`: KMeans clustering and cache helpers.
- `src/caslr/labels.py`: cluster utility, masked-softmax soft labels, hard-label ablation, and random-success labels.
- `src/caslr/training.py`: BERT sequence-classification router trained with soft-label cross entropy.
- `src/caslr/inference.py`: query-to-expert routing API.
- `src/caslr/evaluation.py`: expert accuracy, router accuracy, oracle/random baselines, and selection ratios.
- `src/caslr/analysis.py`: routing-noise summary and routing-collapse utility-gap estimate.

For a detailed derivation, see [docs/method.md](docs/method.md).

## Repository Structure

```text
.
+-- README.md
+-- LICENSE
+-- pyproject.toml
+-- requirements.txt
+-- configs/
+-- data/
|   +-- sample/
+-- docs/
|   +-- assets/
|   +-- method.md
|   +-- reproduction.md
|   +-- results.md
|   +-- code_inventory.md
+-- scripts/
+-- src/caslr/
+-- tests/
```

The public release surface is organized around `src/caslr/`, `scripts/`, `configs/`, `data/sample/`, and the documentation under `docs/`.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Data Format

Input data can be CSV, TSV, XLS, or XLSX. Each file must contain:

- `question`: user query.
- `dataset`: benchmark/task name. If omitted, the file stem is used.
- `router_<expert_name>`: binary correctness label for each expert.

Example:

```csv
question,dataset,router_Qwen2.5-7B-Instruct,router_Qwen2.5-Math-7B-Instruct
"Solve 2+2.",Math,1,1
"Write binary search.",MBPP,0,1
```

A synthetic sample is provided at [data/sample/sample_router_data.csv](data/sample/sample_router_data.csv). It is intended for schema validation and lightweight smoke runs.

## Quick Start

Prepare cluster-aware soft labels:

```bash
python scripts/prepare_dataset.py --config configs/default.yaml --sample
```

Evaluate base expert columns:

```bash
python scripts/evaluate_experts.py --config configs/default.yaml --sample
```

Evaluate prepared router labels or saved predictions:

```bash
python scripts/evaluate_router.py --config configs/default.yaml --sample
```

Train the BERT router:

```bash
python scripts/train_router.py --config configs/five_expert.yaml
```

Measure router latency:

```bash
python scripts/measure_latency.py --config configs/five_expert.yaml --query "Solve 2+2." --dataset Math
```

## Reproduction

Full reproduction requires the benchmark files used by the paper, candidate expert correctness columns, and local/Hugging Face access to the configured BGE and BERT models. See [docs/reproduction.md](docs/reproduction.md).

Raw benchmark data and trained checkpoints are not included. Reproduction requires users to provide benchmark files with the schema described above and local or Hugging Face access to the configured embedding and router backbones.

## Results

The paper reports that the five-expert CASLR configuration reaches **79.79%** average accuracy and that the nine-expert configuration reaches **85.00%** average accuracy. The router latency reported on HumanEval is **1.13s**, compared with thousands of seconds for candidate LLM generation time in the same setting.

### Main Routing Baselines

<p align="center">
  <img src="docs/assets/paper/paper_table1_routing_baselines.png" width="92%" alt="Comparison with routing baselines">
</p>

### Large-Scale LLM Comparison

<p align="center">
  <img src="docs/assets/paper/paper_table2_large_llms.png" width="92%" alt="Comparison with large-scale LLMs">
</p>

CASLR also visualizes how the router distributes queries to experts across tasks. These distribution figures make the routing behavior inspectable, which is useful when analyzing whether the router is relying on domain specialists rather than a single general model.

<p align="center">
  <img src="docs/assets/paper/paper_figure4_selection_distribution.png" width="86%" alt="Response distribution across tasks">
</p>

### Expert Pool Size

<p align="center">
  <img src="docs/assets/paper/paper_table4_expert_pool_size.png" width="92%" alt="Impact of the number of experts">
</p>

### Candidate Model Quality

<p align="center">
  <img src="docs/assets/paper/paper_table3_model_quality.png" width="92%" alt="Impact of adding high- and low-performance models">
</p>

### Router Latency

<p align="center">
  <img src="docs/assets/paper/paper_table5_latency.png" width="92%" alt="Router latency compared with LLM generation">
</p>

### Soft-Labeling Ablation

<p align="center">
  <img src="docs/assets/paper/paper_table6_soft_label_ablation.png" width="92%" alt="Soft-labeling ablation">
</p>

### Additional Routing Distributions

<p align="center">
  <img src="docs/assets/paper/paper_figureA5_five_expert_distribution.png" width="86%" alt="Five-expert response distribution">
</p>

<p align="center">
  <img src="docs/assets/paper/paper_figureA6_distribution.png" width="86%" alt="Response distribution">
</p>

<p align="center">
  <img src="docs/assets/paper/paper_figureA7_distribution.png" width="86%" alt="Response distribution">
</p>

<p align="center">
  <img src="docs/assets/paper/paper_figureA8_distribution.png" width="86%" alt="Response distribution">
</p>

More tables are in [docs/results.md](docs/results.md).

## Citation

```bibtex
@article{lu2026breakingtie,
  title = {Breaking the Tie: A Cluster-Aware Routing Framework for Large Language Models},
  author = {Lu, Yao and Ji, Zhaiyuan and Gao, Yaxin and Wang, Zeyu and Tang, Zhe and Wei, Jiaheng and Zhu, Zhaowei and Yu, Shanqing and Xuan, Qi},
  year = {2026},
  note = {Manuscript under review}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments

This repository follows the paper's research code and documentation structure. The README embeds paper-derived figure and table crops so that the GitHub page mirrors the academic presentation style of the paper.
