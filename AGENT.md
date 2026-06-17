# Agent Guide for Open-Sourcing "Breaking the Tie"

This repository must be organized around the paper:

**Breaking the Tie: A Cluster-Aware Routing Framework for Large Language Models**

CASLR is the method name, not the project title. The README, documentation, and repository presentation must use the full paper title as the primary identity, while implementation modules may use `caslr` as the Python package name.

## Primary Goal

Turn the current experimental code under `code/` into a clean, reproducible, academic open-source project suitable for GitHub release. The final project should make the paper method easy to inspect, run, evaluate, and cite.

The work must not be a file-copy cleanup. Rewrite and normalize the code into coherent modules, preserving the paper's actual algorithmic content:

1. Embed queries into a semantic vector space.
2. Cluster query embeddings with KMeans.
3. Compute cluster-level expert utility scores.
4. Construct masked-softmax soft labels.
5. Train a lightweight BERT router with soft-label cross entropy.
6. Route unseen queries to the expert with maximum router probability.
7. Evaluate the router against expert-only results, hard-label ablations, random/oracle baselines, and related routing baselines where code is available.

## Source Material

Use these files as authoritative local sources:

- `kBS_LLM_ROUTING.pdf`: paper text, figures, tables, method definitions, and project title.
- The experiment workbook at the repository root: experiment tables and values used for README and documentation.
- `code/`: current experimental implementation, including duplicated scripts and ablation variants.

Use the AutoAnnotator README only as a style reference for academic open-source presentation:

- Paper-style title.
- Badges and links near the top.
- Figure-rich introduction.
- Method architecture image.
- Dataset/model/experiment sections.
- Clear local-running commands.
- Citation block.

Do not copy AutoAnnotator content.

## Non-Negotiable Paper Coverage

Every paper claim that is represented as implementation must have a clear code path or documented reason why it is data-only, result-only, or not released.

### Method Coverage

The final source tree must include code for:

- `Routing Noise` and `Routing Collapse` analysis utilities, at least as reproducible metrics or analysis scripts.
- Offline cluster-aware dataset construction.
- BGE or configurable embedding model encoding.
- KMeans clustering and cluster assignment caching.
- Global cluster utility calculation `Ug(M_k, C_i)`.
- Masked softmax soft-label construction.
- Naive hard-label construction for ablation.
- BERT sequence-classification router training.
- Router inference over a candidate expert pool.
- Expert accuracy and router accuracy evaluation.
- Expert selection ratio reporting.

### Experiment Coverage

The final project should include runnable scripts or documented recipes for:

- Main CASLR training and evaluation.
- Base expert evaluation.
- Soft-label vs hard-label ablation.
- Expert-pool size experiments.
- Candidate model quality experiments.
- Router latency measurement.
- Cluster visualization or expert selection visualization if the assets can be generated from available data.

If a baseline depends on unreleased external implementation or unavailable raw outputs, document that limitation explicitly in `docs/reproduction.md` and keep the README honest.

## Target Repository Structure

Use this structure unless a later implementation pass finds a concrete reason to adjust it:

```text
.
+-- README.md
+-- AGENT.md
+-- pyproject.toml
+-- requirements.txt
+-- configs/
|   +-- default.yaml
|   +-- five_expert.yaml
+-- data/
|   +-- README.md
|   +-- sample/
+-- docs/
|   +-- assets/
|   +-- method.md
|   +-- reproduction.md
|   +-- results.md
+-- scripts/
|   +-- prepare_dataset.py
|   +-- train_router.py
|   +-- evaluate_router.py
|   +-- evaluate_experts.py
|   +-- run_ablation_hard_label.py
|   +-- measure_latency.py
|   +-- export_readme_assets.py
+-- src/
|   +-- caslr/
|       +-- __init__.py
|       +-- config.py
|       +-- data.py
|       +-- embeddings.py
|       +-- clustering.py
|       +-- labels.py
|       +-- router.py
|       +-- training.py
|       +-- inference.py
|       +-- evaluation.py
|       +-- analysis.py
|       +-- plotting.py
+-- tests/
    +-- test_labels.py
    +-- test_clustering.py
    +-- test_evaluation.py
    +-- test_data.py
```

## Refactoring Rules

1. Preserve algorithmic behavior before improving interfaces.
2. Remove hard-coded private server or user paths; replace them with YAML config values and CLI flags.
3. Remove duplicated dated or temporary scripts after their logic is represented in the normalized modules.
4. Do not keep mojibake comments or emoji logs in source code.
5. Make all scripts runnable from the repository root.
6. Keep the package importable with `python -m` and console scripts.
7. Keep data formats explicit: input files must contain `question` and `router_<expert_name>` columns.
8. Write tests for pure logic first, especially label construction, utility scoring, split behavior, and metric computation.
9. Keep heavyweight model training optional in tests. Unit tests must run without downloading BERT or BGE.
10. Use small sample data under `data/sample/` for smoke tests and README commands.

## README Requirements

The final `README.md` must be academic and figure-rich. It should resemble a serious paper-code release, not a generic package README.

Required sections:

1. Title: `Breaking the Tie: A Cluster-Aware Routing Framework for Large Language Models`
2. Badges: Python, license, paper, code style or tests if available.
3. Authors and paper links.
4. News or updates.
5. Overview with a concise problem statement about routing noise and routing collapse.
6. Framework figure from the paper.
7. Method summary for CASLR.
8. Repository structure.
9. Installation.
10. Data format.
11. Quick start on sample data.
12. Full reproduction commands.
13. Results tables from the paper.
14. Ablation study summary.
15. Citation.
16. Acknowledgments.

Image requirements:

- Extract or recreate paper figures into `docs/assets/`.
- Include the graphical abstract if usable.
- Include Figure 1 for routing noise/collapse.
- Include Figure 2 for the CASLR framework.
- Include appendix model-response distribution figures if readable and relevant.
- Use Markdown image tags with relative paths that work on GitHub, for example `![CASLR framework](docs/assets/framework.png)`.

## Implementation Phases

### Phase 1: Inventory and Mapping

- Build a table mapping each current script under `code/` to its final destination or deletion reason.
- Map paper sections to implementation modules.
- Extract paper figures and result tables into `docs/assets/` and `docs/results.md`.

### Phase 2: Package Skeleton

- Create `pyproject.toml`, `requirements.txt`, `src/caslr/`, `scripts/`, `configs/`, `tests/`, and `data/sample/`.
- Add a small synthetic/sample Excel or CSV file with the same schema as the real data.
- Add configuration dataclasses and YAML loading.

### Phase 3: Core Method Rewrite

- Implement dataset loading in `src/caslr/data.py`.
- Implement embedding wrappers in `src/caslr/embeddings.py`.
- Implement KMeans utilities in `src/caslr/clustering.py`.
- Implement cluster utility, masked softmax, and hard-label ablation in `src/caslr/labels.py`.
- Add unit tests before relying on the implementation.

### Phase 4: Router Training and Inference

- Implement Hugging Face BERT router dataset, collator, soft-label trainer, training entrypoint, and save/load conventions.
- Implement inference APIs that return selected expert, scores, and optional top-k experts.
- Keep GPU, bf16, batch size, model paths, cache paths, and output paths configurable.

### Phase 5: Evaluation and Analysis

- Implement expert-only evaluation.
- Implement router evaluation.
- Implement expert selection ratio reporting.
- Implement routing collapse/noise analysis utilities.
- Implement latency measurement.
- Generate result tables in CSV/Markdown form.

### Phase 6: Documentation and Release Polish

- Write `README.md`, `docs/method.md`, `docs/reproduction.md`, and `docs/results.md`.
- Include figures and tables from the paper.
- Add citation metadata and license placeholder.
- Add smoke-test commands.
- Run formatting, unit tests, and sample workflow checks.

## Verification Checklist

Before calling the project ready for release:

- `python -m pytest` passes.
- `python scripts/prepare_dataset.py --config configs/default.yaml --sample` runs on sample data.
- `python scripts/train_router.py --config configs/default.yaml --sample --max-steps 1` runs without private paths.
- `python scripts/evaluate_router.py --config configs/default.yaml --sample` produces metrics.
- README images render from `docs/assets/`.
- README commands do not reference private absolute paths.
- Every paper table or figure shown in README has a local asset or documented source.
- No temporary directory names such as `tempory`, `2026_05_22`, `SOFT LABELS`, or numbered scratch folders remain as primary public API.
- No source file contains private machine paths.

## Naming Convention

- Project display name: `Breaking the Tie`
- Full README title: `Breaking the Tie: A Cluster-Aware Routing Framework for Large Language Models`
- Method name: `CASLR`
- Python package: `caslr`
- CLI prefix, if added: `caslr`

## Documentation Tone

Use concise academic English. Explain the method clearly enough for reproduction, but avoid marketing language. Results must match the paper or clearly state when they come from sample data.
