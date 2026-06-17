# Breaking the Tie Open-Source Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:writing-plans before implementation. Implementation steps should use checkbox syntax for tracking.

**Goal:** Convert the experimental code for "Breaking the Tie: A Cluster-Aware Routing Framework for Large Language Models" into a clean, reproducible academic GitHub project.

**Architecture:** The repository will expose the paper method through a normalized `src/caslr` Python package, CLI scripts, YAML configs, sample data, tests, and documentation. The README will use the paper title as the project identity, while CASLR remains the method and package name.

**Tech Stack:** Python, PyTorch, Transformers, scikit-learn, pandas, NumPy, PyYAML, pytest, matplotlib, openpyxl.

---

## Context

The current workspace contains:

- `kBS_LLM_ROUTING.pdf`: 35-page paper source.
- The experiment workbook at the repository root.
- `code/`: duplicated experimental scripts with hard-coded private paths, dated folders, temporary folders, and baseline variants.

The paper describes a two-stage method:

1. Offline cluster-aware dataset construction.
2. Router training and inference using a lightweight BERT router.

The key implementation requirements are BGE/text embedding, KMeans clustering, cluster-level expert utility, masked-softmax soft labels, BERT soft-label training, inference, and evaluation.

## Recommended Approach

Use a paper-first open-source rewrite.

The public project should not expose the current script layout. Instead, preserve the algorithmic behavior in focused modules:

- `data.py`: load benchmark files and normalize router columns.
- `embeddings.py`: encode queries and manage embedding cache.
- `clustering.py`: KMeans fit/predict and cluster cache.
- `labels.py`: cluster utility, soft-label, and hard-label construction.
- `training.py`: BERT router dataset, collator, trainer, and training loop.
- `inference.py`: model loading and query-to-expert routing.
- `evaluation.py`: expert and router metrics.
- `analysis.py`: routing noise/collapse and ablation helpers.
- `plotting.py`: README/report visual assets.

The root README should follow the style of an academic paper-code release, using the full paper title, figures extracted from the paper, result tables from the workbook/paper, and clear reproduction commands.

## Alternatives Considered

### Minimal Cleanup

Keep most scripts and rename folders. This is fast but leaves hard-coded paths, repeated logic, and unclear public APIs. It is not suitable for a serious GitHub release.

### Full Framework Rewrite

Build a general-purpose LLM-routing framework. This would be cleaner long term, but it risks adding features not present in the paper and delaying release.

### Recommended: Paper-First Rewrite

Rewrite only the functionality needed to reproduce the paper method and experiments. This balances rigor, clarity, and release speed.

## Public Structure

```text
.
+-- README.md
+-- AGENT.md
+-- pyproject.toml
+-- requirements.txt
+-- configs/
+-- data/
+-- docs/
+-- scripts/
+-- src/caslr/
+-- tests/
```

## Documentation Requirements

The README must include:

- The exact paper title as the H1.
- A concise explanation of routing noise and routing collapse.
- Figure 1 and Figure 2 from the paper if readable after extraction.
- A method overview of CASLR.
- Dataset schema and sample data instructions.
- Training, evaluation, ablation, and latency commands.
- Paper result tables and citation.

Supporting docs:

- `docs/method.md`: method details aligned to the paper.
- `docs/reproduction.md`: full reproduction workflow and limitations.
- `docs/results.md`: tables and experiment summaries.

## Testing Strategy

Heavy BERT/BGE training is not required in unit tests. Tests should cover:

- Loading sample tabular data.
- Extracting `router_<expert>` columns.
- Computing cluster utility.
- Constructing masked soft labels.
- Constructing hard-label ablation labels.
- Computing expert/router accuracy.
- Ensuring configs do not require private paths for sample runs.

## Open Decisions

- License text must be chosen by the repository owner.
- Paper URL/arXiv/conference link should be added once available.
- Full raw benchmark data release status must be confirmed before public upload.
- Whether trained router checkpoints can be released must be confirmed.

## Acceptance Criteria

- No public script depends on private absolute server or user paths.
- The paper method is represented by clean modules and scripts.
- README commands run from repository root.
- README uses paper figures and tables.
- Sample workflow runs without private data.
- Tests pass for all pure logic.
