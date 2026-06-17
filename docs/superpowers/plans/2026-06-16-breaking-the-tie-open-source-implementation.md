# Breaking the Tie Open-Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a release-ready academic codebase for "Breaking the Tie: A Cluster-Aware Routing Framework for Large Language Models".

**Architecture:** Rewrite the experimental scripts into a `src/caslr` package with small modules for data loading, configuration, clustering, label construction, training, inference, evaluation, analysis, and plotting. Public scripts call package APIs, while documentation and README explain the paper method with local figures and result tables.

**Tech Stack:** Python, PyTorch, Transformers, scikit-learn, pandas, NumPy, PyYAML, pytest, matplotlib, openpyxl.

---

## File Map

- Create: `pyproject.toml` for package metadata and pytest config.
- Create: `requirements.txt` for installable dependencies.
- Create: `configs/default.yaml` and `configs/five_expert.yaml` for portable configs.
- Create: `data/README.md` and `data/sample/sample_router_data.csv`.
- Create: `src/caslr/__init__.py`.
- Create: `src/caslr/config.py` for YAML config loading.
- Create: `src/caslr/data.py` for table loading, router column parsing, sample normalization, and split helpers.
- Create: `src/caslr/embeddings.py` for configurable embedding and deterministic test embeddings.
- Create: `src/caslr/clustering.py` for KMeans clustering and cache IO.
- Create: `src/caslr/labels.py` for cluster utility, masked softmax labels, hard labels, and random labels.
- Create: `src/caslr/training.py` for BERT soft-label trainer and smoke-safe training entrypoint.
- Create: `src/caslr/inference.py` for router loading and top-k prediction.
- Create: `src/caslr/evaluation.py` for expert/router metrics and selection ratios.
- Create: `src/caslr/analysis.py` for routing noise/collapse analysis.
- Create: `src/caslr/plotting.py` for result and selection-ratio plots.
- Create: scripts under `scripts/` for prepare, train, evaluate, ablation, latency, and asset export.
- Create: `docs/method.md`, `docs/reproduction.md`, `docs/results.md`.
- Create: `README.md`.
- Test: `tests/test_data.py`, `tests/test_labels.py`, `tests/test_clustering.py`, `tests/test_evaluation.py`, `tests/test_config.py`.

## Task 1: Project Skeleton and Test Artifacts

**Files:**
- Create: `tests/test_data.py`
- Create: `tests/test_labels.py`
- Create: `tests/test_clustering.py`
- Create: `tests/test_evaluation.py`
- Create: `tests/test_config.py`
- Create: `data/sample/sample_router_data.csv`

- [ ] **Step 1: Write tests for expected public APIs**

Tests should import from `caslr` modules that do not exist yet. This is the required RED state.

- [ ] **Step 2: Do not execute tests in this pass**

The user explicitly requested code organization without running code. Keep the tests as static artifacts for future CI/local validation.

## Task 2: Core Pure Logic Implementation

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `configs/default.yaml`
- Create: `configs/five_expert.yaml`
- Create: `src/caslr/__init__.py`
- Create: `src/caslr/config.py`
- Create: `src/caslr/data.py`
- Create: `src/caslr/clustering.py`
- Create: `src/caslr/labels.py`
- Create: `src/caslr/evaluation.py`
- Create: `src/caslr/analysis.py`

- [ ] **Step 1: Implement minimal APIs required by the tests**

Implement portable config loading, dataset loading, router column discovery, cluster utility, masked softmax, hard labels, deterministic KMeans wrappers, expert accuracy, router accuracy, expert selection ratios, and noise/collapse summaries.

- [ ] **Step 2: Leave verification commands documented, but do not run them in this pass**

Run: `python -m pytest tests -q`

Expected future result: PASS.

## Task 3: Embedding, Training, Inference, and CLI Scripts

**Files:**
- Create: `src/caslr/embeddings.py`
- Create: `src/caslr/training.py`
- Create: `src/caslr/inference.py`
- Create: `src/caslr/plotting.py`
- Create: `scripts/prepare_dataset.py`
- Create: `scripts/train_router.py`
- Create: `scripts/evaluate_router.py`
- Create: `scripts/evaluate_experts.py`
- Create: `scripts/run_ablation_hard_label.py`
- Create: `scripts/measure_latency.py`
- Create: `scripts/export_readme_assets.py`

- [ ] **Step 1: Add deterministic sample-mode embedding**

This keeps smoke tests offline and avoids requiring BGE downloads.

- [ ] **Step 2: Add BERT training code with `--sample --max-steps 1` guardrails**

Sample mode may use a tiny local/random initialization path if no pretrained model is available. Full reproduction mode should use configured Hugging Face paths.

- [ ] **Step 3: Add CLI wrappers**

Every script must run from repository root and accept `--config`.

- [ ] **Step 4: Document sample workflows without executing them**

Run:
- `python scripts/prepare_dataset.py --config configs/default.yaml --sample`
- `python scripts/evaluate_experts.py --config configs/default.yaml --sample`
- `python scripts/evaluate_router.py --config configs/default.yaml --sample`

Expected future result: all exit 0 and write outputs under `outputs/sample/`.

## Task 4: Paper Assets and Documentation

**Files:**
- Create or update: `docs/assets/*`
- Create: `docs/method.md`
- Create: `docs/reproduction.md`
- Create: `docs/results.md`
- Create: `README.md`

- [ ] **Step 1: Extract or create paper assets**

Render/extract images for graphical abstract, routing noise/collapse, CASLR framework, and results/ablation visuals. If direct PDF extraction is not clean, create faithful schematic assets and document that they are recreated from the paper.

- [ ] **Step 2: Write paper-first README**

The H1 must be exactly `Breaking the Tie: A Cluster-Aware Routing Framework for Large Language Models`.

- [ ] **Step 3: Write supporting docs**

Explain method, reproduction limits, data format, scripts, and result tables. State unresolved release constraints for raw data, checkpoints, license, and paper URL.

## Task 5: Final Verification

**Files:**
- All public files.

- [ ] **Step 1: Document unit-test command**

Run: `python -m pytest tests -q`

Expected future result: PASS.

- [ ] **Step 2: Document sample workflow**

Run:
- `python scripts/prepare_dataset.py --config configs/default.yaml --sample`
- `python scripts/train_router.py --config configs/default.yaml --sample --max-steps 1`
- `python scripts/evaluate_router.py --config configs/default.yaml --sample`

Expected future result: all exit 0.

- [ ] **Step 3: Document private-path scan**

Run a repository-private-path scan over `README.md`, `docs`, `src`, `scripts`, `configs`, `tests`, `pyproject.toml`, and `requirements.txt`.

Expected future result: no matches in public project files.

- [ ] **Step 4: Check README assets**

Verify every `docs/assets/...` image referenced by README exists.
