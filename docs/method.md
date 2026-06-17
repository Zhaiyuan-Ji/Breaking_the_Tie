# Method

This document maps the paper method to the organized code.

## Problem Setting

Given a model zoo `P = {M1, M2, ..., MN}` and an input query `x`, LLM routing chooses the expert that maximizes expected utility. The paper observes that standard hard-label routing fails when multiple candidate experts answer the same query correctly.

## Routing Noise

For a query `x_i`, routing noise is the set of models that match the true domain expert on the single query but have lower intrinsic ability in the broader domain. In code, this is approximated from binary correctness columns by counting how many experts are successful for each query:

- `src/caslr/analysis.py::summarize_routing_noise`
- `multi_success_samples` counts samples where multiple experts are correct.

## Routing Collapse

The paper defines routing collapse as performance degradation caused by selecting routing-noise experts instead of the true domain expert. With only binary correctness columns available in released data, `src/caslr/analysis.py::estimate_routing_collapse_gap` approximates this as the cluster utility gap between:

- the best successful expert for that query, and
- the router-selected expert.

## Cluster-Aware Dataset Construction

The implementation follows Section 4.1:

1. Load query/expert correctness rows with `caslr.data`.
2. Encode queries with `caslr.embeddings`.
3. Cluster embeddings with KMeans using `caslr.clustering`.
4. Compute global cluster utility:

```text
Ug(M_k, C_i) = sum_{x_j in C_i} v_{j,k}
```

This is implemented by `caslr.labels.compute_cluster_utilities`.

## Masked Softmax Labels

For each query, failed experts receive zero probability. Successful experts receive probability proportional to their cluster utility:

```text
y_{i,k} = exp(Ug(M_k, C_i)) / sum exp(Ug(M_m, C_i))
```

where the denominator only includes experts that answered query `x_i` correctly.

Implementation:

- `caslr.labels.masked_softmax`
- `caslr.labels.build_soft_labels`

The hard-label ablation is implemented by `caslr.labels.build_hard_labels`, which selects the successful expert with the largest cluster utility.

## Router Training

The BERT router is implemented in `caslr.training`. It uses soft-label cross entropy:

```text
loss = mean(sum(-y * log_softmax(logits)))
```

The public entrypoint is:

```bash
python scripts/train_router.py --config configs/five_expert.yaml
```

## Inference

The inference phase follows Section 4.2: the query is sent to the trained router, the router outputs expert probabilities, and the system selects the maximum-probability expert.

Implementation:

- `caslr.inference.BertRouter`
- `scripts/measure_latency.py`
