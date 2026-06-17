# Results

Values in this file are transcribed from the paper draft and the local experiment workbook. They are included for documentation and README presentation.

## Main Observations

- CASLR addresses routing noise by replacing hard one-hot labels with cluster-aware soft labels.
- The five-expert setup is presented as a balanced efficiency/accuracy configuration.
- Larger expert pools improve average performance until marginal gains converge.
- Soft labeling substantially outperforms the hard-label ablation.
- Router latency is reported as 1.13s in the HumanEval latency comparison.

## Expert Pool Size

| Expert pool | Math | GSM-Symbolic | AIME1983-2025 | HumanEval | MBPP | MMLU | HellaSwag | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 experts (1-3) | 77.00 | 92.10 | 26.02 | 69.70 | 61.00 | 73.76 | 65.85 | 66.49 |
| 4 experts (1-4) | 77.76 | 91.50 | 54.08 | 78.79 | 54.00 | 73.90 | 65.70 | 70.82 |
| 5 experts (1-5) | 81.44 | 93.20 | 66.33 | 93.94 | 69.00 | 82.52 | 72.10 | 79.79 |
| 6 experts (1-6) | 81.52 | 94.20 | 67.35 | 90.91 | 69.00 | 82.48 | 72.50 | 79.71 |
| 7 experts (1-7) | 82.80 | 96.00 | 65.82 | 90.91 | 64.00 | 86.88 | 79.25 | 80.81 |
| 8 experts (1-8) | 83.52 | 95.50 | 75.00 | 78.79 | 74.00 | 88.46 | 79.05 | 82.05 |
| 9 experts (1-9) | 86.80 | 96.50 | 81.63 | 87.88 | 73.00 | 89.15 | 80.05 | 85.00 |
| 10 experts (1-10) | 84.20 | 95.50 | 80.10 | 84.85 | 72.00 | 87.89 | 79.25 | 83.40 |

## Soft Label vs Hard Label

| Labeling strategy | Math | GSM-Symbolic | AIME1983-2025 | HumanEval | MBPP | MMLU | HellaSwag | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hard label | 78.32 | 92.10 | 54.08 | 87.88 | 46.00 | 70.62 | 70.10 | 71.30 |
| CASLR soft label | 81.44 | 93.20 | 66.33 | 93.94 | 69.00 | 82.52 | 72.10 | 79.79 |

## Router Computational Cost

| Model | Time cost |
|---|---:|
| DeepSeek-R1-Distill-Qwen-14B | 3325.10s |
| Llama-3.3-70B-Instruct | 8771.70s |
| Qwen3-30B-A3B | 13724.83s |
| Qwen3-30B-A3B-Instruct-2507 | 9434.59s |
| Qwen3-Coder-30B-A3B-Instruct | 10150.45s |
| Router | 1.13s |
