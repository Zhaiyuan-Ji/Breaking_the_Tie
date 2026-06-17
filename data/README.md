# Data Format

The public scripts expect CSV or Excel files with one row per query.

Required columns:

- `question`: input query.
- `dataset`: benchmark or task name. If absent, the file stem is used.
- `router_<expert_name>`: binary correctness for each candidate expert.

Example:

```csv
question,dataset,router_Qwen2.5-7B-Instruct,router_Qwen2.5-Math-7B-Instruct
"Solve 2+2.",Math,1,1
"Write binary search.",MBPP,0,1
```

The sample file at `data/sample/sample_router_data.csv` is synthetic and intended only for schema validation and lightweight smoke tests. It is not a replacement for the paper benchmark data.
