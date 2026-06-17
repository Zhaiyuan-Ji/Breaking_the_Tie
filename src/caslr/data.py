from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RouterRecord:
    question: str
    dataset: str
    success: dict[str, int]


def extract_router_columns(columns: Iterable[str], prefix: str = "router_") -> list[str]:
    return [col[len(prefix) :] for col in columns if str(col).startswith(prefix)]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported data file extension: {path.suffix}")


def load_router_table(
    path: str | Path,
    question_column: str = "question",
    dataset_column: str = "dataset",
    router_prefix: str = "router_",
    experts: list[str] | None = None,
) -> list[RouterRecord]:
    path = Path(path)
    df = read_table(path).reset_index(drop=True)
    if question_column not in df.columns:
        raise ValueError(f"Missing required question column: {question_column}")

    discovered = extract_router_columns(df.columns, router_prefix)
    selected_experts = experts or discovered
    if not selected_experts:
        raise ValueError(f"No router columns found with prefix {router_prefix!r}")

    records: list[RouterRecord] = []
    for _, row in df.iterrows():
        question = str(row.get(question_column, "")).strip()
        if not question or question.lower() == "nan":
            continue
        dataset = str(row.get(dataset_column, path.stem)).strip()
        if not dataset or dataset.lower() == "nan":
            dataset = path.stem

        success: dict[str, int] = {}
        for expert in selected_experts:
            col = f"{router_prefix}{expert}"
            if col in df.columns:
                success[expert] = int(row.get(col, 0) == 1)
        records.append(RouterRecord(question=question, dataset=dataset, success=success))
    return records


def load_many_router_tables(
    paths: Iterable[str | Path],
    question_column: str = "question",
    dataset_column: str = "dataset",
    router_prefix: str = "router_",
    experts: list[str] | None = None,
) -> list[RouterRecord]:
    records: list[RouterRecord] = []
    for path in paths:
        records.extend(
            load_router_table(
                path,
                question_column=question_column,
                dataset_column=dataset_column,
                router_prefix=router_prefix,
                experts=experts,
            )
        )
    return records


def records_to_arrays(records: list[RouterRecord], experts: list[str]) -> tuple[list[str], list[str], np.ndarray]:
    questions = [record.question for record in records]
    datasets = [record.dataset for record in records]
    success = np.zeros((len(records), len(experts)), dtype=np.int64)
    for row_id, record in enumerate(records):
        for col_id, expert in enumerate(experts):
            success[row_id, col_id] = int(record.success.get(expert, 0))
    return questions, datasets, success


def format_router_text(question: str, dataset: str | None = None) -> str:
    if dataset:
        return f"[{dataset}] Question: {question}"
    return question
