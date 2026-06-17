from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_markdown_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def plot_bar_table(values: dict[str, float], path: str | Path, title: str, ylabel: str = "Accuracy") -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(values.keys()), list(values.values()), color="#365c8d")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
