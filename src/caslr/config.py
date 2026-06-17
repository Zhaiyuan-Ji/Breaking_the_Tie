from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_TITLE = "Breaking the Tie: A Cluster-Aware Routing Framework for Large Language Models"
METHOD_NAME = "CASLR"


@dataclass(frozen=True)
class DataConfig:
    files: list[Path]
    question_column: str = "question"
    dataset_column: str = "dataset"
    router_prefix: str = "router_"
    test_size: float = 0.2


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "deterministic"
    model_name_or_path: str = "BAAI/bge-large-en-v1.5"
    batch_size: int = 32
    cache_dir: Path = Path("outputs/cache/embeddings")
    normalize: bool = True


@dataclass(frozen=True)
class ClusteringConfig:
    n_clusters: int | str = "auto"
    max_clusters: int = 20
    min_clusters: int = 3
    cache_dir: Path = Path("outputs/cache/clusters")


@dataclass(frozen=True)
class RouterConfig:
    base_model_name_or_path: str = "bert-base-uncased"
    output_dir: Path = Path("outputs/router")
    max_length: int = 512
    learning_rate: float = 4e-5
    epochs: int = 10
    per_device_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    bf16: bool = False


@dataclass(frozen=True)
class OutputConfig:
    root: Path = Path("outputs")
    prepared_dir: Path = Path("outputs/prepared")
    results_dir: Path = Path("outputs/results")


@dataclass(frozen=True)
class CASLRConfig:
    project_title: str = PROJECT_TITLE
    method_name: str = METHOD_NAME
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=lambda: DataConfig(files=[]))
    experts: list[str] = field(default_factory=list)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    @property
    def data_files(self) -> list[Path]:
        return self.data.files


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "extends" in data:
        parent = _read_yaml(Path(data.pop("extends")))
        return _merge_dicts(parent, data)
    return data


def _path_list(values: list[str | Path]) -> list[Path]:
    return [Path(v) for v in values]


def load_config(path: str | Path, sample: bool = False) -> CASLRConfig:
    raw = _read_yaml(Path(path))

    data_raw = raw.get("data", {})
    files = _path_list(data_raw.get("files", []))
    if sample:
        files = [Path("data/sample/sample_router_data.csv")]

    data = DataConfig(
        files=files,
        question_column=data_raw.get("question_column", "question"),
        dataset_column=data_raw.get("dataset_column", "dataset"),
        router_prefix=data_raw.get("router_prefix", "router_"),
        test_size=float(data_raw.get("test_size", 0.2)),
    )

    embedding_raw = raw.get("embedding", {})
    embedding = EmbeddingConfig(
        provider=embedding_raw.get("provider", "deterministic"),
        model_name_or_path=embedding_raw.get("model_name_or_path", "BAAI/bge-large-en-v1.5"),
        batch_size=int(embedding_raw.get("batch_size", 32)),
        cache_dir=Path(embedding_raw.get("cache_dir", "outputs/cache/embeddings")),
        normalize=bool(embedding_raw.get("normalize", True)),
    )

    clustering_raw = raw.get("clustering", {})
    clustering = ClusteringConfig(
        n_clusters=clustering_raw.get("n_clusters", "auto"),
        max_clusters=int(clustering_raw.get("max_clusters", 20)),
        min_clusters=int(clustering_raw.get("min_clusters", 3)),
        cache_dir=Path(clustering_raw.get("cache_dir", "outputs/cache/clusters")),
    )

    router_raw = raw.get("router", {})
    router = RouterConfig(
        base_model_name_or_path=router_raw.get("base_model_name_or_path", "bert-base-uncased"),
        output_dir=Path(router_raw.get("output_dir", "outputs/router")),
        max_length=int(router_raw.get("max_length", 512)),
        learning_rate=float(router_raw.get("learning_rate", 4e-5)),
        epochs=int(router_raw.get("epochs", 10)),
        per_device_batch_size=int(router_raw.get("per_device_batch_size", 32)),
        gradient_accumulation_steps=int(router_raw.get("gradient_accumulation_steps", 1)),
        bf16=bool(router_raw.get("bf16", False)),
    )

    outputs_raw = raw.get("outputs", {})
    outputs = OutputConfig(
        root=Path(outputs_raw.get("root", "outputs")),
        prepared_dir=Path(outputs_raw.get("prepared_dir", "outputs/prepared")),
        results_dir=Path(outputs_raw.get("results_dir", "outputs/results")),
    )

    return CASLRConfig(
        project_title=raw.get("project_title", PROJECT_TITLE),
        method_name=raw.get("method_name", METHOD_NAME),
        seed=int(raw.get("seed", 42)),
        device=raw.get("device", "auto"),
        data=data,
        experts=list(raw.get("experts", [])),
        embedding=embedding,
        clustering=clustering,
        router=router,
        outputs=outputs,
    )
