from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def deterministic_text_embeddings(texts: list[str], dim: int = 64, normalize: bool = True) -> np.ndarray:
    """Create deterministic offline embeddings for sample workflows and tests."""

    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for row_id, text in enumerate(texts):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = np.frombuffer((digest * ((dim // len(digest)) + 1))[:dim], dtype=np.uint8)
        vectors[row_id] = (values.astype(np.float32) / 127.5) - 1.0
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)
    return vectors


def encode_with_transformers(
    texts: list[str],
    model_name_or_path: str,
    batch_size: int = 32,
    device: str = "auto",
    normalize: bool = True,
) -> np.ndarray:
    """Encode texts with a Hugging Face encoder using mean pooling.

    This is intended for BGE-style encoders used by the paper. Imports are local
    so that documentation and pure-logic modules remain importable without loading
    heavyweight ML dependencies.
    """

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    model.eval()

    all_embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            token_embeddings = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
        all_embeddings.append(pooled.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def load_embedding_cache(path: str | Path, texts: list[str]) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    cache = np.load(path, allow_pickle=True)
    if cache["texts"].tolist() != list(texts):
        return None
    return cache["embeddings"]


def save_embedding_cache(path: str | Path, texts: list[str], embeddings: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, texts=np.asarray(texts, dtype=object), embeddings=np.asarray(embeddings, dtype=np.float32))


def encode_texts(
    texts: list[str],
    provider: str = "deterministic",
    model_name_or_path: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 32,
    device: str = "auto",
    normalize: bool = True,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    if cache_path is not None:
        cached = load_embedding_cache(cache_path, texts)
        if cached is not None:
            return cached

    if provider == "deterministic":
        embeddings = deterministic_text_embeddings(texts, normalize=normalize)
    elif provider in {"transformers", "bge"}:
        embeddings = encode_with_transformers(
            texts,
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            device=device,
            normalize=normalize,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    if cache_path is not None:
        save_embedding_cache(cache_path, texts, embeddings)
    return embeddings
