from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from caslr.data import format_router_text


@dataclass(frozen=True)
class RouterTrainingBundle:
    questions: list[str]
    datasets: list[str]
    labels: np.ndarray
    experts: list[str]


class SoftLabelDataset:
    """A lightweight adapter used by the Hugging Face training entrypoint."""

    def __init__(self, questions: list[str], datasets: list[str], labels: np.ndarray, tokenizer: Any, max_length: int):
        self.questions = questions
        self.datasets = datasets
        self.labels = np.asarray(labels, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = format_router_text(self.questions[idx], self.datasets[idx])
        encoded = self.tokenizer(text, truncation=True, padding=False, max_length=self.max_length)
        encoded["labels"] = self.labels[idx]
        return encoded


def make_collator(tokenizer: Any, max_length: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        labels = np.stack([item.pop("labels") for item in batch]).astype(np.float32)
        inputs = tokenizer.pad(batch, padding=True, max_length=max_length, return_tensors="pt")
        import torch

        inputs["labels"] = torch.tensor(labels, dtype=torch.float32)
        return inputs

    return collate


def train_bert_router(
    bundle: RouterTrainingBundle,
    base_model_name_or_path: str,
    output_dir: str | Path,
    max_length: int = 512,
    learning_rate: float = 4e-5,
    epochs: int = 10,
    per_device_batch_size: int = 32,
    max_steps: int | None = None,
    bf16: bool = False,
) -> Path:
    """Train a BERT sequence classifier with soft-label cross entropy."""

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    class SoftLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = torch.sum(-labels * F.log_softmax(logits, dim=-1), dim=-1).mean()
            return (loss, outputs) if return_outputs else loss

    output_dir = Path(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name_or_path,
        num_labels=len(bundle.experts),
    )
    dataset = SoftLabelDataset(bundle.questions, bundle.datasets, bundle.labels, tokenizer, max_length)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps if max_steps is not None else -1,
        save_strategy="epoch",
        report_to="none",
        bf16=bf16,
    )
    trainer = SoftLabelTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=make_collator(tokenizer, max_length),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "experts.txt").write_text("\n".join(bundle.experts) + "\n", encoding="utf-8")
    return output_dir
