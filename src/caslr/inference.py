from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoutingPrediction:
    expert: str
    score: float
    top_k: list[tuple[str, float]]


def load_experts(path: str | Path) -> list[str]:
    path = Path(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class BertRouter:
    def __init__(self, model_dir: str | Path, device: str = "auto"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_dir = Path(model_dir)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.experts = load_experts(self.model_dir / "experts.txt")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir)).to(device)
        self.model.eval()

    def route(self, question: str, dataset: str | None = None, top_k: int = 1) -> RoutingPrediction:
        import torch
        from caslr.data import format_router_text

        text = format_router_text(question, dataset)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits[0], dim=-1).cpu()
        k = min(top_k, len(self.experts))
        values, indices = torch.topk(probs, k=k)
        top = [(self.experts[int(idx)], float(value)) for value, idx in zip(values, indices)]
        return RoutingPrediction(expert=top[0][0], score=top[0][1], top_k=top)
