"""Word-occlusion evidence for multi-label transformer predictions.

The core scorer is model-agnostic: callers provide a batch prediction
function returning one probability vector per text.  Each word is masked once;
the probability drop for the requested category is the word's local evidence
score.  A lightweight Hugging Face adapter is provided for saved transformer
checkpoints, but heavy dependencies are imported only when that adapter is
constructed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

from .config import CATEGORIES, canonicalize_category
from .templates import DISCLAIMER

_WORD = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def word_occlusion_evidence(
    predict_batch: Callable[[Sequence[str]], np.ndarray],
    text: str,
    category: str,
    *,
    mask_token: str = "[MASK]",
    k: int = 3,
    min_drop: float = 0.0,
) -> dict:
    """Return the top non-overlapping words supporting ``category``.

    ``predict_batch`` must return an ``(n_texts, len(CATEGORIES))`` array in
    ``CATEGORIES`` order.  Evidence scores are ``p(original) - p(masked)``;
    only positive drops greater than ``min_drop`` are retained.
    """
    category = canonicalize_category(category)
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if k < 0:
        raise ValueError("k must be non-negative")

    matches = list(_WORD.finditer(text))
    base = np.asarray(predict_batch([text]), dtype=float)
    expected = (1, len(CATEGORIES))
    if base.shape != expected:
        raise ValueError(
            f"predict_batch returned shape {base.shape}; expected {expected}"
        )
    category_index = CATEGORIES.index(category)
    base_probability = float(base[0, category_index])
    if not matches or k == 0:
        return {
            "category": category,
            "base_probability": base_probability,
            "method": "word_occlusion",
            "status": "no_positive_evidence",
            "spans": [],
            "phrases": [],
            "disclaimer": DISCLAIMER,
        }

    masked_texts = [
        text[:match.start()] + mask_token + text[match.end():]
        for match in matches
    ]
    masked = np.asarray(predict_batch(masked_texts), dtype=float)
    expected_masked = (len(matches), len(CATEGORIES))
    if masked.shape != expected_masked:
        raise ValueError(
            f"predict_batch returned shape {masked.shape}; "
            f"expected {expected_masked}"
        )

    candidates = []
    for match, probability in zip(matches, masked[:, category_index]):
        drop = base_probability - float(probability)
        if drop > min_drop:
            candidates.append({
                "feature": match.group(0),
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "contribution": drop,
                "masked_probability": float(probability),
            })
    candidates.sort(key=lambda item: (-item["contribution"], item["start"]))
    items = sorted(candidates[:k], key=lambda item: item["start"])
    return {
        "category": category,
        "base_probability": base_probability,
        "method": "word_occlusion",
        "status": "ok" if items else "no_positive_evidence",
        "spans": items,
        "phrases": [item["text"] for item in items],
        "disclaimer": DISCLAIMER,
    }


def transformer_evidence_by_category(
    predict_batch: Callable[[Sequence[str]], np.ndarray],
    text: str,
    categories,
    *,
    mask_token: str = "[MASK]",
    k: int = 3,
) -> dict:
    """Compute word-occlusion evidence for every predicted category."""
    return {
        canonicalize_category(category): word_occlusion_evidence(
            predict_batch,
            text,
            category,
            mask_token=mask_token,
            k=k,
        )
        for category in categories
    }


class HuggingFaceMultiLabelPredictor:
    """Batch probability adapter for a saved Hugging Face sequence classifier."""

    def __init__(self, model_path, *, max_length=256, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.max_length = max_length
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @property
    def mask_token(self) -> str:
        return self.tokenizer.mask_token or "[MASK]"

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = self.torch.sigmoid(logits)
        return probabilities.detach().cpu().numpy()
