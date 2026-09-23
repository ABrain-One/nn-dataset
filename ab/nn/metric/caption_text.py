"""Shared decoding helpers for tensor-based caption metrics."""

from __future__ import annotations

from typing import Iterable

import torch
from nltk.tokenize import TreebankWordTokenizer

from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB
from ab.nn.util.captioning.blip2.context import active_tokenizer

_WORDS = TreebankWordTokenizer()


def _valid_ids(values: Iterable[int]) -> list[int]:
    return [int(value) for value in values if int(value) >= 0]


def decode_ids(values: Iterable[int]) -> str:
    ids = _valid_ids(values)
    tokenizer = active_tokenizer()
    if tokenizer is not None:
        return " ".join(
            tokenizer.decode(ids, skip_special_tokens=True).split()
        ).strip()
    idx2word = GLOBAL_CAPTION_VOCAB.get("idx2word")
    if idx2word is not None:
        special = {"<PAD>", "<SOS>", "<BOS>", "<EOS>", "<UNK>"}
        return " ".join(
            word for value in ids
            if (word := idx2word.get(value, "")) and word not in special
        ).strip()
    raise RuntimeError("No caption decoder is configured for the active dataset.")


def words(values: Iterable[int]) -> list[str]:
    text = decode_ids(values).lower()
    return _WORDS.tokenize(text) if text else []


def prediction_ids(predictions: torch.Tensor) -> list[list[int]]:
    if predictions.ndim == 3:
        predictions = predictions.argmax(dim=-1)
    if predictions.ndim != 2:
        raise ValueError(f"Expected prediction shape (B,T) or (B,T,V), got {tuple(predictions.shape)}")
    return predictions.detach().cpu().tolist()


def reference_ids(labels: torch.Tensor) -> list[list[list[int]]]:
    if labels.ndim == 2:
        labels = labels.unsqueeze(1)
    if labels.ndim != 3:
        raise ValueError(f"Expected label shape (B,T) or (B,R,T), got {tuple(labels.shape)}")
    result = []
    for sample in labels.detach().cpu().tolist():
        references = [reference for reference in sample if _valid_ids(reference)]
        result.append(references)
    return result


def decoded_batch(predictions: torch.Tensor, labels: torch.Tensor):
    hypotheses = [words(value) for value in prediction_ids(predictions)]
    references = []
    for sample in reference_ids(labels):
        decoded = []
        for value in sample:
            tokens = words(value)
            if tokens:
                decoded.append(tokens)
        references.append(decoded)
    if len(hypotheses) != len(references):
        raise ValueError("Caption prediction and reference batch sizes differ.")
    if any(not sample for sample in references):
        raise ValueError("Every evaluated image must have a nonempty reference caption.")
    return hypotheses, references
