"""Per-execution decoder context for BLIP caption metrics."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any


_ACTIVE_TOKENIZER: ContextVar[Any | None] = ContextVar(
    "blip2_caption_tokenizer", default=None
)


def active_tokenizer():
    return _ACTIVE_TOKENIZER.get()


def select_tokenizer(tokenizer) -> None:
    _ACTIVE_TOKENIZER.set(tokenizer)


def clear_tokenizer() -> None:
    _ACTIVE_TOKENIZER.set(None)
