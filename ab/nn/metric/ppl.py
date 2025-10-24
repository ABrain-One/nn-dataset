import math
from typing import Optional

import torch
import torch.nn.functional as F

from .base import BaseMetric
from ab.nn.metric.base.utils import flatten_logits_and_labels

PPL_MIN = 1
PPL_MAX = 1_000

class Perplexity(BaseMetric):
    """
    Per-token perplexity for [B,V] or [B,S,V] outputs
    """
    def __init__(self, ignore_index: Optional[int] = None):
        super().__init__()
        self._total_tokens = None
        self._total_loss = None
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self._total_loss: float = 0.0
        self._total_tokens: int = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        outputs, targets = flatten_logits_and_labels(outputs, targets)  # [B*,V]  /  [B*]
        targets = targets.long()

        if self.ignore_index is not None:
            loss = F.cross_entropy(
                outputs,
                targets,
                reduction="sum",
                ignore_index=self.ignore_index,
            )
            valid = (targets != self.ignore_index).sum().item()
        else:
            loss = F.cross_entropy(outputs, targets, reduction="sum")
            valid = targets.numel()

        self._total_loss += loss.item()
        self._total_tokens += valid

    def result(self) -> float:
        if self._total_tokens == 0:
            return float("nan")

        # perplexity as it is
        avg_nll = self._total_loss / self._total_tokens
        ppl = math.exp(avg_nll)

        # Perplexity metric with min-max normalisation in log-scale score in [0;1],
        # where 1 - perfect model, 0 - worst-threshold model
        p_log = math.log1p(ppl)
        p_min = math.log1p(PPL_MIN)
        p_max = math.log1p(PPL_MAX)
        p_norm = (p_log - p_min) / (p_max - p_min)
        score = max(0.0, min(1.0, 1.0 - p_norm))
        return score

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor):
        self.update(outputs, targets)
        return self.result(), self._total_tokens


def compute(outputs: torch.Tensor, targets: torch.Tensor, ignore_index: Optional[int] = None):
    metric = Perplexity(ignore_index)
    metric.update(outputs, targets)
    return metric.result(), 1


def create_metric(out_shape=None, **kwargs):
    return Perplexity(**kwargs)
