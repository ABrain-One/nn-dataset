import torch

def flatten_logits_and_labels(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reduces shapes to flat, understandable for most classification metrics / cross-entropy-loss.

    # logits: [B, V] → [B, V]
    # logits: [B, S, V] → [B*S, V]

    # labels: [B] → [B]
    # labels: [B, S] → [B*S]
    """
    if logits.dim() == 3:                # [B, S, V] → [B*S, V]
        B, S, V = logits.shape
        logits  = logits.reshape(B * S, V)

    if labels.dim() == 2:                # [B, S] → [B*S]
        labels = labels.reshape(-1)

    return logits, labels