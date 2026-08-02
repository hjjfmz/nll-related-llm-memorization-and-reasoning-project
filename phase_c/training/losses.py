from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels sequence dimensions must match")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction=reduction,
    )
