"""Selective CE with a fixed full-token denominator.

This does not preserve the full-CLM gradient magnitude: removing terms
changes the gradient. It avoids implicit selected-count renormalization.
"""
import torch
import torch.nn.functional as F
from typing import Optional


def selective_loss(
    logits:    torch.Tensor,          # [B, T, V] float32
    input_ids: torch.Tensor,          # [B, T] int64
    mask:      Optional[torch.Tensor] = None,  # [B, T] bool; None = full CLM
    ignore_index: int = -100,
) -> torch.Tensor:                    # scalar
    B, T, V = logits.shape

    # Standard CLM shift: token t predicts token t+1
    shift_logits = logits[:, :-1].contiguous()        # [B, T-1, V]
    shift_labels = input_ids[:, 1:].contiguous()      # [B, T-1]

    if mask is not None:
        m = mask[:, 1:]  # align with shift_labels [B, T-1]
        shift_labels = shift_labels.masked_fill(~m, ignore_index)

    # reduction="sum" so we control the denominator precisely
    loss = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )

    # Full-token denominator; gradient magnitude still depends on selection.
    return loss / (B * (T - 1))
