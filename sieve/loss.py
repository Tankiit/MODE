"""
Gradient-invariant selective cross-entropy.

Key property: loss normalised by B*(T-1) regardless of how many tokens
are selected. This keeps gradient magnitude identical to full CLM
training at any selection ratio α — the most common bug in token-masking
papers is to normalise by n_selected instead, which implicitly scales
the learning rate by 1/α.
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
        # Unselected positions → ignore_index (excluded from sum)
        shift_labels = shift_labels.masked_fill(~mask[:, 1:], ignore_index)

    # reduction="sum" so we control the denominator precisely
    loss = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )

    # Normalise by FULL sequence length — gradient scale invariant to α
    return loss / (B * (T - 1))
