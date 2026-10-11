"""
Selected-token cross-entropy (paper Eq. train_loss).

The loss is the mean token NLL over the retained tokens: the sum over
selected positions divided by the number of selected positions. Every
sequence retains the same number of tokens (per-sequence top-k), so this
equals the batch mean of the per-sequence losses in the paper. With
mask=None (full CLM) it is the mean over all B·(T−1) predicted tokens,
which matches the model's internal loss used on CUDA for CLM.
"""
import torch
import torch.nn.functional as F
from typing import Optional


def selective_loss(
    logits:    torch.Tensor,          # [B, T, V]
    input_ids: torch.Tensor,          # [B, T] int64
    mask:      Optional[torch.Tensor] = None,  # [B, T] bool; None = full CLM
    ignore_index: int = -100,
) -> torch.Tensor:                    # scalar
    B, T, V = logits.shape

    # Standard CLM shift: logits[t] predicts token t+1. Cast to float32:
    # the loss is computed outside autocast, logits may be bf16.
    shift_logits = logits[:, :-1].float()             # [B, T-1, V]
    shift_labels = input_ids[:, 1:].contiguous()      # [B, T-1]

    if mask is not None:
        m = mask[:, 1:]  # align with shift_labels [B, T-1]
        if not m.any():
            raise ValueError("selection mask is empty")
        shift_labels = shift_labels.masked_fill(~m, ignore_index)

    # reduction="mean" with ignore_index averages over the selected tokens
    return F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )
