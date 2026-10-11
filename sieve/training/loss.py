"""Selected-token loss (paper Eq. train_loss)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def masked_mean_nll(logits: torch.Tensor, input_ids: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mean NLL over the selected tokens; mask[t] selects the term
    (logits[t-1], input_ids[t]). mask=None averages over all T−1 terms (full CLM).
    Every sequence retains the same number of tokens, so this is also the
    mean of the per-sequence losses. Computed in float32.
    """
    B, T, V = logits.shape
    labels = input_ids[:, 1:]
    if mask is not None:
        m = mask[:, 1:]
        if not m.any():
            raise ValueError("selection mask is empty")
        labels = labels.masked_fill(~m, -100)
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, V), labels.reshape(-1),
                           ignore_index=-100)
