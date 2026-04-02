"""Per-sequence top-k mask cache. Reused for Δ steps between rescores."""
from __future__ import annotations
import torch
from typing import Optional


class MaskCache:
    """
    Stores the [B, T] boolean selection mask.
    Refreshed every Δ steps; reused otherwise.

    Per-sequence top-k (not global):
      Each sequence independently contributes exactly round(T·α) tokens.
      Prevents one high-loss sequence dominating the batch budget.
    """

    def __init__(self, selection_ratio: float, rescore_interval: int):
        self.alpha = selection_ratio
        self.delta = rescore_interval
        self._mask:      Optional[torch.Tensor] = None
        self._last_step: int = -9999
        self.n_selected: int = 0

    def needs_rescore(self, step: int) -> bool:
        return self._mask is None or (step - self._last_step) >= self.delta

    def update(
        self,
        scores:    torch.Tensor,              # [B, T] float
        step:      int,
        attn_mask: Optional[torch.Tensor] = None,   # [B, T] 0/1
    ) -> torch.Tensor:                        # [B, T] bool
        B, T  = scores.shape
        k     = max(1, round(T * self.alpha))

        # Mask padding before top-k — padding logits have garbage entropy
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        # Per-sequence top-k
        _, idx  = scores.topk(k, dim=-1)               # [B, k]
        mask    = torch.zeros(B, T, dtype=torch.bool, device=scores.device)
        mask.scatter_(1, idx, True)

        self._mask      = mask
        self._last_step = step
        self.n_selected = mask.sum().item()
        return mask

    @property
    def mask(self) -> Optional[torch.Tensor]:
        return self._mask
