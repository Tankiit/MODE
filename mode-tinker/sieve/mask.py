"""Per-batch, per-sequence top-k target selection."""
from __future__ import annotations
import torch
from typing import Optional


class MaskCache:
    """Cache only within the same step; different steps contain new batches.

    rescore_interval is retained for API compatibility, but cannot permit
    positional masks to leak into unrelated batches.
    """

    def __init__(self, selection_ratio: float, rescore_interval: int):
        if not 0 < selection_ratio <= 1:
            raise ValueError("selection_ratio must be in (0, 1]")
        self.alpha = selection_ratio
        self.delta = rescore_interval
        self._mask:      Optional[torch.Tensor] = None
        self._last_step: int = -9999
        self.n_selected: int = 0

    def needs_rescore(self, step: int) -> bool:
        return self._mask is None or step != self._last_step

    def update(
        self,
        scores:    torch.Tensor,              # [B, T] float
        step:      int,
        attn_mask: Optional[torch.Tensor] = None,   # [B, T] 0/1
    ) -> torch.Tensor:                        # [B, T] bool
        B, T = scores.shape
        if T < 2:
            raise ValueError("Need at least two tokens")
        valid = torch.ones_like(scores, dtype=torch.bool)
        valid[:, 0] = False
        if attn_mask is not None:
            valid &= attn_mask.bool()
            valid[:, 1:] &= attn_mask[:, :-1].bool()
        if not torch.isfinite(scores[valid]).all():
            raise ValueError("Non-finite token scores")
        mask = torch.zeros_like(valid)
        for row in range(B):
            count = int(valid[row].sum())
            if not count:
                continue
            k = min(count, max(1, round(count * self.alpha)))
            ranked = scores[row].masked_fill(~valid[row], float("-inf"))
            # Stable tie handling makes comparisons reproducible.
            idx = torch.argsort(ranked, descending=True, stable=True)[:k]
            mask[row, idx] = True

        self._mask      = mask
        self._last_step = step
        self.n_selected = mask.sum().item()
        return mask

    @property
    def mask(self) -> Optional[torch.Tensor]:
        return self._mask
