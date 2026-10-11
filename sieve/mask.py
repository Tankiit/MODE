"""Per-sequence top-k token mask, recomputed from the current batch at every step."""
from __future__ import annotations
import torch
from typing import Optional


class MaskCache:
    """
    Builds the [B, T] boolean selection mask for the current batch.

    The mask must be recomputed at every step: batches are sampled at
    random offsets, so positions selected on one batch say nothing about
    the tokens of the next one. The last mask is kept only for logging.

    Per-sequence top-k (not global):
      Each sequence independently contributes exactly round(T·α) tokens.
      Prevents one high-loss sequence dominating the batch budget.
    """

    def __init__(self, selection_ratio: float):
        self.alpha = selection_ratio
        self._mask:      Optional[torch.Tensor] = None
        self._last_step: int = -1
        self.n_selected: int = 0

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

        # Per-sequence top-k on positions 1..T-1 only.
        # selective_loss() applies the mask as mask[:, 1:] vs shift_labels, so
        # column 0 never supervises the CLM loss — selecting only index 0 used to
        # yield all-ignore_index and NaN gradients.
        scores_inner = scores[:, 1:]                   # [B, T-1]
        Ti = scores_inner.shape[1]
        k = max(1, min(k, Ti))
        _, idx = scores_inner.topk(k, dim=-1)          # indices in [0, Ti-1)
        mask = torch.zeros(B, T, dtype=torch.bool, device=scores.device)
        mask.scatter_(1, idx + 1, True)                # map → columns 1..T-1

        self._mask      = mask
        self._last_step = step
        self.n_selected = mask.sum().item()
        return mask

    @property
    def mask(self) -> Optional[torch.Tensor]:
        return self._mask
