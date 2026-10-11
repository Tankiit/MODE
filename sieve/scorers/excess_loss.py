"""S_E: excess loss (paper §3)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Scorer, ScoringInputs, pad_first_column


class ExcessLoss(Scorer):
    """S_E = ℓ_θ(y_t) − ℓ_ref(y_t): trained-model NLL minus frozen-reference NLL, same context."""
    name, slot, needs_reference = "excess_loss", "S_E", True

    @torch.no_grad()
    def score(self, x: ScoringInputs) -> torch.Tensor:
        if x.ref_nll is None:
            raise ValueError("excess_loss needs the reference model's NLL (ref_nll)")
        B, T, V = x.logits.shape
        nll = F.cross_entropy(x.logits[:, :-1].reshape(-1, V), x.input_ids[:, 1:].reshape(-1),
                              reduction="none").reshape(B, T - 1)
        return pad_first_column(nll - x.ref_nll[:, 1:])
