"""S_U: predictive entropy (paper §3)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Scorer, ScoringInputs, pad_first_column


class Entropy(Scorer):
    """S_U = −Σ p log p of the distribution that predicts token t (logits[t-1])."""
    name, slot = "entropy", "S_U"

    @torch.no_grad()
    def score(self, x: ScoringInputs) -> torch.Tensor:
        log_p = F.log_softmax(x.logits[:, :-1], dim=-1)
        p = log_p.exp()
        return pad_first_column(-torch.einsum("btv,btv->bt", p, log_p))
