"""S_L: negative logit margin (paper §3)."""
from __future__ import annotations

import torch

from .base import Scorer, ScoringInputs, pad_first_column


class LogitMargin(Scorer):
    """S_L = −(largest − second-largest logit) of the distribution that predicts token t."""
    name, slot = "logit_margin", "S_L"

    @torch.no_grad()
    def score(self, x: ScoringInputs) -> torch.Tensor:
        top2 = x.logits[:, :-1].topk(2, dim=-1).values
        return pad_first_column(-(top2[..., 0] - top2[..., 1]))
