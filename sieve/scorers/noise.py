"""Control for the S_D slot: Gaussian noise (paper App. B, "Noise scorer substitution")."""
from __future__ import annotations

import torch

from .base import Scorer, ScoringInputs


class Noise(Scorer):
    """N(0, 1) values drawn anew at every scoring call, from a seeded generator."""
    name, slot = "noise", "S_D"

    def __init__(self, seed: int, device: torch.device):
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    @torch.no_grad()
    def score(self, x: ScoringInputs) -> torch.Tensor:
        return torch.randn(x.input_ids.shape, device=x.input_ids.device, generator=self.gen)
