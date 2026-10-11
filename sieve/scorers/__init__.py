"""Scorer registry: config names → scorer classes."""
from __future__ import annotations

import torch

from .base import Scorer, ScoringInputs
from .bigram import Bigram
from .entropy import Entropy
from .excess_loss import ExcessLoss
from .frequency import Frequency
from .logit_margin import LogitMargin
from .noise import Noise

SCORERS = {cls.name: cls for cls in (ExcessLoss, Entropy, LogitMargin, Frequency, Bigram, Noise)}


def build_scorer(name: str, data_dir: str, seed: int, device: torch.device) -> Scorer:
    if name not in SCORERS:
        raise ValueError(f"unknown scorer {name!r}; known: {sorted(SCORERS)}")
    cls = SCORERS[name]
    if cls in (Frequency, Bigram):
        return cls(data_dir, device)
    if cls is Noise:
        return cls(seed, device)
    return cls()


__all__ = ["SCORERS", "Scorer", "ScoringInputs", "build_scorer"]
