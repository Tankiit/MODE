"""Fixed weights: single-scorer baselines (one scorer, weight 1) and uniform mixtures."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base import Mixer


class FixedMixer(Mixer):

    def __init__(self, k: int, weights: Optional[list] = None):
        super().__init__(k)
        w = np.full(k, 1.0 / k) if weights is None else np.asarray(weights, dtype=np.float64)
        if w.shape != (k,) or (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"fixed weights must be {k} non-negative values, got {weights}")
        self.weights = w

    def sample(self) -> torch.Tensor:
        return torch.tensor(self.weights, dtype=torch.float32)

    def expected_weights(self) -> np.ndarray:
        return self.weights.copy()
