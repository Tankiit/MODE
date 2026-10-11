"""
Dirichlet mixture (paper §3, "Adaptive mixture over scorers").

Pseudo-counts α ∈ R^K_{>0}, initialised to 1_K. `sample()` draws
w ~ Dir(α) and makes it the active mixture; `update(r)` with the clipped
validation reward r ≥ 0 applies

    α ← α + r · w                    (DirichletMixer, paper Eq. update)
    α ← α + r · w / (σ̂ + ε)          (VNDirichletMixer, paper Eq. vnts)

where σ̂_k is the standard deviation of r_j · w_j^(k) over all past
validation events (σ̂ = 1 until `sigma_warmup` rewards have been seen; the
paper does not specify the cold start). A heuristic, not exact Bayesian
inference.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base import Mixer


class DirichletMixer(Mixer):
    learns = True

    def __init__(self, k: int, seed: int):
        super().__init__(k)
        self.alpha = np.ones(k, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.active: Optional[np.ndarray] = None

    def sample(self) -> torch.Tensor:
        self.active = self.rng.dirichlet(self.alpha)
        return torch.tensor(self.active, dtype=torch.float32)

    def update(self, reward: float) -> None:
        if reward < 0:
            raise ValueError(f"reward must be clipped to >= 0, got {reward}")
        if self.active is not None:
            self.alpha += self._increment(reward, self.active)

    def _increment(self, reward: float, w: np.ndarray) -> np.ndarray:
        return reward * w

    def expected_weights(self) -> np.ndarray:
        return self.alpha / self.alpha.sum()

    def state(self) -> dict:
        return {"alpha": self.alpha.tolist()}


class VNDirichletMixer(DirichletMixer):

    def __init__(self, k: int, seed: int, eps: float = 1e-3, sigma_warmup: int = 2):
        super().__init__(k, seed)
        self.eps = eps
        self.sigma_warmup = sigma_warmup
        self.contrib: list[np.ndarray] = []

    def sigma(self) -> np.ndarray:
        if len(self.contrib) < self.sigma_warmup:
            return np.ones(self.k)
        return np.std(np.stack(self.contrib), axis=0)

    def _increment(self, reward: float, w: np.ndarray) -> np.ndarray:
        self.contrib.append(reward * w)
        return reward * w / (self.sigma() + self.eps)

    def state(self) -> dict:
        return {"alpha": self.alpha.tolist(), "sigma": self.sigma().tolist()}
