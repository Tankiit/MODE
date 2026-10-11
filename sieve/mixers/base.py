"""
Mixer interface: how the K scorer utilities are weighted.

The selector calls `sample()` every T_rescore steps to get the weights used
for selection, and `update(reward)` at each validation event to credit the
weights that were active. Fixed mixers ignore the reward.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class Mixer(ABC):
    learns: bool = False

    def __init__(self, k: int):
        self.k = k

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """Weights for the next block of steps, [K] float32 on CPU."""

    def update(self, reward: float) -> None:
        """Credit the active weights with a non-negative reward (no-op by default)."""

    @abstractmethod
    def expected_weights(self) -> np.ndarray:
        """Current expected weights, [K] (logged; used by the analyses)."""

    def state(self) -> dict:
        return {}
