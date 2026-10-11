"""
TokenSelector: scorers + mixer + per-sequence top-k (paper §3, Algorithm 1).

Per training step:
  begin_step(step)   every T_rescore steps, sample new mixture weights
  select(inputs)     score the CURRENT batch and return its mask
Per validation event:
  on_validation(L)   r = max{0, L_prev − L} credits the active weights
                     (the first event only records L)
"""
from __future__ import annotations

from typing import Optional

import torch

from ..mixers import Mixer
from ..scorers import Scorer, ScoringInputs
from .utility import mixed_utility, topk_mask


class TokenSelector:

    def __init__(self, scorers: list[Scorer], mixer: Mixer, alpha: float, rescore_interval: int):
        if mixer.k != len(scorers):
            raise ValueError("one mixer weight per scorer is required")
        self.scorers = scorers
        self.mixer = mixer
        self.alpha = alpha
        self.rescore_interval = rescore_interval
        self.weights: Optional[torch.Tensor] = None
        self._last_sample: Optional[int] = None
        self._prev_val: Optional[float] = None
        self.sample_log: list[dict] = []
        self.reward_log: list[dict] = []
        self.n_selected = 0

    @property
    def needs_reference(self) -> bool:
        return any(s.needs_reference for s in self.scorers)

    @property
    def slots(self) -> list[str]:
        return [s.slot for s in self.scorers]

    def begin_step(self, step: int) -> None:
        if self._last_sample is None or step - self._last_sample >= self.rescore_interval:
            self.weights = self.mixer.sample()
            self._last_sample = step
            self.sample_log.append({"step": step, "weights": self.weights.tolist()})

    @torch.no_grad()
    def select(self, x: ScoringInputs) -> torch.Tensor:
        if self.weights is None:
            raise RuntimeError("begin_step() must be called before select()")
        raw = torch.stack([s.score(x) for s in self.scorers], dim=-1)
        mask = topk_mask(mixed_utility(raw, self.weights), self.alpha)
        self.n_selected = int(mask.sum())
        return mask

    def on_validation(self, val_loss: float, step: int) -> None:
        if self._prev_val is None:
            self._prev_val = val_loss
            return
        reward = max(0.0, self._prev_val - val_loss)
        self._prev_val = val_loss
        self.mixer.update(reward)
        self.reward_log.append({"step": step, "val_loss": val_loss, "reward": reward})
