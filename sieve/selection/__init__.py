from __future__ import annotations

from typing import Optional

import torch

from ..config import RunConfig
from ..mixers import build_mixer
from ..scorers import build_scorer
from .selector import TokenSelector
from .utility import mixed_utility, retained_count, topk_mask


def build_selector(cfg: RunConfig, device: torch.device) -> Optional[TokenSelector]:
    """None for full CLM (selection.alpha is null)."""
    s = cfg.selection
    if s.alpha is None:
        return None
    noise_seed = cfg.seed if s.noise_seed is None else s.noise_seed
    scorers = [build_scorer(n, cfg.data.dir, noise_seed, device) for n in s.scorers]
    mixer = build_mixer(s, len(scorers), cfg.seed)
    return TokenSelector(scorers, mixer, s.alpha, s.rescore_interval)


__all__ = ["TokenSelector", "build_selector", "mixed_utility", "retained_count", "topk_mask"]
