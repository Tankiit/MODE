"""AdamW with linear warmup and cosine decay (paper App. B)."""
from __future__ import annotations

import math

import torch

from ..config import TrainingConfig


def lr_factor(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """Multiplier of the peak lr at 0-based `step`: linear warmup, then cosine to min_ratio at `total`."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: torch.nn.Module, cfg: TrainingConfig):
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                              weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: lr_factor(s, cfg.warmup_steps, cfg.steps, cfg.min_lr_ratio))
    return optim, sched
