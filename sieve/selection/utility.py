"""Per-token utility (paper Eq. utility) and per-sequence top-k mask (paper §3)."""
from __future__ import annotations

import math

import torch


def mixed_utility(raw: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    raw [B, T, K] scorer values → utility [B, T] = Σ_k w_k · z(S_k).

    Each scorer is z-normalised over the batch tokens; column 0 (no loss
    term) is excluded from the statistics.
    """
    inner = raw[:, 1:]
    mean = inner.mean(dim=(0, 1), keepdim=True)
    std = inner.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    norm = (raw - mean) / std
    return (norm * weights.to(raw.device).view(1, 1, -1)).sum(dim=-1)


def retained_count(seq_len: int, alpha: float) -> int:
    """⌈α·n⌉ for the n = seq_len − 1 positions that carry a loss term (at least 1)."""
    n = seq_len - 1
    return max(1, min(n, math.ceil(alpha * n - 1e-9)))


def topk_mask(utility: torch.Tensor, alpha: float) -> torch.Tensor:
    """[B, T] bool: the top-⌈α(T−1)⌉ positions of each sequence among columns 1..T−1."""
    B, T = utility.shape
    idx = utility[:, 1:].topk(retained_count(T, alpha), dim=-1).indices
    mask = torch.zeros(B, T, dtype=torch.bool, device=utility.device)
    mask.scatter_(1, idx + 1, True)
    return mask
