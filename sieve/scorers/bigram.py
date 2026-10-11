"""Control for the S_D slot: bigram rarity (paper App. B, "Bigram scorer substitution")."""
from __future__ import annotations

from pathlib import Path

import torch

from .base import Scorer, ScoringInputs


class Bigram(Scorer):
    """
    −log p_bigram(y_t | y_{t−1}) estimated on the training corpus (table
    <data_dir>/bigram_freq.pt). Column 0 has no previous token and gets the
    row mean.
    """
    name, slot = "bigram", "S_D"

    def __init__(self, data_dir: str, device: torch.device):
        path = Path(data_dir) / "bigram_freq.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (scripts/prepare_data.py --bigram)")
        t = torch.load(path, map_location="cpu", weights_only=False)
        self.keys = t["keys"].to(device)                          # sorted prev*V + tok
        self.log_p = t["log_freq"].to(device, dtype=torch.float32)
        self.vocab = int(t["vocab_size"])
        self.fallback = float(t["fallback_log_freq"])

    @torch.no_grad()
    def score(self, x: ScoringInputs) -> torch.Tensor:
        ids = x.input_ids
        query = (ids[:, :-1] * self.vocab + ids[:, 1:]).reshape(-1)
        idx = torch.searchsorted(self.keys, query).clamp(max=len(self.keys) - 1)
        found = self.keys[idx] == query
        log_p = torch.where(found, self.log_p[idx], torch.full_like(self.log_p[idx], self.fallback))
        out = torch.empty(ids.shape, dtype=torch.float32, device=ids.device)
        out[:, 1:] = -log_p.reshape(ids.shape[0], -1)
        out[:, 0] = out[:, 1:].mean(dim=1)
        return out
