"""S_D: inverse token frequency (paper §3)."""
from __future__ import annotations

from pathlib import Path

import torch

from .base import Scorer, ScoringInputs


class Frequency(Scorer):
    """S_D = −log of the token's empirical corpus frequency (table <data_dir>/freq.pt)."""
    name, slot = "frequency", "S_D"

    def __init__(self, data_dir: str, device: torch.device):
        path = Path(data_dir) / "freq.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (run scripts/prepare_data.py)")
        self.table = torch.load(path, map_location="cpu").to(device, dtype=torch.float32)

    @torch.no_grad()
    def score(self, x: ScoringInputs) -> torch.Tensor:
        return self.table[x.input_ids]
