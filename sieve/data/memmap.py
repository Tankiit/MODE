"""
Token streams stored as flat uint16 memmaps (written by scripts/prepare_data.py).

Training batches are windows of seq_len tokens at uniformly random offsets
(with replacement); validation is a sequential, non-overlapping scan.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch


class MemmapDataset:

    def __init__(self, data_dir: str, split: str, seq_len: int,
                 seed: int, fraction: float = 1.0):
        p = Path(data_dir)
        path = p / f"{split}.bin"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (run scripts/prepare_data.py)")
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.T = seq_len
        frac = min(1.0, max(0.0, fraction)) if split == "train" else 1.0
        self.N = max(self.T + 2, min(int(len(self.arr) * frac), len(self.arr)))
        meta = p / "meta.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        # train and val use different streams of the same seed
        self.rng = np.random.default_rng(seed if split == "train" else seed + 1)

    def sample_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """[B, T] int64 windows at random offsets."""
        offsets = self.rng.integers(0, self.N - self.T - 1, size=batch_size)
        ids = np.stack([self.arr[o:o + self.T].astype(np.int64) for o in offsets])
        t = torch.from_numpy(ids)
        if device.type == "cuda":
            return t.pin_memory().to(device, non_blocking=True)
        return t.to(device)

    def val_batches(self, batch_size: int, device: torch.device,
                    max_batches: Optional[int] = None) -> Iterator[torch.Tensor]:
        """Sequential non-overlapping windows; all of them if max_batches is None."""
        T, n = self.T, 0
        for start in range(0, self.N - T, T * batch_size):
            rows = [self.arr[s:s + T].astype(np.int64)
                    for s in range(start, start + T * batch_size, T) if s + T <= self.N]
            if not rows:
                return
            yield torch.from_numpy(np.stack(rows)).to(device)
            n += 1
            if max_batches is not None and n >= max_batches:
                return
