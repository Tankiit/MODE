"""
Zero-overhead memmap dataset.
Run prepare.py once to create the .bin files, then use this everywhere.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import torch
from typing import Optional, Iterator
from .config import SieveConfig


class MemmapDataset:
    """
    Reads a flat uint16 token array from disk.
    All I/O is random-offset slicing — no multiprocessing, MPS-safe.
    """

    def __init__(
        self,
        data_dir: str,
        split:    str,          # "train" or "val"
        cfg:      SieveConfig,
    ):
        p         = Path(data_dir)
        self.arr  = np.memmap(p / f"{split}.bin", dtype=np.uint16, mode="r")
        self.N    = len(self.arr)
        self.T    = cfg.seq_len

        with open(p / "meta.json") as f:
            self.meta = json.load(f)

        # Optional S_D scorer table — [vocab_size] float32 log-IDF
        self.freq_table: Optional[torch.Tensor] = None
        fp = p / "freq.pt"
        if fp.exists():
            self.freq_table = torch.load(fp, map_location="cpu")

        # Optional S_E offline reference losses — [n_tokens] float32
        self.ref_losses: Optional[np.memmap] = None
        rp = p / "ref_losses.bin"
        if rp.exists():
            self.ref_losses = np.memmap(rp, dtype=np.float32, mode="r")

        self.rng = np.random.default_rng(
            cfg.seed if split == "train" else cfg.seed + 1
        )
        print(f"[data] {split}: {self.N/1e6:.1f}M tokens")

    def sample_batch(
        self,
        batch_size: int,
        device:     torch.device,
    ) -> dict:
        """Random batch. Returns input_ids [B,T] and optionally ref_losses [B,T]."""
        T       = self.T
        offsets = self.rng.integers(0, self.N - T - 1, size=batch_size)
        ids = np.stack([self.arr[o:o+T].astype(np.int64) for o in offsets])
        out = {"input_ids": torch.from_numpy(ids).to(device)}

        if self.ref_losses is not None:
            rl = np.stack([self.ref_losses[o:o+T] for o in offsets])
            out["ref_losses"] = torch.from_numpy(rl.copy()).to(device)

        return out

    def val_batches(
        self,
        batch_size: int,
        device:     torch.device,
    ) -> Iterator[dict]:
        """Non-overlapping sequential scan for PPL evaluation."""
        T = self.T
        for start in range(0, self.N - T, T * batch_size):
            batch = []
            for b in range(batch_size):
                s = start + b * T
                if s + T > self.N:
                    break
                batch.append(self.arr[s:s+T].astype(np.int64))
            if batch:
                yield {
                    "input_ids": torch.from_numpy(
                        np.stack(batch)
                    ).to(device)
                }
