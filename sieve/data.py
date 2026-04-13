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
        full_N    = len(self.arr)
        frac      = float(cfg.data_fraction) if split == "train" else 1.0
        frac      = max(0.0, min(1.0, frac))
        target    = int(full_N * frac)
        self.T    = cfg.seq_len
        # Random windows must fit inside [0, N); need N >= T+2 for sample_batch
        self.N    = max(self.T + 2, min(target, full_N))

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
        frac_note = f" (data_fraction={frac:.2g})" if split == "train" and frac < 1.0 else ""
        print(f"[data] {split}: {self.N/1e6:.1f}M tokens{frac_note}")

    def steps_per_epoch(self, batch_size: int) -> int:
        """
        Approximate optimizer steps for one pass over the train token prefix.
        Each step consumes batch_size * seq_len token positions; random sampling
        is with replacement, so this is a conventional scale, not a strict epoch.
        """
        return max(1, int(math.ceil(self.N / (batch_size * self.T))))

    def sample_batch(
        self,
        batch_size: int,
        device:     torch.device,
    ) -> dict:
        """Random batch. Returns input_ids [B,T] and optionally ref_losses [B,T]."""
        T       = self.T
        offsets = self.rng.integers(0, self.N - T - 1, size=batch_size)
        ids     = np.stack([self.arr[o:o+T].astype(np.int64) for o in offsets])

        # Build on CPU first; use pinned memory + non_blocking H2D on CUDA
        input_ids = torch.from_numpy(ids)
        if device.type == "cuda":
            input_ids = input_ids.pin_memory().to(device, non_blocking=True)
        else:
            input_ids = input_ids.to(device)

        out = {"input_ids": input_ids}

        if self.ref_losses is not None:
            rl = np.stack([self.ref_losses[o:o+T] for o in offsets])
            ref = torch.from_numpy(rl.copy())  # ensure contiguous float32
            if device.type == "cuda":
                ref = ref.pin_memory().to(device, non_blocking=True)
            else:
                ref = ref.to(device)
            out["ref_losses"] = ref

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
