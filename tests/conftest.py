"""Shared fixtures: a small synthetic corpus and a tiny-model configuration."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VOCAB_USED = 500        # token ids drawn from [0, 500) to keep the tables small


@pytest.fixture()
def data_dir(tmp_path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    rng = np.random.default_rng(0)
    train = rng.integers(0, VOCAB_USED, 40_000, dtype=np.uint16)
    train.tofile(d / "train.bin")
    rng.integers(0, VOCAB_USED, 4_000, dtype=np.uint16).tofile(d / "val.bin")
    counts = np.bincount(train, minlength=50257)
    torch.save(torch.from_numpy(-np.log((counts + 1.0) / (len(train) + 50257))).float(), d / "freq.pt")
    keys, cnt = np.unique(train[:-1].astype(np.int64) * 50257 + train[1:], return_counts=True)
    prev_cnt = np.bincount(keys // 50257, weights=cnt, minlength=50257)
    log_p = (np.log(cnt) - np.log(prev_cnt[keys // 50257])).astype(np.float32)
    torch.save({"keys": torch.from_numpy(keys), "log_freq": torch.from_numpy(log_p),
                "vocab_size": 50257, "fallback_log_freq": float(log_p.min() - np.log(2))},
               d / "bigram_freq.pt")
    return d


@pytest.fixture()
def tiny_config(tmp_path, data_dir):
    """Write a YAML config for a fast tiny-model run of `method`; returns its path."""
    def make(method: str = "mix4") -> Path:
        root = Path(__file__).resolve().parents[1] / "configs"
        cfg = {
            "include": [str(root / "base.yaml"), str(root / "models" / "tiny.yaml"),
                        str(root / "methods" / f"{method}.yaml")],
            "data": {"name": "synthetic", "dir": str(data_dir), "seq_len": 17, "batch_size": 2},
            "selection": {"reference_model": "tiny", "rescore_interval": 10},
            "training": {"steps": 60, "warmup_steps": 5},
            "eval": {"interval": 20, "batches": 2},
            "output": {"root": str(tmp_path / "runs"), "log_interval": 1},
        }
        path = tmp_path / f"{method}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path
    return make
