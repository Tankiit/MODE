#!/usr/bin/env python3
"""
Tokenise N tokens from C4 (English) and save raw uint16 memmaps for train_sieve MemmapDataset.

Run once locally or via Modal (see modal_prepare_c4.py). Streams to disk — does not
hold 1B tokens in RAM.

Output:
  <out>/train.bin, val.bin  — raw uint16 (same format as prepare_sieve.py)
  <out>/meta.json
  <out>/freq.pt             — log-IDF for S_D (same recipe as prepare_sieve.py)

Usage:
  python scripts/prepare_c4.py
  python scripts/prepare_c4.py --tokens 500_000_000
  python scripts/prepare_c4.py --output ./data/c4_1b
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

SEQ_LEN_DEFAULT = 512


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tokens",
        type=int,
        default=1_000_000_000,
        help="Total tokens to extract (default: 1B)",
    )
    p.add_argument("--output", default="./data/c4_1b", help="Output directory")
    p.add_argument(
        "--model",
        default="gpt2",
        help="Tokenizer HF id (must match train_sieve model vocab)",
    )
    p.add_argument(
        "--val_frac",
        type=float,
        default=0.05,
        help="Validation fraction (default: 5%%)",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=SEQ_LEN_DEFAULT,
        help="Recorded in meta.json (must match SieveConfig.seq_len)",
    )
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = args.tokens
    n_train = int(n_total * (1.0 - args.val_frac))
    n_val = n_total - n_train

    print(f"Target:    {n_total/1e6:.0f}M tokens  (train {n_train/1e6:.2f}M, val {n_val/1e6:.2f}M)")
    print(f"Output:    {out_dir}")
    print(f"Tokenizer: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    train_mm = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(n_train,))
    val_mm = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(n_val,))

    print("\nLoading C4 (streaming)...")
    ds = load_dataset(
        "allenai/c4",
        "en",
        streaming=True,
        split="train",
        trust_remote_code=True,
    )

    i_train = 0
    i_val = 0
    n_docs = 0
    t0 = time.time()
    milestone_m = 0

    def total_written() -> int:
        return i_train + i_val

    for ex in ds:
        ids = tokenizer.encode(ex["text"], add_special_tokens=False)
        n_docs += 1
        for tid in ids:
            tid = int(tid)
            if tid < 0 or tid >= vocab_size:
                tid = max(0, min(tid, vocab_size - 1))
            if i_train < n_train:
                train_mm[i_train] = np.uint16(tid)
                i_train += 1
            elif i_val < n_val:
                val_mm[i_val] = np.uint16(tid)
                i_val += 1
            else:
                break
        tw = total_written()
        if tw >= 10_000_000:
            m10 = tw // 10_000_000
            if m10 > milestone_m:
                milestone_m = m10
                elapsed = time.time() - t0
                rate = tw / max(elapsed, 1e-6) / 1e6
                eta_min = (n_total - tw) / max(rate * 1e6, 1e-9) / 60
                print(
                    f"  {tw/1e6:6.0f}M tokens  ({n_docs:,} docs)  "
                    f"{rate:.1f}M tok/s  ETA: {eta_min:.0f} min"
                )
        if i_train >= n_train and i_val >= n_val:
            break

    train_mm.flush()
    val_mm.flush()
    del train_mm, val_mm

    print(f"\nCollected {total_written():,} tokens from {n_docs:,} documents")

    meta = {
        "vocab_size": vocab_size,
        "seq_len": args.seq_len,
        "dataset": "c4_1b",
        "tokenizer": args.model,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote meta.json (seq_len={args.seq_len})")

    print("\nBuilding S_D frequency table (log-IDF) over train split...")
    train_mm = np.memmap(train_path, dtype=np.uint16, mode="r")
    counts = np.zeros(vocab_size, dtype=np.float64)
    chunk = 10_000_000
    for i in range(0, len(train_mm), chunk):
        part = np.asarray(train_mm[i : i + chunk], dtype=np.int64)
        part = part.clip(0, vocab_size - 1)
        counts += np.bincount(part, minlength=vocab_size)

    n_tok = float(len(train_mm))
    log_idf = np.log((n_tok + 1.0) / (counts + 1.0)).astype(np.float32)
    freq_path = out_dir / "freq.pt"
    torch.save(torch.from_numpy(log_idf), str(freq_path))
    coverage = int((counts > 0).sum())
    print(f"  Vocab coverage: {coverage:,}/{vocab_size:,} types seen")
    print(f"  Saved: {freq_path}")

    print("\nVerification:")
    tr = np.memmap(train_path, dtype=np.uint16, mode="r")
    va = np.memmap(val_path, dtype=np.uint16, mode="r")
    print(f"  train.bin: {len(tr):,} tokens, dtype={tr.dtype}")
    print(f"  val.bin:   {len(va):,} tokens")
    fl = torch.load(str(freq_path), map_location="cpu")
    print(f"  freq.pt:   shape={tuple(fl.shape)}, dtype={fl.dtype}")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")
    print("Use DATASET_DIRS entry c4_1b -> ./data/c4_1b in train_sieve.py")
    print("Modal: modal run modal_prepare_c4.py")


if __name__ == "__main__":
    main()
