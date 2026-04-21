"""
One-shot dataset preparation on Modal.

Builds memmap + freq.pt for SIEVE training on:
  - wikitext2       (~2M tokens)
  - wikitext103     (~100M tokens)
  - openwebmath_1b  (~1B tokens, streaming subset of full OWM)

Writes to the `sieve-data` volume:
  /data/<name>/
    train.bin      uint16 tokenized stream (GPT-2 BPE)
    val.bin
    freq.pt        corpus-level inverse-frequency table for S_D

CRITICAL: dtype is uint16 throughout — must match MemmapDataset in sieve/data.py.
GPT-2 vocab is 50257, which fits comfortably in uint16 (max 65535).

Usage:
  modal run modal_prepare_data.py                                   # wt103 default
  modal run modal_prepare_data.py --datasets openwebmath_1b        # OWM only
  modal run modal_prepare_data.py --datasets wikitext103,openwebmath_1b

Idempotent: skips datasets whose files already exist AND are complete.
Partial outputs (e.g. from preempted streaming runs) trigger rebuild.
"""
from __future__ import annotations

from pathlib import Path

import modal


app = modal.App("sieve-prepare-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets==2.21.0",
        "transformers==4.44.2",
        "numpy==1.26.4",
        "torch==2.4.0",
        "tqdm",
    )
)

volume_data = modal.Volume.from_name("sieve-data", create_if_missing=True)
volume_hf   = modal.Volume.from_name("sieve-hf-cache", create_if_missing=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "wikitext2": {
        "hf_name":            "wikitext",
        "hf_config":          "wikitext-2-raw-v1",
        "train_tokens_target": None,
        "val_tokens_target":   None,
        "text_field":         "text",
        "stream":             False,
    },
    "wikitext103": {
        "hf_name":            "wikitext",
        "hf_config":          "wikitext-103-raw-v1",
        "train_tokens_target": None,
        "val_tokens_target":   None,
        "text_field":         "text",
        "stream":             False,
    },
    "openwebmath_1b": {
        "hf_name":            "open-web-math/open-web-math",
        "hf_config":          None,
        "train_tokens_target": 1_000_000_000,
        "val_tokens_target":   1_000_000,
        "text_field":         "text",
        "stream":             True,
    },
}

# All memmap I/O uses this dtype. MUST match sieve/data.py MemmapDataset.
DTYPE = "uint16"


# ─────────────────────────────────────────────────────────────────────────────
# Main prepare function
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    cpu=4, memory=32_768,
    timeout=4 * 3600,
    volumes={"/data": volume_data, "/hf_cache": volume_hf},
)
def prepare(datasets: list[str] = ["wikitext103"],
            tokenizer_model: str = "gpt2") -> dict:
    import os
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache"
    os.environ["HF_HOME"]           = "/hf_cache"

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # Sanity: vocab must fit in uint16
    assert vocab_size <= 65535, (
        f"vocab_size={vocab_size} does not fit in uint16. "
        f"Use uint32 and update sieve/data.py MemmapDataset."
    )

    summary: dict = {}

    for ds_key in datasets:
        if ds_key not in DATASET_REGISTRY:
            summary[ds_key] = {"status": "unknown_dataset"}
            print(f"[prepare] {ds_key} not in registry — skip")
            continue

        meta = DATASET_REGISTRY[ds_key]
        out  = Path(f"/data/{ds_key}")
        train_bin = out / "train.bin"
        val_bin   = out / "val.bin"
        freq_pt   = out / "freq.pt"

        # ── Idempotence check with completeness verification ──────────
        target_train = meta.get("train_tokens_target")
        meta_path    = out / "meta.json" 
        if train_bin.exists() and val_bin.exists() and freq_pt.exists() and meta_path.exists():
            n = len(np.memmap(train_bin, dtype=DTYPE, mode="r"))
            # Accept if target is None (whole-split) OR we hit >= 95% of target
            is_complete = (target_train is None) or (n >= 0.95 * target_train)
            if is_complete:
                print(f"[prepare] {ds_key} complete ({n:,} tokens) — skip")
                summary[ds_key] = {"status": "cached", "train_tokens": int(n)}
                continue
            else:
                pct = 100.0 * n / target_train
                print(f"[prepare] {ds_key} PARTIAL ({n:,}/{target_train:,} "
                      f"tokens, {pct:.0f}%) — REBUILDING")
                train_bin.unlink()
                freq_pt.unlink()
                if val_bin.exists():
                    val_bin.unlink()
                if meta_path.exists():
                    meta_path.unlink()    
        out.mkdir(parents=True, exist_ok=True)

        # ── Train split ────────────────────────────────────────────────
        if not train_bin.exists():
            print(f"[prepare] building {ds_key}/train")
            if meta["stream"]:
                _build_streaming(
                    bin_path      = train_bin,
                    hf_name       = meta["hf_name"],
                    hf_config     = meta["hf_config"],
                    split         = "train",
                    tokens_target = meta["train_tokens_target"],
                    text_field    = meta["text_field"],
                    tokenizer     = tokenizer,
                )
            else:
                _build_full(
                    bin_path      = train_bin,
                    hf_name       = meta["hf_name"],
                    hf_config     = meta["hf_config"],
                    split         = "train",
                    text_field    = meta["text_field"],
                    tokenizer     = tokenizer,
                )

        # ── Val split ─────────────────────────────────────────────────
        if not val_bin.exists():
            print(f"[prepare] building {ds_key}/val")
            if meta["stream"]:
                _build_streaming(
                    bin_path      = val_bin,
                    hf_name       = meta["hf_name"],
                    hf_config     = meta["hf_config"],
                    split         = "train",
                    tokens_target = meta["val_tokens_target"],
                    text_field    = meta["text_field"],
                    tokenizer     = tokenizer,
                    skip_examples = _estimate_skip_for_owm(meta),
                )
            else:
                _build_full(
                    bin_path      = val_bin,
                    hf_name       = meta["hf_name"],
                    hf_config     = meta["hf_config"],
                    split         = "validation",
                    text_field    = meta["text_field"],
                    tokenizer     = tokenizer,
                )

        # ── freq.pt — corpus-level IDF for S_D scorer ─────────────────
        if not freq_pt.exists():
            print(f"[prepare] building {ds_key}/freq.pt")
            train_arr = np.memmap(train_bin, dtype=DTYPE, mode="r")
            counts = np.bincount(train_arr, minlength=vocab_size).astype(np.float64)
            inv_freq = 1.0 / (counts + 1.0)
            inv_freq = inv_freq / inv_freq.mean()
            torch.save(torch.from_numpy(inv_freq).float(), freq_pt)
            print(f"[prepare]   freq.pt vocab={vocab_size}  "
                  f"mean={inv_freq.mean():.3f}  max={inv_freq.max():.1f}")
        meta_path = out / "meta.json"
        if not meta_path.exists():
            print(f"[prepare] building {ds_key}/meta.json")
            import json
            meta_json = {
                "vocab_size": int(vocab_size),
                "seq_len":    1024,              # ← sensible default, overridden by cfg anyway
                "dataset":    ds_key,
                "tokenizer":  tokenizer_model,
                "dtype":      DTYPE,             # documents the on-disk format
            }
            meta_path.write_text(json.dumps(meta_json, indent=2))
            print(f"[prepare]   meta.json written: {meta_json}")

        n_train = len(np.memmap(train_bin, dtype=DTYPE, mode="r"))
        n_val   = len(np.memmap(val_bin,   dtype=DTYPE, mode="r"))
        summary[ds_key] = {
            "status":       "built",
            "train_tokens": int(n_train),
            "val_tokens":   int(n_val),
            "vocab_size":   int(vocab_size),
        }
        print(f"[prepare] {ds_key}: train={n_train:,} val={n_val:,}")

    volume_data.commit()
    volume_hf.commit()
    print("[prepare] done.")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_full(bin_path, hf_name, hf_config, split, text_field, tokenizer):
    """For small corpora (wt2, wt103) — load everything then tokenize."""
    from datasets import load_dataset
    from tqdm import tqdm
    import numpy as np

    ds = load_dataset(hf_name, hf_config, split=split, cache_dir="/hf_cache")
    ids_all: list[int] = []
    for ex in tqdm(ds, desc=f"tokenize {hf_name}/{split}"):
        txt = ex[text_field]
        if not txt or not txt.strip():
            continue
        ids_all.extend(tokenizer.encode(txt))
    arr = np.array(ids_all, dtype=DTYPE)
    arr.tofile(bin_path)
    print(f"[prepare]   {bin_path.name}: {len(arr):,} tokens")


def _build_streaming(bin_path, hf_name, hf_config, split,
                     tokens_target, text_field, tokenizer,
                     skip_examples: int = 0):
    """
    For large corpora (OWM) — stream examples, tokenize, stop at budget.
    Uses chunked writes to avoid RAM blowup.
    """
    from datasets import load_dataset
    from tqdm import tqdm
    import numpy as np

    ds = load_dataset(
        hf_name, hf_config,
        split=split, streaming=True, cache_dir="/hf_cache",
    )
    if skip_examples > 0:
        ds = ds.skip(skip_examples)

    CHUNK_TOKENS = 10_000_000
    buffer: list[int] = []
    total_written = 0

    with open(bin_path, "wb") as f:
        pbar = tqdm(total=tokens_target, desc=f"stream {hf_name}", unit="tok")
        for ex in ds:
            txt = ex.get(text_field)
            if not txt or not txt.strip():
                continue
            ids = tokenizer.encode(txt)
            buffer.extend(ids)

            if len(buffer) >= CHUNK_TOKENS:
                arr = np.array(buffer, dtype=DTYPE)
                remaining = tokens_target - total_written
                if len(arr) > remaining:
                    arr = arr[:remaining]
                arr.tofile(f)
                total_written += len(arr)
                pbar.update(len(arr))
                buffer = []
                if total_written >= tokens_target:
                    break

        if buffer and total_written < tokens_target:
            arr = np.array(buffer, dtype=DTYPE)
            remaining = tokens_target - total_written
            if len(arr) > remaining:
                arr = arr[:remaining]
            arr.tofile(f)
            total_written += len(arr)
            pbar.update(len(arr))

        pbar.close()

    print(f"[prepare]   {bin_path.name}: {total_written:,} tokens written")


def _estimate_skip_for_owm(meta: dict) -> int:
    if meta["train_tokens_target"] is None:
        return 0
    return max(100, meta["train_tokens_target"] // 2000)


# ─────────────────────────────────────────────────────────────────────────────
# Local entrypoint
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(datasets: str = "wikitext103"):
    ds_list = [d.strip() for d in datasets.split(",")]
    print(f"[launcher] preparing: {ds_list}")
    result = prepare.remote(ds_list)
    print("\n[launcher] summary:")
    for ds, info in result.items():
        print(f"  {ds}: {info}")