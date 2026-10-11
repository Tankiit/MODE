#!/usr/bin/env python3
"""
Prepare a corpus for SIEVE training (RERUN_PLAN.md §3).

Writes to --out:
  train.bin, val.bin  uint16 GPT-2 token streams (format of sieve/data.py)
  freq.pt             float32 [V]: S_D(v) = −log f(v), f(v) = (count(v)+1) / (N+V)
  bigram_freq.pt      (--bigram) sorted keys prev*V+tok with log p(tok | prev),
                      in the format read by TokenScorers.score_bigram_freq
  meta.json           sources, token counts and recipe choices

S_E reference losses are not precomputed: the frozen reference model runs
on each training batch (sieve/reference.py).

Datasets
  wikitext103, wikitext2  HF "Salesforce/wikitext" (raw configs), splits
                          train / validation. Rows are concatenated as in the
                          original text (each row keeps its trailing "\\n").
  openwebmath             HF "open-web-math/open-web-math", streamed. The
                          first --train_tokens tokens go to train.bin, the
                          next --val_tokens to val.bin; the document that
                          crosses the boundary is truncated and its remainder
                          dropped, so no document is in both splits.
                          <|endoftext|> follows each document.

Usage
  python scripts/prepare_data.py --dataset wikitext103 --out ./data/wikitext103 --bigram
  python scripts/prepare_data.py --dataset openwebmath --out ./data/openwebmath_1b \\
      --train_tokens 1000000000 --val_tokens 1000000 --bigram
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

DTYPE = np.uint16          # must match sieve/data.py
CHUNK = 50_000_000         # tokens per pass when scanning train.bin

WIKITEXT = {"wikitext103": "wikitext-103-raw-v1", "wikitext2": "wikitext-2-raw-v1"}


class TokenWriter:
    """Append token ids to a uint16 file, stopping at an optional budget."""

    def __init__(self, path: Path, budget: int | None):
        self.f = open(path, "wb")
        self.budget = budget
        self.n = 0

    @property
    def full(self) -> bool:
        return self.budget is not None and self.n >= self.budget

    def write(self, ids: list[int]) -> None:
        if self.budget is not None:
            ids = ids[: self.budget - self.n]
        if ids:
            np.asarray(ids, dtype=DTYPE).tofile(self.f)
            self.n += len(ids)

    def close(self) -> int:
        self.f.close()
        return self.n


def build_wikitext(name: str, out: Path, tok, train_budget, val_budget) -> dict:
    from datasets import load_dataset

    config = WIKITEXT[name]
    counts = {}
    for split, fname, budget in (("train", "train.bin", train_budget),
                                 ("validation", "val.bin", val_budget)):
        ds = load_dataset("Salesforce/wikitext", config, split=split)
        w = TokenWriter(out / fname, budget)
        rows = ds["text"]
        for i in range(0, len(rows), 2000):
            w.write(tok("".join(rows[i:i + 2000]))["input_ids"])
            if w.full:
                break
        counts[fname] = w.close()
        print(f"[prepare] {fname}: {counts[fname]:,} tokens")
    return {"hf_name": "Salesforce/wikitext", "hf_config": config,
            "val_source": "validation split", "eos_between_documents": False,
            "train_tokens": counts["train.bin"], "val_tokens": counts["val.bin"]}


def build_openwebmath(out: Path, tok, train_budget: int, val_budget: int) -> dict:
    from datasets import load_dataset

    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    eos = tok.eos_token_id
    train_w = TokenWriter(out / "train.bin", train_budget)
    val_w = TokenWriter(out / "val.bin", val_budget)
    n_docs = {"train": 0, "val": 0}
    batch: list[str] = []
    t0 = time.time()

    def flush(texts: list[str]) -> bool:
        for ids in tok(texts)["input_ids"]:
            ids = ids + [eos]
            if not train_w.full:
                # the document that crosses the budget is truncated; its
                # remainder is dropped, so no document is in both splits
                train_w.write(ids)
                n_docs["train"] += 1
            elif not val_w.full:
                val_w.write(ids)
                n_docs["val"] += 1
            else:
                return True
        print(f"[prepare] train {train_w.n:,}  val {val_w.n:,}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        return val_w.full

    for ex in ds:
        text = ex.get("text")
        if text and text.strip():
            batch.append(text)
        if len(batch) == 512:
            if flush(batch):
                break
            batch = []
    else:
        flush(batch)
    n_train, n_val = train_w.close(), val_w.close()
    print(f"[prepare] train.bin: {n_train:,} tokens  val.bin: {n_val:,} tokens")
    return {"hf_name": "open-web-math/open-web-math", "hf_config": None,
            "val_source": "documents following the train documents in the stream",
            "eos_between_documents": True, "documents": n_docs,
            "train_tokens": n_train, "val_tokens": n_val}


def train_chunks(path: Path):
    arr = np.memmap(path, dtype=DTYPE, mode="r")
    for s in range(0, len(arr), CHUNK):
        # one token of overlap so that bigrams across chunk borders are counted once
        yield s, np.asarray(arr[s:min(s + CHUNK + 1, len(arr))], dtype=np.int64)


def build_freq(out: Path, vocab: int) -> dict:
    counts = np.zeros(vocab, dtype=np.int64)
    n = 0
    for s, chunk in train_chunks(out / "train.bin"):
        body = chunk[:CHUNK]                       # drop the overlap token
        counts += np.bincount(body, minlength=vocab)[:vocab]
        n += len(body)
    log_f = np.log((counts + 1.0) / (n + vocab))
    torch.save(torch.from_numpy(-log_f).float(), out / "freq.pt")
    print(f"[prepare] freq.pt: V={vocab}, N={n:,}, seen types={int((counts > 0).sum()):,}")
    return {"freq_formula": "-log((count+1)/(N+V))", "freq_seen_types": int((counts > 0).sum())}


def build_bigram(out: Path, vocab: int) -> dict:
    keys = np.empty(0, dtype=np.int64)
    cnts = np.empty(0, dtype=np.int64)
    for s, chunk in train_chunks(out / "train.bin"):
        k = chunk[:-1] * vocab + chunk[1:]
        uk, uc = np.unique(k, return_counts=True)
        # merge with the running table
        allk = np.concatenate([keys, uk])
        allc = np.concatenate([cnts, uc])
        keys, inv = np.unique(allk, return_inverse=True)
        cnts = np.bincount(inv, weights=allc).astype(np.int64)
        print(f"[prepare] bigrams after {s + len(chunk):,} tokens: {len(keys):,} types")
    prev = keys // vocab
    prev_counts = np.bincount(prev, weights=cnts, minlength=vocab)
    log_p = (np.log(cnts) - np.log(prev_counts[prev])).astype(np.float32)
    fallback = float(log_p.min() - math.log(2.0))
    torch.save({"keys": torch.from_numpy(keys), "log_freq": torch.from_numpy(log_p),
                "vocab_size": vocab, "fallback_log_freq": fallback},
               out / "bigram_freq.pt")
    print(f"[prepare] bigram_freq.pt: {len(keys):,} bigram types")
    return {"bigram_estimate": "MLE log p(tok|prev) over train.bin",
            "bigram_fallback": "min observed log p minus log 2 (unseen pairs)",
            "bigram_types": int(len(keys))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["wikitext103", "wikitext2", "openwebmath"])
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--train_tokens", type=int, default=None,
                   help="train budget (required for openwebmath; optional cap otherwise)")
    p.add_argument("--val_tokens", type=int, default=None,
                   help="validation budget (required for openwebmath; optional cap otherwise)")
    p.add_argument("--bigram", action="store_true", help="also write bigram_freq.pt")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok.model_max_length = int(1e12)       # we tokenize long texts on purpose
    vocab = len(tok)
    assert vocab <= np.iinfo(DTYPE).max + 1, "vocabulary does not fit in uint16"

    if args.dataset == "openwebmath":
        if args.train_tokens is None or args.val_tokens is None:
            p.error("openwebmath needs --train_tokens and --val_tokens")
        meta = build_openwebmath(out, tok, args.train_tokens, args.val_tokens)
    else:
        meta = build_wikitext(args.dataset, out, tok, args.train_tokens, args.val_tokens)

    meta.update({"dataset": args.dataset, "tokenizer": args.tokenizer, "vocab_size": vocab,
                 "dtype": "uint16", "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    meta.update(build_freq(out, vocab))
    if args.bigram:
        meta.update(build_bigram(out, vocab))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[prepare] done: {out}", flush=True)


if __name__ == "__main__":
    main()
    # The streaming parquet reader of `datasets` can leave threads that block
    # interpreter shutdown (observed with datasets 5.1 / Python 3.14, even
    # in a minimal script). All outputs are written and closed here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
