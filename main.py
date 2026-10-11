#!/usr/bin/env python3
"""
Run one training experiment from YAML configuration files.

  python main.py --config configs/experiments/smoke.yaml
  python main.py --config configs/base.yaml --config configs/models/gpt2-large.yaml \\
                 --config configs/data/wikitext103.yaml --config configs/methods/mix4.yaml \\
                 --set seed=43 --set selection.alpha=0.5

Files are merged left to right, then --set overrides. Each run gets a new
directory under output.root (see sieve/runs/run_dir.py); nothing is ever
overwritten. --skip-if-done exits without training if a completed run with
the same configuration (same config hash) already exists.
"""
from __future__ import annotations

import argparse
import random
import sys

import numpy as np
import torch
import yaml

from sieve.config import config_hash, load_config, to_dict
from sieve.data import MemmapDataset
from sieve.models import build_model, load_reference
from sieve.runs import (create_run_dir, find_completed, mark_done, setup_logging,
                        write_json_atomic, write_metadata)
from sieve.selection import build_selector
from sieve.training import train


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", action="append", required=True, help="YAML file (repeatable)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override, e.g. training.steps=300 (repeatable)")
    p.add_argument("--skip-if-done", action="store_true",
                   help="exit if a completed run with the same config hash exists")
    p.add_argument("--dry-run", action="store_true", help="print the resolved config and exit")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.set)
    h = config_hash(cfg)
    if args.dry_run:
        print(yaml.safe_dump({**to_dict(cfg), "config_hash": h}, sort_keys=False))
        return 0
    if args.skip_if_done:
        done = find_completed(cfg.output.root, h)
        if done:
            print(f"[skip] {cfg.run_name}: already completed in {done[-1]}")
            return 0

    run_dir = create_run_dir(cfg.output.root, cfg.run_name)
    log = setup_logging(run_dir)
    write_metadata(run_dir, cfg)
    log.info("run %s  (config hash %s)", run_dir, h)

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = MemmapDataset(cfg.data.dir, "train", cfg.data.seq_len, cfg.seed, cfg.data.fraction)
    val_ds = MemmapDataset(cfg.data.dir, "val", cfg.data.seq_len, cfg.seed)
    m = cfg.model
    model = build_model(m.name, cfg.data.seq_len, device, m.tiny_layers, m.tiny_dim, m.tiny_heads)
    selector = build_selector(cfg, device)
    ref_model = None
    if selector is not None and selector.needs_reference:
        ref_model = load_reference(cfg.selection.reference_model, device)
    log.info("device %s | model %s (%.1fM params) | method %s | alpha %s | reference %s",
             device, m.name, sum(p.numel() for p in model.parameters()) / 1e6,
             cfg.selection.method, cfg.selection.alpha,
             cfg.selection.reference_model if ref_model is not None else "-")

    history = train(model, selector, ref_model, train_ds, val_ds, cfg, device)

    if cfg.output.save_final:
        torch.save(model.state_dict(), run_dir / "final.pt")
    write_json_atomic(run_dir / "history.json", history)
    mark_done(run_dir, {"final_val_ppl": history["final_val_ppl"],
                        "final_val_loss": history["final_val_loss"],
                        "n_steps": history["n_steps"], "wall_time_s": history["wall_time_s"],
                        "config_hash": h})
    log.info("done: %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
