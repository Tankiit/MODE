#!/usr/bin/env python3
"""
Non-regression check against the reference implementation (branch
fix-per-step-mask, commit d965f77): both implementations train the same
tiny GPT-2 on the same synthetic corpus, and every step must give the same
mask, the same loss and the same mixture weights.

  python tests/regression/compare_reference.py --reference ../MODE

Settings are chosen where the two implementations are meant to agree
exactly: T = 17 and α = 0.5, so that the reference's round(α·T) and the
paper's ⌈α(T−1)⌉ (used here) both retain 8 tokens; the noise control uses the
reference's fixed generator seed 12345.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
NEW_ROOT = HERE.parents[2]
METHODS = ["mix4", "vn_mix", "se_only", "su_only", "mix4_bigram", "mix4_noise", "clm"]
STEPS, T, B, ALPHA, RESCORE, EVAL, SEED = 60, 17, 2, 0.5, 10, 20, 42


def make_data(d: Path) -> None:
    import torch
    rng = np.random.default_rng(0)
    train = rng.integers(0, 500, 40_000, dtype=np.uint16)
    train.tofile(d / "train.bin")
    rng.integers(0, 500, 4_000, dtype=np.uint16).tofile(d / "val.bin")
    (d / "meta.json").write_text("{}")
    counts = np.bincount(train, minlength=50257)
    torch.save(torch.from_numpy(-np.log((counts + 1.0) / (len(train) + 50257))).float(), d / "freq.pt")
    keys, cnt = np.unique(train[:-1].astype(np.int64) * 50257 + train[1:], return_counts=True)
    prev = np.bincount(keys // 50257, weights=cnt, minlength=50257)
    log_p = (np.log(cnt) - np.log(prev[keys // 50257])).astype(np.float32)
    torch.save({"keys": torch.from_numpy(keys), "log_freq": torch.from_numpy(log_p),
                "vocab_size": 50257, "fallback_log_freq": float(log_p.min() - np.log(2))},
               d / "bigram_freq.pt")


def tiny_models():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(vocab_size=50257, n_positions=T, n_embd=32, n_layer=2, n_head=2,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg)
    torch.manual_seed(1)
    ref = GPT2LMHeadModel(cfg).eval().requires_grad_(False)
    return model, ref


def worker_old(method: str, data: Path, repo: Path) -> dict:
    sys.path.insert(0, str(repo))
    import torch
    from sieve.config import SieveConfig
    from sieve.data import MemmapDataset
    from sieve.device import DeviceCfg
    from sieve.loop import train
    from sieve.rho1 import Rho1Baseline
    from sieve.state import SieveState

    cfg = SieveConfig(max_steps=STEPS, rescore_interval=RESCORE, eval_interval=EVAL, eval_batches=2,
                      log_interval=1, batch_size=B, seq_len=T, warmup_steps=5, selection_ratio=ALPHA,
                      seed=SEED, data_dir=str(data), freq_table_path=str(data / "freq.pt"),
                      sigma_warmup=2)
    if method == "vn_mix":
        cfg.variance_normalized_bandit = True
    if method == "mix4_bigram":
        cfg.sd_scorer, cfg.bigram_table_path = "bigram_freq", str(data / "bigram_freq.pt")
    if method == "mix4_noise":
        cfg.sd_scorer = "random_noise"
    dev = DeviceCfg(device=torch.device("cpu"), dtype=torch.float32, backend="cpu", use_amp=False)
    tr, va = MemmapDataset(str(data), "train", cfg), MemmapDataset(str(data), "val", cfg)
    model, ref = tiny_models()
    if method == "clm":
        sel = None
    elif method == "se_only":
        sel = Rho1Baseline(selection_ratio=ALPHA)
    elif method == "su_only":
        cfg.scorer_names = ["S_U"]
        sel = SieveState(cfg, torch.device("cpu"))
        def fixed(step, max_steps, prev_loss, prev_grad):
            sel.step, sel.weights = step, torch.ones(1)
        sel.step_begin = fixed
    else:
        sel = SieveState(cfg, torch.device("cpu"))
    masks = []
    if sel is not None:
        orig = sel.mask_cache.update
        sel.mask_cache.update = lambda s, step, a=None: masks.append(orig(s, step, a).clone()) or masks[-1]
    h = train(model, tr, va, cfg, dev, sieve=sel, ref_model=ref if sel is not None else None)
    return {"loss": h["loss"], "masks": [np.nonzero(m.numpy().ravel())[0].tolist() for m in masks],
            "samples": [s["weights"] for s in h.get("sample_log", [])],
            "rewards": [r["reward"] for r in h.get("reward_log", [])],
            "val_ppl": h["val_ppl"], "final_val_ppl": h["final_val_ppl"]}


def worker_new(method: str, data: Path) -> dict:
    sys.path.insert(0, str(NEW_ROOT))
    import torch
    from sieve.config import load_config
    from sieve.data import MemmapDataset
    from sieve.selection import build_selector
    from sieve.training import train

    c = NEW_ROOT / "configs"
    cfg = load_config([str(c / "base.yaml"), str(c / "methods" / f"{method}.yaml")], [
        f"seed={SEED}", f"data.dir={data}", f"data.seq_len={T}", f"data.batch_size={B}",
        f"training.steps={STEPS}", "training.warmup_steps=5", f"eval.interval={EVAL}",
        "eval.batches=2", f"selection.rescore_interval={RESCORE}", "output.log_interval=1",
        "selection.noise_seed=12345"] + ([] if method == "clm" else [f"selection.alpha={ALPHA}"]))
    cpu = torch.device("cpu")
    tr = MemmapDataset(str(data), "train", T, SEED)
    va = MemmapDataset(str(data), "val", T, SEED)
    model, ref = tiny_models()
    sel = build_selector(cfg, cpu)
    masks = []
    if sel is not None:
        orig = sel.select
        sel.select = lambda x: masks.append(orig(x).clone()) or masks[-1]
    h = train(model, sel, ref if sel is not None else None, tr, va, cfg, cpu)
    return {"loss": h["loss"], "masks": [np.nonzero(m.numpy().ravel())[0].tolist() for m in masks],
            "samples": [s["weights"] for s in h.get("sample_log", [])],
            "rewards": [r["reward"] for r in h.get("reward_log", [])],
            "val_ppl": h["val_ppl"], "final_val_ppl": h["final_val_ppl"]}


def compare(a: dict, b: dict) -> list[str]:
    errs = []
    if a["masks"] != b["masks"]:
        bad = next(i for i, (x, y) in enumerate(zip(a["masks"], b["masks"])) if x != y) \
            if len(a["masks"]) == len(b["masks"]) else "count"
        errs.append(f"masks differ (first at step {bad})")
    for k, tol in [("loss", 1e-5), ("val_ppl", 1e-4)]:
        if len(a[k]) != len(b[k]) or not np.allclose(a[k], b[k], rtol=tol, atol=tol):
            errs.append(f"{k} differ")
    if a["samples"]:                       # learning mixtures: same draws and rewards
        if len(a["samples"]) != len(b["samples"]) or not np.allclose(a["samples"], b["samples"], atol=1e-6):
            errs.append("sampled weights differ")
        if len(a["rewards"]) != len(b["rewards"]) or not np.allclose(a["rewards"], b["rewards"], atol=1e-6):
            errs.append("rewards differ")
    elif b["samples"] and not np.allclose(b["samples"], b["samples"][0]):
        # fixed weights: the reference did not log them; they must stay constant
        errs.append("fixed weights changed")
    if not np.isclose(a["final_val_ppl"], b["final_val_ppl"], rtol=1e-5):
        errs.append(f"final ppl {a['final_val_ppl']:.6f} vs {b['final_val_ppl']:.6f}")
    return errs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", help="checkout of the reference branch (fix-per-step-mask)")
    p.add_argument("--worker", choices=["old", "new"])
    p.add_argument("--method")
    p.add_argument("--data")
    p.add_argument("--out")
    args = p.parse_args()
    if args.worker:
        data = Path(args.data)
        res = worker_old(args.method, data, Path(args.reference)) if args.worker == "old" \
            else worker_new(args.method, data)
        Path(args.out).write_text(json.dumps(res))
        return

    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_data(tmp)
        for m in METHODS:
            res = {}
            for impl in ["old", "new"]:
                out = tmp / f"{m}_{impl}.json"
                cmd = [sys.executable, str(HERE), "--worker", impl, "--method", m, "--data", str(tmp),
                       "--out", str(out), "--reference", str(Path(args.reference).resolve())]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(r.stdout[-2000:], r.stderr[-4000:])
                    raise SystemExit(f"{impl} worker failed for {m}")
                res[impl] = json.loads(out.read_text())
            errs = compare(res["old"], res["new"])
            failed |= bool(errs)
            print(f"{m:12s} {'OK' if not errs else 'MISMATCH: ' + '; '.join(errs)}"
                  f"   (steps {len(res['new']['loss'])}, masks {len(res['new']['masks'])}, "
                  f"final ppl {res['new']['final_val_ppl']:.4f})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
