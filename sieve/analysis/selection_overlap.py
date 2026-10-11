"""
Selection-mask overlap and scorer correlations (paper App. "Selection-Mask
Overlap" and "Scorer Correlation Matrices").

  python -m sieve.analysis.selection_overlap --runs runs --out results

Needs the final checkpoints (output.save_final) of: S_E-only, S_U, S_L,
S_D only and Mix-4 on --jaccard_data, and Mix-4 on every --corr_data
(same model, α, steps and seed).

A fixed validation batch (the first --batch windows of val.bin) is scored by
each run's final model with its own scorers and weights (Mix-4: final
posterior mean E[w]); masks are the per-sequence top-k. Spearman correlations
use the raw values of the four scorers at the first --n_tokens scored
positions (column 0 excluded), under the final Mix-4 model of each corpus.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from ..data import MemmapDataset
from ..models import autocast, build_model, load_reference, reference_nll
from ..scorers import ScoringInputs, build_scorer
from ..selection import mixed_utility, topk_mask
from .records import load_runs, select

JACCARD_METHODS = ["se_only", "su_only", "sl_only", "sd_only", "mix4"]
FOUR = ["excess_loss", "entropy", "logit_margin", "frequency"]


def checkpoint(records, **key) -> dict:
    found = [r for r in select(records, **key) if (r["dir"] / "final.pt").exists()]
    if not found:
        raise SystemExit(f"no run with final.pt for {key}")
    return found[0]


@torch.no_grad()
def score_with(rec: dict, names: list[str], ids: torch.Tensor, ref, device) -> torch.Tensor:
    """Raw [B, T, K] values of `names` under this run's final model."""
    cfg = rec["config"]
    m = cfg["model"]
    model = build_model(m["name"], cfg["data"]["seq_len"], device, m["tiny_layers"], m["tiny_dim"],
                        m["tiny_heads"])
    model.load_state_dict(torch.load(rec["dir"] / "final.pt", map_location=device))
    model.eval()
    with autocast(device, cfg["training"]["autocast"]):
        logits = model(input_ids=ids, use_cache=False).logits.float()
    del model
    ref_nll = reference_nll(ref, ids, autocast(device, cfg["training"]["autocast"]))
    x = ScoringInputs(ids, logits, ref_nll)
    return torch.stack([build_scorer(n, cfg["data"]["dir"], cfg["seed"], device).score(x) for n in names], -1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    p.add_argument("--model", default="gpt2-large")
    p.add_argument("--jaccard_data", default="wikitext103")
    p.add_argument("--corr_data", nargs="+", default=["wikitext103", "owm"])
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--steps", type=int, default=25000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n_tokens", type=int, default=1000)
    args = p.parse_args()

    from scipy.stats import spearmanr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rs = load_runs(args.runs)
    key = dict(model=args.model, alpha=args.alpha, steps=args.steps, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    refs: dict = {}

    def batch_for(rec):
        cfg = rec["config"]
        ds = MemmapDataset(cfg["data"]["dir"], "val", cfg["data"]["seq_len"], cfg["seed"])
        ids = next(ds.val_batches(args.batch, device, 1))
        name = cfg["selection"]["reference_model"]
        refs.setdefault(name, load_reference(name, device))
        return ids, refs[name]

    # ── Jaccard ───────────────────────────────────────────────────────
    masks = {}
    for m in JACCARD_METHODS:
        rec = checkpoint(rs, data=args.jaccard_data, method=m, **key)
        ids, ref = batch_for(rec)
        names = rec["config"]["selection"]["scorers"]
        raw = score_with(rec, names, ids, ref, device)
        w = rec["history"]["expected_weights_final"]
        masks[m] = topk_mask(mixed_utility(raw, torch.tensor(w, dtype=torch.float32)), args.alpha)[:, 1:].cpu()
    jac = {f"{a}|{b}": float((masks[a] & masks[b]).sum() / (masks[a] | masks[b]).sum())
           for a, b in itertools.combinations(JACCARD_METHODS, 2)}
    lines = ["| | " + " | ".join(JACCARD_METHODS) + " |", "|---" * (len(JACCARD_METHODS) + 1) + "|"]
    for i, a in enumerate(JACCARD_METHODS):
        row = [""] * i + ["1.000"] + [f"{jac[f'{a}|{b}']:.3f}" for b in JACCARD_METHODS[i + 1:]]
        lines.append(f"| {a} | " + " | ".join(row) + " |")
    lines.append(f"\nIndependent-selection baseline α/(2−α) = {args.alpha / (2 - args.alpha):.3f}")
    (out / "jaccard.md").write_text("\n".join(lines) + "\n")

    # ── Spearman ──────────────────────────────────────────────────────
    corr = {}
    for d in args.corr_data:
        rec = checkpoint(rs, data=d, method="mix4", **key)
        ids, ref = batch_for(rec)
        raw = score_with(rec, FOUR, ids, ref, device)[:, 1:].reshape(-1, 4)[: args.n_tokens].cpu().numpy()
        corr[d] = {f"{a}|{b}": float(spearmanr(raw[:, i], raw[:, j]).correlation)
                   for (i, a), (j, b) in itertools.combinations(enumerate(FOUR), 2)}
    lines = ["| Scorer pair | " + " | ".join(args.corr_data) + " |", "|---|" + "---|" * len(args.corr_data)]
    for a, b in itertools.combinations(FOUR, 2):
        lines.append(f"| ρ({a}, {b}) | " + " | ".join(f"{corr[d][f'{a}|{b}']:.3f}" for d in args.corr_data) + " |")
    (out / "correlations.md").write_text("\n".join(lines) + "\n")

    (out / "selection_analysis.json").write_text(json.dumps({"jaccard": jac, "spearman": corr}, indent=2))
    print((out / "jaccard.md").read_text())
    print((out / "correlations.md").read_text())


if __name__ == "__main__":
    main()
