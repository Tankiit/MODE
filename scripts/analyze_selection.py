#!/usr/bin/env python3
"""
Selection-mask overlap and scorer correlations (paper App. "Selection-Mask
Overlap" and "Scorer Correlation Matrices").

Needs the final checkpoints written with --save_final (RERUN_PLAN.md §5):
  Jaccard      S_E-only, S_U, S_L, S_D only and Mix-4, one corpus/seed/α
  Correlations Mix-4 on each corpus

Fixed evaluation batch: the first --batch sequences of val.bin (length
seq_len, non-overlapping). Each variant scores this batch with its own final
model; Mix-4 uses its final posterior mean weights E[w] = α/Σα. Masks are the
per-sequence top-α positions (MaskCache). Spearman correlations use the raw
scorer values of the first --n_tokens scored positions (column 0 excluded),
computed with the final Mix-4 model of each corpus.

Writes jaccard.md, correlations.md and selection_analysis.json to --out.
Usage: python scripts/analyze_selection.py --runs runs --out results
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from aggregate import variant_of  # noqa: E402
from sieve.mask import MaskCache  # noqa: E402
from sieve.reference import load_reference, reference_nll  # noqa: E402
from sieve.scorers import TokenScorers  # noqa: E402
from train_sieve import build_model  # noqa: E402

JACCARD_VARIANTS = ["seonly", "su_only", "sl_only", "sd_only", "mix4"]
SCORERS = ["S_E", "S_U", "S_L", "S_D"]


def find(runs_dir: Path, model, dataset, alpha, seed, variant, steps=25000):
    for f in sorted(runs_dir.glob("*/history.json")):
        rec = json.loads(f.read_text())
        a = rec["args"]
        if (variant_of(a) == variant and a["model"] == model and a["dataset"] == dataset
                and float(a["alpha"]) == alpha and int(a["seed"]) == seed
                and int(a["steps"]) == steps and (f.parent / "final.pt").exists()):
            return f.parent, rec
    raise SystemExit(f"no final.pt for {variant} {model} {dataset} α={alpha} seed={seed}")


def val_batch(data_dir: Path, n_seq: int, seq_len: int, device) -> torch.Tensor:
    arr = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    ids = np.asarray(arr[: n_seq * seq_len], dtype=np.int64).reshape(n_seq, seq_len)
    return torch.from_numpy(ids).to(device)


@torch.no_grad()
def raw_scores(run_dir: Path, rec: dict, ids, ref_nll, freq, device) -> dict:
    """Raw [B, T] values of the four scorers under this run's final model."""
    model = build_model(rec["args"]["model"], ids.shape[1], device)
    model.load_state_dict(torch.load(run_dir / "final.pt", map_location=device))
    model.eval()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(input_ids=ids, use_cache=False).logits.float()
    sc = TokenScorers(freq_table=freq, scorer_names=SCORERS)
    out = {"S_E": sc.score_excess_loss(logits, ids, ref_nll),
           "S_U": sc.score_entropy(logits),
           "S_L": sc.score_logit_spread(logits),
           "S_D": sc.score_inv_freq(ids, device)}
    del model, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def utility(raw: dict, names: list[str], weights: list[float]) -> torch.Tensor:
    """Same z-normalisation and weighting as TokenScorers.score_all."""
    stack = torch.stack([raw[n] for n in names], dim=-1)
    inner = stack[:, 1:]
    z = (stack - inner.mean(dim=(0, 1), keepdim=True)) / inner.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    return (z * torch.tensor(weights, device=z.device).view(1, 1, -1)).sum(-1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    p.add_argument("--model", default="gpt2l")
    p.add_argument("--jaccard_dataset", default="wikitext103")
    p.add_argument("--corr_datasets", nargs="+", default=["wikitext103", "owm"])
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=25000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n_tokens", type=int, default=1000)
    p.add_argument("--ref_model", default="gpt2")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = load_reference(args.ref_model, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {}

    def inputs(rec):
        data_dir = Path(rec["config"]["data_dir"])
        ids = val_batch(data_dir, args.batch, rec["config"]["seq_len"], device)
        ac = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
        freq = torch.load(data_dir / "freq.pt", map_location="cpu")
        return ids, reference_nll(ref, ids, ac), freq

    # ── Jaccard ───────────────────────────────────────────────────────
    masks = {}
    for v in JACCARD_VARIANTS:
        run_dir, rec = find(Path(args.runs), args.model, args.jaccard_dataset, args.alpha, args.seed, v, args.steps)
        ids, ref_nll, freq = inputs(rec)
        raw = raw_scores(run_dir, rec, ids, ref_nll, freq, device)
        if v == "mix4":
            names = rec["history"]["scorer_names"]
            u = utility(raw, names, rec["history"]["expected_weights_final"])
        else:
            u = utility(raw, [{"seonly": "S_E", "su_only": "S_U", "sl_only": "S_L",
                               "sd_only": "S_D"}[v]], [1.0])
        masks[v] = MaskCache(args.alpha).update(u, 0)[:, 1:].cpu()
    jac = {}
    for a, b in itertools.combinations(JACCARD_VARIANTS, 2):
        inter = (masks[a] & masks[b]).sum().item()
        union = (masks[a] | masks[b]).sum().item()
        jac[f"{a}|{b}"] = inter / union
    result["jaccard"] = jac
    result["jaccard_independent_baseline"] = args.alpha / (2 - args.alpha)
    lines = ["| | " + " | ".join(JACCARD_VARIANTS) + " |", "|---" * (len(JACCARD_VARIANTS) + 1) + "|"]
    for i, a in enumerate(JACCARD_VARIANTS):
        row = [""] * i + ["1.000"] + [f"{jac[f'{a}|{b}']:.3f}" for b in JACCARD_VARIANTS[i + 1:]]
        lines.append(f"| {a} | " + " | ".join(row) + " |")
    lines.append(f"\nIndependent-selection baseline α/(2−α) = {args.alpha / (2 - args.alpha):.3f}")
    (out / "jaccard.md").write_text("\n".join(lines) + "\n")

    # ── Spearman correlations ─────────────────────────────────────────
    corr = {}
    for d in args.corr_datasets:
        run_dir, rec = find(Path(args.runs), args.model, d, args.alpha, args.seed, "mix4", args.steps)
        ids, ref_nll, freq = inputs(rec)
        raw = raw_scores(run_dir, rec, ids, ref_nll, freq, device)
        flat = {n: raw[n][:, 1:].reshape(-1)[: args.n_tokens].float().cpu().numpy() for n in SCORERS}
        corr[d] = {f"{a}|{b}": spearman(flat[a], flat[b]) for a, b in itertools.combinations(SCORERS, 2)}
    result["spearman"] = corr
    lines = ["| Scorer pair | " + " | ".join(args.corr_datasets) + " |",
             "|---|" + "---|" * len(args.corr_datasets)]
    for a, b in itertools.combinations(SCORERS, 2):
        lines.append(f"| ρ({a}, {b}) | " + " | ".join(f"{corr[d][f'{a}|{b}']:.3f}" for d in args.corr_datasets) + " |")
    (out / "correlations.md").write_text("\n".join(lines) + "\n")

    (out / "selection_analysis.json").write_text(json.dumps(result, indent=2))
    print((out / "jaccard.md").read_text())
    print((out / "correlations.md").read_text())


if __name__ == "__main__":
    main()
