#!/usr/bin/env python3
"""
Bandit-dynamics figures and numbers (paper Fig. 1, Fig. 2, App. "Bandit Dynamics").

Uses the Mix-4 + bigram(S_D) and Mix-4 + noise(S_D) runs (GPT-2 Large,
WikiText-103, α = 0.7, 25k steps, seeds 42/43/44) and their history.json
(expected weights E[w] = α/Σα at every log step).

Writes to --out (default: figs/):
  fig_seed_dominance_bandit_dynamics.png   Fig. 1: L1 distances + terminal E[w_SD]
  fig_arm_trajectories_grid.pdf            Fig. 2: per-arm trajectories (seeds × variants)
  bandit_numbers.json                      numbers quoted in the appendix
Usage: python scripts/plot_bandit.py --runs runs --out figs
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import variant_of  # noqa: E402

VARIANTS = {"mix4_bigram": "bigram-substituted", "mix4_noise": "noise-substituted"}


def load(runs_dir: Path, model, dataset, alpha, steps) -> dict:
    """{(variant, seed): (steps array, weights [n, K], scorer names)}"""
    out = {}
    for f in sorted(runs_dir.glob("*/history.json")):
        rec = json.loads(f.read_text())
        a, h = rec["args"], rec["history"]
        v = variant_of(a)
        if (v in VARIANTS and a["model"] == model and a["dataset"] == dataset
                and float(a["alpha"]) == alpha and int(a["steps"]) == steps):
            w = np.array(h["weights"], dtype=float)
            st = np.array(h["log_steps"][:len(w)], dtype=float)
            out[(v, int(a["seed"]))] = (st, w, h.get("scorer_names", ["S_E", "S_U", "S_L", "S_D"]))
    return out


def entropy(w: np.ndarray) -> float:
    w = np.clip(w, 1e-12, None)
    return float(-(w * np.log(w)).sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="figs")
    p.add_argument("--model", default="gpt2l")
    p.add_argument("--dataset", default="wikitext103")
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--steps", type=int, default=25000)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load(Path(args.runs), args.model, args.dataset, args.alpha, args.steps)
    seeds = sorted({s for _, s in runs})
    if not runs or any((v, s) not in runs for v in VARIANTS for s in seeds):
        raise SystemExit(f"missing runs: found {sorted(runs)}")
    names = next(iter(runs.values()))[2]
    sd = names.index("S_D")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def l1(a, b):
        sa, wa, _ = runs[a]
        sb, wb, _ = runs[b]
        n = min(len(wa), len(wb))
        return sa[:n], np.abs(wa[:n] - wb[:n]).sum(axis=1)

    same_seed = {s: l1(("mix4_bigram", s), ("mix4_noise", s)) for s in seeds}
    cross_seed = {(v, s1, s2): l1((v, s1), (v, s2))
                  for v in VARIANTS for s1, s2 in itertools.combinations(seeds, 2)}

    # ── Fig. 1 ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for s, (st, d) in same_seed.items():
        ax1.plot(st, d, lw=1.8, label=f"same seed {s} (bigram vs noise)")
    for (v, s1, s2), (st, d) in cross_seed.items():
        ax1.plot(st, d, lw=0.9, ls="--", alpha=0.7,
                 label=f"{VARIANTS[v].split('-')[0]}: seed {s1} vs {s2}")
    ax1.set_xlabel("training step")
    ax1.set_ylabel(r"$L_1$ distance between $\mathbb{E}[w]$")
    ax1.legend(fontsize=6, ncol=2)
    for i, (v, lab) in enumerate(VARIANTS.items()):
        ax2.scatter([s + (i - 0.5) * 0.15 for s in seeds],
                    [runs[(v, s)][1][-1, sd] for s in seeds], label=lab, s=40)
    ax2.axhline(1 / len(names), ls=":", c="gray")
    ax2.set_xticks(seeds)
    ax2.set_xlabel("seed")
    ax2.set_ylabel(r"terminal $\mathbb{E}[w_{S_D}]$")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_seed_dominance_bandit_dynamics.png", dpi=200)

    # ── Fig. 2 ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(seeds), 2, figsize=(9, 2.4 * len(seeds)),
                             sharex=True, sharey=True, squeeze=False)
    for r, s in enumerate(seeds):
        for c, (v, lab) in enumerate(VARIANTS.items()):
            st, w, _ = runs[(v, s)]
            ax = axes[r][c]
            for k, nm in enumerate(names):
                ax.plot(st, w[:, k], label=nm)
            ax.axhline(1 / len(names), ls=":", c="gray")
            ax.set_title(f"seed {s} — {lab}", fontsize=9)
    axes[0][0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("training step")
    fig.tight_layout()
    fig.savefig(out / "fig_arm_trajectories_grid.pdf")

    # ── Numbers for the appendix ──────────────────────────────────────
    def first_above(st, d, thr=0.05):
        idx = np.nonzero(d > thr)[0]
        return float(st[idx[0]]) if len(idx) else None

    final_same = {s: float(d[-1]) for s, (_, d) in same_seed.items()}
    final_cross = {f"{v}:{s1}-{s2}": float(d[-1]) for (v, s1, s2), (_, d) in cross_seed.items()}
    nums = {
        "same_seed_final_L1": final_same,
        "same_seed_final_L1_range_mean": [min(final_same.values()), max(final_same.values()),
                                          float(np.mean(list(final_same.values())))],
        "same_seed_first_step_L1_above_0.05": {s: first_above(st, d) for s, (st, d) in same_seed.items()},
        "cross_seed_final_L1": final_cross,
        "cross_seed_final_L1_range_mean": [min(final_cross.values()), max(final_cross.values()),
                                           float(np.mean(list(final_cross.values())))],
        "terminal_entropy": {f"{v}:{s}": entropy(runs[(v, s)][1][-1]) for v in VARIANTS for s in seeds},
        "max_entropy": float(np.log(len(names))),
        "terminal_weights": {f"{v}:{s}": dict(zip(names, map(float, runs[(v, s)][1][-1])))
                             for v in VARIANTS for s in seeds},
    }
    (out / "bandit_numbers.json").write_text(json.dumps(nums, indent=2))
    print(json.dumps(nums, indent=2))
    print(f"figures → {out}/")


if __name__ == "__main__":
    main()
