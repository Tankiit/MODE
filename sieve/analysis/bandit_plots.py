"""
Mixture-dynamics figures and numbers (paper Fig. 1, Fig. 2, App. "Bandit Dynamics").

  python -m sieve.analysis.bandit_plots --runs runs --out figs

Uses the Mix-4 + bigram and Mix-4 + noise runs (default: GPT-2 Large,
WikiText-103, α = 0.7, 25k steps) and the expected weights E[w] = α/Σα
logged every output.log_interval steps. Writes
fig_seed_dominance_bandit_dynamics.png, fig_arm_trajectories_grid.pdf and
bandit_numbers.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .records import load_runs, select

VARIANTS = {"mix4_bigram": "bigram-substituted", "mix4_noise": "noise-substituted"}


def entropy(w: np.ndarray) -> float:
    w = np.clip(w, 1e-12, None)
    return float(-(w * np.log(w)).sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="figs")
    p.add_argument("--model", default="gpt2-large")
    p.add_argument("--data", default="wikitext103")
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--steps", type=int, default=25000)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {}
    for v in VARIANTS:
        for r in select(load_runs(args.runs), model=args.model, data=args.data, method=v,
                        alpha=args.alpha, steps=args.steps):
            h = r["history"]
            # logged E[w] plus the state after the last update (made at the final evaluation)
            steps = np.array(h["log_steps"] + [h["n_steps"]], float)
            weights = np.array(h["expected_weights"] + [h["expected_weights_final"]], float)
            runs[(v, r["seed"])] = (steps, weights, h["slots"])
    seeds = sorted({s for _, s in runs})
    if not seeds or any((v, s) not in runs for v in VARIANTS for s in seeds):
        raise SystemExit(f"need both variants for every seed; found {sorted(runs)}")
    slots = next(iter(runs.values()))[2]
    sd = slots.index("S_D")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def l1(a, b):
        (sa, wa, _), (_, wb, _) = runs[a], runs[b]
        n = min(len(wa), len(wb))
        return sa[:n], np.abs(wa[:n] - wb[:n]).sum(axis=1)

    same = {s: l1(("mix4_bigram", s), ("mix4_noise", s)) for s in seeds}
    cross = {(v, a, b): l1((v, a), (v, b)) for v in VARIANTS for a, b in itertools.combinations(seeds, 2)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for s, (st, d) in same.items():
        ax1.plot(st, d, lw=1.8, label=f"seed {s}: bigram vs noise")
    for (v, a, b), (st, d) in cross.items():
        ax1.plot(st, d, lw=0.9, ls="--", alpha=0.7, label=f"{v.split('_')[1]}: seed {a} vs {b}")
    ax1.set_xlabel("training step")
    ax1.set_ylabel(r"$L_1$ distance between $\mathbb{E}[w]$")
    ax1.legend(fontsize=6, ncol=2)
    for i, (v, lab) in enumerate(VARIANTS.items()):
        ax2.scatter([s + (i - 0.5) * 0.15 for s in seeds], [runs[(v, s)][1][-1, sd] for s in seeds],
                    label=lab, s=40)
    ax2.axhline(1 / len(slots), ls=":", c="gray")
    ax2.set_xticks(seeds)
    ax2.set_xlabel("seed")
    ax2.set_ylabel(r"terminal $\mathbb{E}[w_{S_D}]$")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_seed_dominance_bandit_dynamics.png", dpi=200)

    fig, axes = plt.subplots(len(seeds), 2, figsize=(9, 2.4 * len(seeds)), sharex=True, sharey=True,
                             squeeze=False)
    for r, s in enumerate(seeds):
        for c, (v, lab) in enumerate(VARIANTS.items()):
            st, w, _ = runs[(v, s)]
            for k, nm in enumerate(slots):
                axes[r][c].plot(st, w[:, k], label=nm)
            axes[r][c].axhline(1 / len(slots), ls=":", c="gray")
            axes[r][c].set_title(f"seed {s} — {lab}", fontsize=9)
    axes[0][0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("training step")
    fig.tight_layout()
    fig.savefig(out / "fig_arm_trajectories_grid.pdf")

    def first_above(st, d, thr=0.05):
        i = np.nonzero(d > thr)[0]
        return float(st[i[0]]) if len(i) else None

    fs = {s: float(d[-1]) for s, (_, d) in same.items()}
    fc = {f"{v}:{a}-{b}": float(d[-1]) for (v, a, b), (_, d) in cross.items()}
    nums = {
        "same_seed_final_L1": fs,
        "same_seed_final_L1_min_max_mean": [min(fs.values()), max(fs.values()), float(np.mean(list(fs.values())))],
        "same_seed_first_step_L1_above_0.05": {s: first_above(st, d) for s, (st, d) in same.items()},
        "cross_seed_final_L1": fc,
        "cross_seed_final_L1_min_max_mean": [min(fc.values()), max(fc.values()), float(np.mean(list(fc.values())))],
        "terminal_entropy": {f"{v}:{s}": entropy(runs[(v, s)][1][-1]) for v in VARIANTS for s in seeds},
        "max_entropy": float(np.log(len(slots))),
        "terminal_weights": {f"{v}:{s}": dict(zip(slots, map(float, runs[(v, s)][1][-1])))
                             for v in VARIANTS for s in seeds},
    }
    (out / "bandit_numbers.json").write_text(json.dumps(nums, indent=2))
    print(json.dumps(nums, indent=2))
    print(f"figures → {out}/")


if __name__ == "__main__":
    main()
