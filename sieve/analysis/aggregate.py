"""
Result tables of the paper from completed runs (RERUN_PLAN.md §7).

  python -m sieve.analysis.aggregate --runs runs --out results

Writes runs.csv and table1.md (GPT-2 Large, α = 0.7), table2.md (α sweep),
paired.md (Mix-4 − S_E-only by seed), matched.md (group C) and compute.md.
Cells are mean ± sample standard deviation (ddof = 1) over seeds; cells
with fewer than 3 seeds show n.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .records import load_runs, select

TABLE1 = ["se_only", "su_only", "sl_only", "sd_only", "mix4", "mix3", "vn_mix",
          "mix4_bigram", "mix4_noise"]
LABELS = {"se_only": "S_E-only", "su_only": "S_U only", "sl_only": "S_L only",
          "sd_only": "S_D only", "mix4": "Mix-4", "mix3": "Mix-3 (drop S_E)",
          "vn_mix": "VN-Mix", "mix4_bigram": "Mix-4 + bigram(S_D)",
          "mix4_noise": "Mix-4 + noise(S_D)", "clm": "Full CLM"}
CORPORA = ["wikitext103", "owm"]
LARGE, STEPS = "gpt2-large", 25000


def fmt(rs: list[dict]) -> str:
    if not rs:
        return "---"
    v = np.array([r["final_val_ppl"] for r in rs])
    if len(v) == 1:
        return f"{v[0]:.3f} (n=1)"
    s = f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"
    return s if len(v) >= 3 else f"{s} (n={len(v)})"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    rs = load_runs(args.runs)
    if not rs:
        raise SystemExit(f"no completed run under {args.runs}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["dir", "group", "model", "data", "method", "alpha", "steps", "seed",
            "final_val_ppl", "wall_time_h"]
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(rs, key=lambda r: str(r["dir"])):
            w.writerow([r[c] for c in cols])

    t1 = ["| Selection method | WikiText-103 | OpenWebMath |", "|---|---|---|"]
    for m in TABLE1:
        t1.append(f"| {LABELS[m]} | " + " | ".join(
            fmt(select(rs, model=LARGE, data=d, method=m, alpha=0.7, steps=STEPS)) for d in CORPORA) + " |")

    alphas = [0.3, 0.5, 0.7, 0.9]
    t2 = ["| Corpus | Method | " + " | ".join(map(str, alphas)) + " |", "|---|---|" + "---|" * len(alphas)]
    for d in CORPORA:
        for m in ["se_only", "mix4", "vn_mix"]:
            t2.append(f"| {d} | {LABELS[m]} | " + " | ".join(
                fmt(select(rs, model=LARGE, data=d, method=m, alpha=a, steps=STEPS)) for a in alphas) + " |")

    pr = ["| Model | Corpus | α | Steps | Mix-4 − S_E-only by seed | max abs |", "|---|---|---|---|---|---|"]
    for mo, d, a, st in sorted({(r["model"], r["data"], r["alpha"], r["steps"]) for r in rs
                                if r["alpha"] is not None}):
        mix = {r["seed"]: r["final_val_ppl"] for r in select(rs, model=mo, data=d, alpha=a, steps=st, method="mix4")}
        se = {r["seed"]: r["final_val_ppl"] for r in select(rs, model=mo, data=d, alpha=a, steps=st, method="se_only")}
        seeds = sorted(set(mix) & set(se))
        if seeds:
            diffs = [mix[s] - se[s] for s in seeds]
            pr.append(f"| {mo} | {d} | {a} | {st} | " + ", ".join(f"s{s}: {x:+.3f}" for s, x in zip(seeds, diffs))
                      + f" | {max(map(abs, diffs)):.3f} |")

    c = select(rs, group="C")
    mc = ["| α | Steps | S_E-only | Mix-4 | Full CLM |", "|---|---|---|---|---|"]
    for a, st in sorted({(r["alpha"] if r["alpha"] is not None else 1.0, r["steps"]) for r in c}):
        alpha = None if a == 1.0 else a
        mc.append(f"| {a} | {st} | " + " | ".join(
            fmt(select(c, alpha=alpha, steps=st, method=m)) for m in ["se_only", "mix4", "clm"]) + " |")

    comp = ["| Model | Corpus | Runs | Mean wall time (h) | Total (GPU-h) |", "|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rs:
        by[(r["model"], r["data"])].append(r["wall_time_h"])
    for (mo, d), ts in sorted(by.items()):
        comp.append(f"| {mo} | {d} | {len(ts)} | {np.mean(ts):.2f} | {np.sum(ts):.1f} |")

    for name, lines in [("table1", t1), ("table2", t2), ("paired", pr), ("matched", mc), ("compute", comp)]:
        (out / f"{name}.md").write_text("\n".join(lines) + "\n")
        print(f"\n## {name}\n" + "\n".join(lines))
    print(f"\n{len(rs)} completed configurations → {out}/")


if __name__ == "__main__":
    main()
