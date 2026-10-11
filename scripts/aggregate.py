#!/usr/bin/env python3
"""
Collect runs/*/history.json into the paper's result tables (RERUN_PLAN.md §7).

Writes to --out (default: results/):
  runs.csv     one row per finished run
  table1.md    Table 1  (GPT-2 Large, α = 0.7, 9 variants × 2 corpora)
  table2.md    Table 2  (GPT-2 Large, α sweep for S_E-only, Mix-4, VN-Mix)
  paired.md    Mix-4 − S_E-only, seed-matched, for every model/corpus/α/steps
  matched.md   matched-token control (GPT-2 Small, WikiText-103, group C)
  compute.md   wall time per run and total GPU-hours

Values are mean ± sample standard deviation (ddof = 1) over seeds, with the
number of seeds; cells with fewer than 3 seeds are flagged.
Usage: python scripts/aggregate.py --runs runs --out results
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TABLE1_VARIANTS = ["seonly", "su_only", "sl_only", "sd_only", "mix4", "mix3",
                   "vnmix", "mix4_bigram", "mix4_noise"]
LABELS = {"seonly": "S_E-only", "su_only": "S_U only", "sl_only": "S_L only",
          "sd_only": "S_D only", "mix4": "Mix-4", "mix3": "Mix-3 (drop S_E)",
          "vnmix": "VN-Mix", "mix4_bigram": "Mix-4 + bigram(S_D)",
          "mix4_noise": "Mix-4 + noise(S_D)", "clm": "Full CLM (α = 1)"}
DATASETS = ["wikitext103", "owm"]


def variant_of(args: dict) -> str:
    b = args["baseline"]
    if b == "rho1":
        return "seonly"
    if b == "sieve_no_se":
        return "mix3"
    if b == "sieve":
        if args.get("variance_normalized_bandit"):
            return "vnmix"
        sd = args.get("sd_scorer", "inv_freq")
        return {"inv_freq": "mix4", "bigram_freq": "mix4_bigram",
                "random_noise": "mix4_noise"}.get(sd, f"mix4_{sd}")
    return b


def load_runs(runs_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(runs_dir.glob("*/history.json")):
        rec = json.loads(f.read_text())
        a, h = rec["args"], rec["history"]
        rows.append({
            "run": f.parent.name,
            "group": (a.get("run_name") or "").split("_")[0],
            "model": a["model"], "dataset": a["dataset"], "variant": variant_of(a),
            "alpha": float(a["alpha"]), "steps": int(a["steps"]), "seed": int(a["seed"]),
            "final_val_ppl": float(h["final_val_ppl"]),
            "n_steps": int(h.get("n_steps", a["steps"])),
            "wall_time_h": float(h.get("wall_time_s", float("nan"))) / 3600.0,
        })
    return rows


def cell(rows, **key) -> list[dict]:
    return [r for r in rows if all(r[k] == v for k, v in key.items())]


def fmt(rs: list[dict]) -> str:
    if not rs:
        return "---"
    v = np.array([r["final_val_ppl"] for r in rs])
    if len(v) == 1:
        return f"{v[0]:.3f} (n=1)"
    s = f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"
    return s if len(v) >= 3 else s + f" (n={len(v)})"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_runs(Path(args.runs))
    if not rows:
        raise SystemExit(f"no history.json under {args.runs}")

    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Table 1 — GPT-2 Large, α = 0.7
    t1 = ["| Selection method | WikiText-103 | OpenWebMath |", "|---|---|---|"]
    for v in TABLE1_VARIANTS:
        cells = [fmt(cell(rows, model="gpt2l", dataset=d, variant=v, alpha=0.7, steps=25000))
                 for d in DATASETS]
        t1.append(f"| {LABELS[v]} | " + " | ".join(cells) + " |")
    (out / "table1.md").write_text("\n".join(t1) + "\n")

    # Table 2 — GPT-2 Large α sweep
    alphas = [0.3, 0.5, 0.7, 0.9]
    t2 = ["| Corpus | Method | " + " | ".join(str(a) for a in alphas) + " |",
          "|---|---|" + "---|" * len(alphas)]
    for d in DATASETS:
        for v in ["seonly", "mix4", "vnmix"]:
            cells = [fmt(cell(rows, model="gpt2l", dataset=d, variant=v, alpha=a, steps=25000))
                     for a in alphas]
            t2.append(f"| {d} | {LABELS[v]} | " + " | ".join(cells) + " |")
    (out / "table2.md").write_text("\n".join(t2) + "\n")

    # Paired Mix-4 − S_E-only, seed-matched
    pr = ["| Model | Corpus | α | Steps | Mix-4 − S_E-only by seed | max abs |", "|---|---|---|---|---|---|"]
    keys = sorted({(r["model"], r["dataset"], r["alpha"], r["steps"]) for r in rows})
    for m, d, a, st in keys:
        mix = {r["seed"]: r["final_val_ppl"] for r in cell(rows, model=m, dataset=d, alpha=a, steps=st, variant="mix4")}
        se = {r["seed"]: r["final_val_ppl"] for r in cell(rows, model=m, dataset=d, alpha=a, steps=st, variant="seonly")}
        seeds = sorted(set(mix) & set(se))
        if not seeds:
            continue
        diffs = [mix[s] - se[s] for s in seeds]
        pr.append(f"| {m} | {d} | {a} | {st} | "
                  + ", ".join(f"s{s}: {x:+.3f}" for s, x in zip(seeds, diffs))
                  + f" | {max(abs(x) for x in diffs):.3f} |")
    (out / "paired.md").write_text("\n".join(pr) + "\n")

    # Matched-token control — group C
    mc = ["| α | Steps | S_E-only | Mix-4 | Full CLM |", "|---|---|---|---|---|"]
    c_rows = [r for r in rows if r["group"] == "C"]
    for a, st in sorted({(r["alpha"], r["steps"]) for r in c_rows}):
        mc.append(f"| {a} | {st} | " + " | ".join(
            fmt(cell(c_rows, alpha=a, steps=st, variant=v)) for v in ["seonly", "mix4", "clm"]) + " |")
    (out / "matched.md").write_text("\n".join(mc) + "\n")

    # Compute
    comp = ["| Model | Corpus | Runs | Mean wall time (h) | Total (GPU-h) |", "|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["dataset"])].append(r["wall_time_h"])
    for (m, d), ts in sorted(by.items()):
        comp.append(f"| {m} | {d} | {len(ts)} | {np.nanmean(ts):.2f} | {np.nansum(ts):.1f} |")
    (out / "compute.md").write_text("\n".join(comp) + "\n")

    for name in ["table1", "table2", "paired", "matched", "compute"]:
        print(f"\n## {name}\n" + (out / f"{name}.md").read_text())
    print(f"{len(rows)} runs collected → {out}/")


if __name__ == "__main__":
    main()
