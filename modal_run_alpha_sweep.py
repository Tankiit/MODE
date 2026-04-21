"""
α-sweep orchestrator for SIEVE NeurIPS paper — single seed 42, wt103, gpt2l.

Matrix:
  sieve × α ∈ {0.3, 0.5, 0.7, 0.9} @ 25k steps     → 4 runs, Pareto curve
  rho1  × α ∈ {0.3, 0.5, 0.7, 0.9} @ 25k steps     → 4 runs, Pareto curve
  clm_A α=1.0 @ 17500 steps                         → 1 run, matched tokens ref @ α=0.7
  clm_B α=1.0 @ 25000 steps                         → 1 run, full budget ref

Total: 10 runs × ~$12 avg = ~$120. Wall: 2-3 batches × ~6hr = ~18-24hr.

Usage:
  modal run modal_run_alpha_sweep.py::run_sweep                     # fire all 10
  modal run modal_run_alpha_sweep.py::run_sweep --dry-run
  modal run modal_run_alpha_sweep.py::run_sweep --baselines sieve   # selectors only
"""
from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import modal

from modal_train_sieve import train_remote, app, volume_results


@dataclass(frozen=True)
class RunSpec:
    baseline: str           # "sieve" | "rho1" | "clm_A" | "clm_B"
    seed:     int
    steps:    int
    alpha:    float

    @property
    def name(self) -> str:
        a_tag = f"a{int(self.alpha*100)}"
        return (
            f"{self.baseline}_wikitext103_gpt2l_{a_tag}"
            f"_s{self.seed}_st{self.steps}"
        )

    def to_cli(self) -> str:
        actual = "clm" if self.baseline.startswith("clm") else self.baseline
        alpha_flag = f"--alpha {self.alpha}" if actual != "clm" else ""
        return (
            f"--dataset wikitext103 --model gpt2l "
            f"--steps {self.steps} "
            f"--baseline {actual} "
            f"--seed {self.seed} "
            f"{alpha_flag} "
            f"--eval_interval 1000 "
            f"--wandb sieve-neurips "
            f"--run_name {self.name}"
        )


def build_matrix(seed: int, baselines: set[str]) -> list[RunSpec]:
    """
    Build the sweep matrix. `baselines` filters which to include
    (for partial reruns if something fails mid-sweep).
    """
    ALPHAS = (0.30, 0.50, 0.70, 0.90)
    specs: list[RunSpec] = []

    if "sieve" in baselines:
        for a in ALPHAS:
            specs.append(RunSpec("sieve", seed, 25000, a))

    if "rho1" in baselines:
        for a in ALPHAS:
            specs.append(RunSpec("rho1", seed, 25000, a))

    if "clm" in baselines:
        # Matched tokens to α=0.7 selectors (71.7M tokens)
        specs.append(RunSpec("clm_A", seed, 17500, 1.0))
        # Full budget reference (102.4M tokens)
        specs.append(RunSpec("clm_B", seed, 25000, 1.0))

    return specs


@app.local_entrypoint()
def run_sweep(
    seed: int = 42,
    baselines: str = "sieve,rho1,clm",
    dry_run: bool = False,
    max_parallel: int = 4,
):
    baseline_set = {b.strip() for b in baselines.split(",")}
    specs = build_matrix(seed, baseline_set)

    print(f"[sweep] seed={seed}  baselines={sorted(baseline_set)}  "
          f"runs={len(specs)}")
    for s in specs:
        print(f"  • {s.name:60s}  steps={s.steps:>5}  α={s.alpha}")

    if dry_run:
        return

    sem = threading.Semaphore(max_parallel)
    results: list[dict] = []
    lock = threading.Lock()
    t0_all = time.time()

    def _run(s: RunSpec):
        with sem:
            t0 = time.time()
            r = None
            status = "failed"
            try:
                r = train_remote.remote(s.to_cli())
                status = r.get("status", "failed") if r else "failed"
            except Exception as e:
                r = {"error": str(e)}
            wall = time.time() - t0

        with lock:
            row = {
                "name": s.name,
                "baseline": s.baseline,
                "seed": s.seed,
                "steps": s.steps,
                "alpha": s.alpha,
                "status": status,
                "wall_seconds": wall,
                "best_ppl": (r or {}).get("best_ppl"),
                "final_val_ppl": (r or {}).get("final_val_ppl"),
                "best_step": (r or {}).get("best_step"),
                "scorer_overhead_pct": (r or {}).get("scorer_overhead_pct"),
                "max_weight": (r or {}).get("max_weight"),
            }
            results.append(row)
            best = row["best_ppl"] or "?"
            print(f"[{wall/60:.0f}min] {s.name:60s} best_ppl={best}  "
                  f"status={status}")

    threads = [threading.Thread(target=_run, args=(s,)) for s in specs]
    for t in threads: t.start()
    for t in threads: t.join()

    total_wall = (time.time() - t0_all) / 60
    print(f"\n[sweep] all runs done in {total_wall:.0f} min wall time")

    out_csv = Path("alpha_sweep_results.csv")
    if results:
        keys = list(results[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(sorted(results, key=lambda r: (r["baseline"], r["alpha"])))
        print(f"[sweep] aggregate → {out_csv.resolve()}")

    _print_pareto_table(results)


def _print_pareto_table(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("α SWEEP RESULTS — wikitext103, gpt2l, seed 42")
    print("=" * 72)

    # Group by baseline
    by_baseline: dict[str, list[dict]] = {}
    for r in results:
        by_baseline.setdefault(r["baseline"], []).append(r)

    # SIEVE and Rho-1 tables
    for b in ("sieve", "rho1"):
        rows = sorted(by_baseline.get(b, []), key=lambda r: r["alpha"])
        if not rows:
            continue
        print(f"\n{b.upper()} (α sweep):")
        print(f"  {'α':>5}  {'tokens_seen_M':>13}  {'best_ppl':>8}  "
              f"{'max_weight':>10}")
        for r in rows:
            a = r["alpha"]
            tokens_M = 25000 * 4 * 1024 * a / 1e6   # batch × seq × α
            bppl = r.get("best_ppl") or float("nan")
            mw = r.get("max_weight") or float("nan")
            print(f"  {a:>5.2f}  {tokens_M:>13.1f}  {bppl:>8.3f}  {mw:>10.3f}")

    # CLM references
    for b in ("clm_A", "clm_B"):
        rows = by_baseline.get(b, [])
        if rows:
            r = rows[0]
            tokens_M = r["steps"] * 4 * 1024 / 1e6
            bppl = r.get("best_ppl") or float("nan")
            label = "matched tokens (=sieve@α=0.7)" if b == "clm_A" else "full budget"
            print(f"\nCLM ({label}): tokens={tokens_M:.1f}M  best_ppl={bppl:.3f}")

    print("\n" + "=" * 72)
    print("Decision points:")
    print("  • SIEVE best α = argmin(best_ppl) across α sweep")
    print("  • SIEVE < Rho-1 at same α = multi-actor helps")
    print("  • SIEVE(best-α) < CLM_A at matched tokens = selection helps")
    print("  • SIEVE(best-α) ≈ CLM_B at 70-90% tokens = efficiency story")
    print("=" * 72)