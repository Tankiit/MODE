"""
Fire the full wt103 gate — 12 runs total.

Matrix:
  baselines: sieve, rho1, clm_A (17.5k steps), clm_B (25k steps)
  seeds:     42, 123, 456
  = 12 runs

Framings (collapsed):
  A (matched tokens, 71.7M seen):  sieve, rho1, clm_A
  B (matched steps, 25k):          sieve, rho1, clm_B

SIEVE and Rho-1 runs are reused across both framings — only CLM runs twice.

Budget: ~$243 + ~$7 pilot = ~$250 total.
Wall time with max_containers=4: ~20-24 hours (overnight).

Usage:
  modal run modal_run_gate.py                    # fire all 12
  modal run modal_run_gate.py --seeds 42         # fire only seed 42 (4 runs)
  modal run modal_run_gate.py --dry-run          # preview without spending
"""
from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import modal

# Import the training function from the existing wrapper.
# This reuses ALL the infrastructure you already have
# (image, volumes, secret, checkpointing, log parsing).
from modal_train_sieve import train_remote, app, volume_results

Baseline = Literal["sieve", "rho1", "clm_A", "clm_B"]


@dataclass(frozen=True)
class RunSpec:
    baseline: Baseline
    seed:     int
    steps:    int
    alpha:    float

    @property
    def name(self) -> str:
        """Must be unique — distinguishes CLM_A from CLM_B."""
        # underlying CLI baseline is always "clm" for both clm_A/clm_B;
        # the name distinguishes them for directory and wandb purposes
        return f"{self.baseline}_wikitext103_gpt2l_a{int(self.alpha*100)}_s{self.seed}_st{self.steps}"

    def to_cli(self) -> str:
        # Map clm_A/clm_B back to the underlying baseline the training script knows
        actual_baseline = "clm" if self.baseline.startswith("clm") else self.baseline
        alpha_flag = f"--alpha {self.alpha}" if actual_baseline != "clm" else ""
        return (
            f"--dataset wikitext103 --model gpt2l "
            f"--steps {self.steps} "
            f"--baseline {actual_baseline} "
            f"--seed {self.seed} "
            f"{alpha_flag} "
            f"--eval_interval 1000 "
            f"--wandb sieve-neurips "
            f"--run_name {self.name}"
        )


def build_matrix(seeds: list[int]) -> list[RunSpec]:
    """
    For each seed:
      - CLM_A at 17,500 steps, alpha=1.0    (matched tokens w/ selectors)
      - CLM_B at 25,000 steps, alpha=1.0    (matched optimizer steps)
      - SIEVE at 25,000 steps, alpha=0.70   (used in both framings)
      - Rho-1 at 25,000 steps, alpha=0.70   (used in both framings)
    """
    specs: list[RunSpec] = []
    for s in seeds:
        specs.append(RunSpec(baseline="clm_A", seed=s, steps=17500, alpha=1.0))
        specs.append(RunSpec(baseline="clm_B", seed=s, steps=25000, alpha=1.0))
        specs.append(RunSpec(baseline="sieve", seed=s, steps=25000, alpha=0.70))
        specs.append(RunSpec(baseline="rho1",  seed=s, steps=25000, alpha=0.70))
    return specs


@app.local_entrypoint()
def run_gate(seeds: str = "42,123,456", dry_run: bool = False, max_parallel: int = 4):
    seed_list = [int(s) for s in seeds.split(",")]
    specs = build_matrix(seed_list)

    print(f"[gate] stage=wt103_full_gate  runs={len(specs)}  seeds={seed_list}")
    for s in specs:
        print(f"  • {s.name:60s}  steps={s.steps:>5}  alpha={s.alpha}")

    if dry_run:
        return

    # Semaphore throttles client-side dispatch; max_containers=4 in the
    # train_remote decorator enforces Modal server-side concurrency cap.
    sem = threading.Semaphore(max_parallel)
    results: list[dict] = []
    lock = threading.Lock()
    t_start = time.time()

    def _run(s: RunSpec):
        with sem:
            t0 = time.time()
            r = None
            status = "failed"
            try:
                r = train_remote.remote(s.to_cli())
                # Even if no exception raised, check the actual status
                if r and r.get("status") == "ok":
                    status = "ok"
                else:
                    status = "failed"
            except Exception as e:
                r = {"name": s.name, "error": str(e)}
                status = "failed"
            wall = time.time() - t0

        rr = r if isinstance(r, dict) else {}
        with lock:
            row = {
                "name": s.name,
                "baseline": s.baseline,
                "seed": s.seed,
                "steps": s.steps,
                "alpha": s.alpha,
                "status": status,
                "wall_seconds": wall,
                "best_ppl": rr.get("best_ppl"),
                "final_val_ppl": rr.get("final_val_ppl"),
                "best_step": rr.get("best_step"),
                "scorer_overhead_pct": rr.get("scorer_overhead_pct"),
                "max_weight": rr.get("max_weight"),
            }
            results.append(row)
            best = row["best_ppl"] or "?"
            print(f"[{wall/60:.0f}min] {s.name:60s} best_ppl={best}  status={status}")

    threads = [threading.Thread(target=_run, args=(s,)) for s in specs]
    for t in threads: t.start()
    for t in threads: t.join()

    # Write aggregate CSV to local disk
    total_wall = (time.time() - t_start) / 60
    print(f"\n[gate] all runs done in {total_wall:.0f} min wall time")

    out_csv = Path("gate_results.csv")
    if results:
        keys = list(results[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        print(f"[gate] aggregate → {out_csv.resolve()}")

    # Print summary tables for both framings
    _print_framing_tables(results)


def _print_framing_tables(results: list[dict]) -> None:
    import statistics
    print("\n" + "=" * 60)
    print("APPROACH A — matched total tokens (~71.7M each)")
    print("=" * 60)
    _print_summary_table(results, "clm_A", "sieve", "rho1")

    print("\n" + "=" * 60)
    print("APPROACH B — matched optimizer steps (25k each)")
    print("=" * 60)
    _print_summary_table(results, "clm_B", "sieve", "rho1")


def _print_summary_table(results: list[dict], *baselines: str) -> None:
    import statistics
    print(f"  {'baseline':<10} {'mean':>8} {'std':>6} {'seeds':>6}")
    for b in baselines:
        ppls = [r["best_ppl"] for r in results
                if r["baseline"] == b and r["best_ppl"] is not None]
        if not ppls:
            print(f"  {b:<10} {'--':>8} {'--':>6} {0:>6}")
            continue
        mean = statistics.mean(ppls)
        std = statistics.stdev(ppls) if len(ppls) > 1 else 0.0
        print(f"  {b:<10} {mean:>8.3f} {std:>6.3f} {len(ppls):>6}")