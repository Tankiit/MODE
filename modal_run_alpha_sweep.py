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

from modal_train_sieve import train_remote, app, volume_results, volume_data, image


@dataclass(frozen=True)
class RunSpec:
    baseline: str           # "sieve" | "rho1" | "clm_A" | "clm_B"
    seed:     int
    steps:    int
    alpha:    float
    dataset: str = "openwebmath_1b"
    model: str = "gpt2l"
    variance_normalized: bool = False
    sigma_warmup: int = 3
    save_interval: int = 1000
    max_checkpoints: int = 0
    sd_scorer: str = "inv_freq"

    @property
    def name(self) -> str:
        a_tag = f"a{int(self.alpha*100)}"
        vn_tag = "_VNTS" if self.variance_normalized else ""
        sd_tag = "" if self.sd_scorer == "inv_freq" else f"_{self.sd_scorer}"
        return f"{self.baseline}{vn_tag}{sd_tag}_{self.dataset}_{self.model}_{a_tag}_s{self.seed}_st{self.steps}"


    def to_cli(self) -> str:
        actual = "clm" if self.baseline.startswith("clm") else self.baseline
        alpha_flag = f"--alpha {self.alpha}" if actual != "clm" else ""
        vn_flag = (
            f"--variance_normalized_bandit --sigma_warmup {self.sigma_warmup}"
            if self.variance_normalized
            else ""
        )
        sd_flag = f"--sd_scorer {self.sd_scorer}" if self.sd_scorer != "inv_freq" else ""
        return (
            f"--dataset {self.dataset} --model {self.model} "
            f"--steps {self.steps} "
            f"--baseline {actual} "
            f"--seed {self.seed} "
            f"{alpha_flag} "
            f"{vn_flag} "
            f"{sd_flag} "
            f"--eval_interval 1000 "
            f"--save_interval {self.save_interval} "
            f"--max_checkpoints {self.max_checkpoints} "
            f"--wandb sieve-neurips "
            f"--run_name {self.name}"
        )


def build_matrix(seed: int, baselines: set[str], dataset: str, model: str) -> list[RunSpec]:
    """
    Build the sweep matrix. `baselines` filters which to include
    (for partial reruns if something fails mid-sweep).
    """
    ALPHAS = (0.30, 0.50, 0.70, 0.90)
    specs: list[RunSpec] = []

    if "sieve" in baselines:
        for a in ALPHAS:
            specs.append(RunSpec("sieve", seed, 25000, a, dataset=dataset, model=model))

    if "rho1" in baselines:
        for a in ALPHAS:
            specs.append(RunSpec("rho1", seed, 25000, a, dataset=dataset, model=model))

    if "clm" in baselines:
        # Matched tokens to α=0.7 selectors (71.7M tokens)
        specs.append(RunSpec("clm_A", seed, 17500, 1.0, dataset=dataset, model=model))
        # Full budget reference (102.4M tokens)
        specs.append(RunSpec("clm_B", seed, 25000, 1.0, dataset=dataset, model=model))

    return specs


@app.function(
    image=image,
    timeout=45 * 60,
    volumes={"/data": volume_data},
)
def prepare_bigram_freq_table(dataset: str = "wikitext103", vocab_size: int = 50257) -> dict:
    """Precompute exact sorted bigram log-frequency table on the data volume."""
    import numpy as np
    import torch

    dataset_dir = "openwebmath_1b" if dataset == "owm" else dataset
    data_dir = Path("/data") / dataset_dir
    out_path = data_dir / "bigram_freq.pt"
    if out_path.exists():
        obj = torch.load(out_path, map_location="cpu", weights_only=False)
        return {
            "path": str(out_path),
            "status": "exists",
            "n_bigrams": int(obj["keys"].numel()),
        }

    arr = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    prev = np.asarray(arr[:-1], dtype=np.int64)
    target = np.asarray(arr[1:], dtype=np.int64)
    keys = prev * int(vocab_size) + target
    uniq, counts = np.unique(keys, return_counts=True)
    total = float(counts.sum())
    log_freq = np.log(counts.astype(np.float64) / total).astype(np.float32)
    obj = {
        "keys": torch.from_numpy(uniq.astype(np.int64)),
        "log_freq": torch.from_numpy(log_freq),
        "vocab_size": int(vocab_size),
        "fallback_log_freq": -20.0,
        "source": str(data_dir / "train.bin"),
    }
    tmp_path = data_dir / "bigram_freq.pt.tmp"
    torch.save(obj, tmp_path)
    tmp_path.replace(out_path)
    volume_data.commit()
    return {
        "path": str(out_path),
        "status": "created",
        "n_bigrams": int(len(uniq)),
        "n_tokens": int(len(arr)),
    }


@app.local_entrypoint()
def run_sweep(
    seed: int = 42,
    baselines: str = "sieve,rho1,clm",
    dataset: str = "openwebmath_1b",
    model: str = "gpt2l",
    dry_run: bool = False,
    max_parallel: int = 4,
):
    baseline_set = {b.strip() for b in baselines.split(",")}
    specs = build_matrix(seed, baseline_set, dataset, model)

    print(f"[sweep] seed={seed}  baselines={sorted(baseline_set)}  "
          f"dataset={dataset}  model={model}  runs={len(specs)}")
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


@app.local_entrypoint()
def run_multi_seed(
    dataset: str = "wikitext103",
    alpha: float = 0.7,
    seeds: str = "123,456",
    baselines: str = "sieve,rho1",
    model: str = "gpt2l",
    steps: int = 25000,
    save_interval: int = 1000,
    max_checkpoints: int = 0,
    dry_run: bool = False,
    max_parallel: int = 4,
):
    """Multi-seed confirmation at one alpha for standard SIEVE/Rho-1."""
    seed_values = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    baseline_set = {b.strip() for b in baselines.split(",") if b.strip()}

    specs: list[RunSpec] = []
    for s in seed_values:
        if "sieve" in baseline_set:
            specs.append(
                RunSpec(
                    "sieve",
                    s,
                    steps,
                    alpha,
                    dataset=dataset,
                    model=model,
                    save_interval=save_interval,
                    max_checkpoints=max_checkpoints,
                )
            )
        if "rho1" in baseline_set:
            specs.append(
                RunSpec(
                    "rho1",
                    s,
                    steps,
                    alpha,
                    dataset=dataset,
                    model=model,
                    save_interval=save_interval,
                    max_checkpoints=max_checkpoints,
                )
            )

    print(f"[multi-seed] dataset={dataset} alpha={alpha}  "
          f"seeds={seed_values}  baselines={sorted(baseline_set)}  "
          f"checkpoint_every={save_interval}  max_checkpoints={max_checkpoints}")
    for s in specs:
        print(f"  • {s.name}")
        print(f"    {s.to_cli()}")
    print(f"  -> {len(specs)} runs, ~${len(specs) * 12} on A100")

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
            final_ppl = (r or {}).get("final_val_ppl")
            row = {
                "name": s.name,
                "baseline": s.baseline,
                "seed": s.seed,
                "steps": s.steps,
                "alpha": s.alpha,
                "dataset": s.dataset,
                "save_interval": s.save_interval,
                "max_checkpoints": s.max_checkpoints,
                "status": status,
                "wall_seconds": wall,
                "best_ppl": (r or {}).get("best_ppl"),
                "final_val_ppl": final_ppl,
                "best_step": (r or {}).get("best_step"),
                "error": (r or {}).get("error"),
            }
            results.append(row)
            print(f"[{wall/60:.0f}min] {s.name:60s}  ppl={final_ppl}  "
                  f"status={status}")

    threads = [threading.Thread(target=_run, args=(s,)) for s in specs]
    for t in threads: t.start()
    for t in threads: t.join()

    total_wall = (time.time() - t0_all) / 60
    print(f"\n[multi-seed] done in {total_wall:.0f} min")

    out_csv = Path(f"multi_seed_{dataset}_a{int(alpha * 100)}_results.csv")
    if results:
        keys = list(results[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(sorted(results, key=lambda r: (r["baseline"], r["seed"])))
        print(f"[multi-seed] -> {out_csv.resolve()}")


@app.local_entrypoint()
def run_multi_seed_sweep(
    dataset: str = "wikitext103",
    alphas: str = "0.3,0.5,0.9",
    seeds: str = "43,44",
    baselines: str = "sieve,rho1",
    model: str = "gpt2l",
    steps: int = 25000,
    save_interval: int = 1000,
    max_checkpoints: int = 0,
    dry_run: bool = False,
):
    """Run standard SIEVE/Rho-1 across multiple alphas and seeds."""
    alpha_values = [float(a.strip()) for a in alphas.split(",") if a.strip()]
    seed_values = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    baseline_set = {b.strip() for b in baselines.split(",") if b.strip()}

    specs: list[RunSpec] = []
    for s in seed_values:
        for a in alpha_values:
            if "sieve" in baseline_set:
                specs.append(
                    RunSpec(
                        "sieve",
                        s,
                        steps,
                        a,
                        dataset=dataset,
                        model=model,
                        save_interval=save_interval,
                        max_checkpoints=max_checkpoints,
                    )
                )
            if "rho1" in baseline_set:
                specs.append(
                    RunSpec(
                        "rho1",
                        s,
                        steps,
                        a,
                        dataset=dataset,
                        model=model,
                        save_interval=save_interval,
                        max_checkpoints=max_checkpoints,
                    )
                )

    print(f"[multi-seed-sweep] dataset={dataset}  model={model}  "
          f"alphas={alpha_values}  seeds={seed_values}  "
          f"baselines={sorted(baseline_set)}  runs={len(specs)}")
    print(f"[multi-seed-sweep] checkpoint_every={save_interval}  "
          f"max_checkpoints={max_checkpoints}")
    for spec in specs:
        print(f"  • {spec.name}")
        print(f"    {spec.to_cli()}")
    print(f"  -> {len(specs)} runs, ~${len(specs) * 12} on A100")

    if dry_run:
        return

    calls = []
    for spec in specs:
        call = train_remote.spawn(spec.to_cli())
        calls.append((spec, call))
        call_id = getattr(call, "object_id", None) or getattr(call, "function_call_id", None)
        print(f"[multi-seed-sweep] spawned {spec.name}" + (f"  call={call_id}" if call_id else ""))
    print(f"[multi-seed-sweep] submitted {len(calls)} runs; waiting for completion")

    completed = 0
    failed = []
    for spec, call in calls:
        try:
            result = call.get()
            status = result.get("status", "unknown") if result else "no-result"
            if status != "ok":
                failed.append((spec.name, status))
            completed += 1
            print(f"[multi-seed-sweep] completed {completed}/{len(calls)} "
                  f"{spec.name} status={status}")
        except Exception as exc:
            failed.append((spec.name, str(exc)))
            completed += 1
            print(f"[multi-seed-sweep] failed {completed}/{len(calls)} "
                  f"{spec.name}: {exc}")

    if failed:
        raise RuntimeError(f"{len(failed)}/{len(calls)} runs failed: {failed}")
    print(f"[multi-seed-sweep] all {len(calls)} runs completed successfully")


@app.local_entrypoint()
def run_vn_ts(
    dataset: str = "wikitext103",
    alphas: str = "0.3,0.7,0.9",
    alpha: float = -1.0,
    seed: int = 42,
    model: str = "gpt2l",
    steps: int = 25000,
    sigma_warmup: int = 2,
    save_interval: int = 1000,
    max_checkpoints: int = 0,
    dry_run: bool = False,
    max_parallel: int = 4,
):
    """VN-TS intervention sweep across alpha values."""
    alpha_arg = str(alpha) if alpha > 0 else alphas
    run_vn_ts_sweep(
        dataset=dataset,
        alphas=alpha_arg,
        seed=seed,
        model=model,
        steps=steps,
        sigma_warmup=sigma_warmup,
        save_interval=save_interval,
        max_checkpoints=max_checkpoints,
        dry_run=dry_run,
        max_parallel=max_parallel,
    )


@app.local_entrypoint()
def run_vn_ts_sweep(
    dataset: str = "wikitext103",
    alphas: str = "0.3,0.5,0.7,0.9",
    seed: int = 42,
    seeds: str = "",
    model: str = "gpt2l",
    steps: int = 25000,
    sigma_warmup: int = 2,
    save_interval: int = 1000,
    max_checkpoints: int = 0,
    dry_run: bool = False,
    max_parallel: int = 4,
):
    """Run VN-TS across alpha values and optionally multiple seeds."""
    alpha_values = [float(a.strip()) for a in alphas.split(",") if a.strip()]
    seed_values = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    if not seed_values:
        seed_values = [seed]
    specs = [
        RunSpec(
            baseline="sieve",
            seed=s,
            steps=steps,
            alpha=a,
            dataset=dataset,
            model=model,
            variance_normalized=True,
            sigma_warmup=sigma_warmup,
            save_interval=save_interval,
            max_checkpoints=max_checkpoints,
        )
        for s in seed_values
        for a in alpha_values
    ]

    print(f"[vn-ts-sweep] seeds={seed_values}  dataset={dataset}  model={model}  "
          f"alphas={alpha_values}  checkpoint_every={save_interval}  "
          f"max_checkpoints={max_checkpoints}  runs={len(specs)}")
    for s in specs:
        print(f"  • {s.name:60s}  steps={s.steps:>5}  α={s.alpha}")
        print(f"    {s.to_cli()}")

    if dry_run:
        return

    # Detached/background VN-TS sweeps must use spawn(). Blocking remote()
    # calls can be canceled when the local entrypoint disconnects.
    calls = []
    for s in specs:
        call = train_remote.spawn(s.to_cli())
        calls.append((s, call))
        call_id = getattr(call, "object_id", None) or getattr(call, "function_call_id", None)
        print(f"[vn-ts-sweep] spawned {s.name}" + (f"  call={call_id}" if call_id else ""))
    print(f"[vn-ts-sweep] submitted {len(calls)} background runs")


@app.local_entrypoint()
def run_sd_control_experiments(
    dataset: str = "wikitext103",
    alpha: float = 0.7,
    seed: int = 42,
    model: str = "gpt2l",
    steps: int = 25000,
    variants: str = "bigram_freq,random_noise",
    save_interval: int = 1000,
    max_checkpoints: int = 0,
    skip_bigram_prepare: bool = False,
    dry_run: bool = False,
):
    """
    Cheap controls for the fourth S_D arm.

    bigram_freq: exact cached corpus bigram rarity statistic.
    pos_inv_freq: deterministic position-conditioned unigram statistic.
    random_noise: non-corpus noisy negative-control arm.
    """
    variant_values = [v.strip() for v in variants.split(",") if v.strip()]
    if "bigram_freq" in variant_values and not dry_run and not skip_bigram_prepare:
        info = prepare_bigram_freq_table.remote(dataset)
        print(f"[sd-controls] bigram table: {info}")
    elif "bigram_freq" in variant_values and skip_bigram_prepare:
        print("[sd-controls] skipping bigram table preparation")
    specs = [
        RunSpec(
            baseline="sieve",
            seed=seed,
            steps=steps,
            alpha=alpha,
            dataset=dataset,
            model=model,
            sd_scorer=variant,
            save_interval=save_interval,
            max_checkpoints=max_checkpoints,
        )
        for variant in variant_values
    ]

    print(
        f"[sd-controls] dataset={dataset} model={model} alpha={alpha} "
        f"seed={seed} variants={variant_values} runs={len(specs)}"
    )
    print(f"[sd-controls] checkpoint_every={save_interval} max_checkpoints={max_checkpoints}")
    for spec in specs:
        print(f"  • {spec.name}")
        print(f"    {spec.to_cli()}")
    print(f"  -> {len(specs)} runs, ~${len(specs) * 12} on A100")

    if dry_run:
        return

    calls = []
    for spec in specs:
        call = train_remote.spawn(spec.to_cli())
        calls.append((spec, call))
        call_id = getattr(call, "object_id", None) or getattr(call, "function_call_id", None)
        print(f"[sd-controls] spawned {spec.name}" + (f"  call={call_id}" if call_id else ""))
    print(f"[sd-controls] submitted {len(calls)} background runs")


@app.local_entrypoint()
def run_scorer_substitution(
    dataset: str = "wikitext103",
    alpha: float = 0.7,
    seeds: str = "42,43,44",
    model: str = "gpt2l",
    steps: int = 25000,
    save_interval: int = 1000,
    max_checkpoints: int = 0,
    skip_bigram_prepare: bool = False,
    dry_run: bool = False,
):
    """
    A1/B1 scorer substitution matrix.

    A1: replace S_D with exact bigram corpus-frequency rarity.
    B1: replace S_D with fresh Gaussian noise.
    Vanilla TS only; no VN-TS.
    """
    seed_values = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    variants = ["bigram_freq", "random_noise"]

    if not dry_run and not skip_bigram_prepare:
        info = prepare_bigram_freq_table.remote(dataset)
        print(f"[scorer-sub] bigram table: {info}")
    elif skip_bigram_prepare:
        print("[scorer-sub] skipping bigram table preparation")
    else:
        print("[scorer-sub] dry-run: skipping bigram table preparation")

    specs = [
        RunSpec(
            baseline="sieve",
            seed=seed,
            steps=steps,
            alpha=alpha,
            dataset=dataset,
            model=model,
            sd_scorer=variant,
            save_interval=save_interval,
            max_checkpoints=max_checkpoints,
        )
        for variant in variants
        for seed in seed_values
    ]

    print(
        f"[scorer-sub] dataset={dataset} model={model} alpha={alpha} "
        f"seeds={seed_values} variants={variants} runs={len(specs)}"
    )
    print(f"[scorer-sub] checkpoint_every={save_interval} max_checkpoints={max_checkpoints}")
    for spec in specs:
        print(f"  • {spec.name}")
        print(f"    {spec.to_cli()}")
    print(f"  -> {len(specs)} runs, ~${len(specs) * 13} on A100")

    if dry_run:
        return

    calls = []
    for spec in specs:
        call = train_remote.spawn(spec.to_cli())
        calls.append((spec, call))
        call_id = getattr(call, "object_id", None) or getattr(call, "function_call_id", None)
        print(f"[scorer-sub] spawned {spec.name}" + (f"  call={call_id}" if call_id else ""))
    print(f"[scorer-sub] submitted {len(calls)} background runs")


@app.local_entrypoint()
def run_drop_se_experiments(
    dataset: str = "wikitext103",
    alpha: float = 0.7,
    seed: int = 42,
    model: str = "gpt2l",
    steps: int = 25000,
    baselines: str = "sieve_no_se,su_only,sl_only,sd_only",
    save_interval: int = 1000,
    max_checkpoints: int = 0,
    dry_run: bool = False,
):
    """
    Drop S_E controls:
      - 3-scorer SIEVE over {S_U, S_L, S_D}
      - single-scorer fixed SIEVE over {S_U}, {S_L}, {S_D}
    """
    baseline_values = [b.strip() for b in baselines.split(",") if b.strip()]
    specs = [
        RunSpec(
            baseline=b,
            seed=seed,
            steps=steps,
            alpha=alpha,
            dataset=dataset,
            model=model,
            save_interval=save_interval,
            max_checkpoints=max_checkpoints,
        )
        for b in baseline_values
    ]
    print(
        f"[drop-se] dataset={dataset} model={model} alpha={alpha} "
        f"seed={seed} runs={len(specs)}"
    )
    for spec in specs:
        print(f"  • {spec.name}")
        print(f"    {spec.to_cli()}")
    print(f"  -> {len(specs)} runs, ~${len(specs) * 12} on A100")

    if dry_run:
        return

    calls = []
    for spec in specs:
        call = train_remote.spawn(spec.to_cli())
        calls.append((spec, call))
        call_id = getattr(call, "object_id", None) or getattr(call, "function_call_id", None)
        print(f"[drop-se] spawned {spec.name}" + (f"  call={call_id}" if call_id else ""))
    print(f"[drop-se] submitted {len(calls)} background runs")
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
                "variance_normalized": s.variance_normalized,
                "sigma_warmup": s.sigma_warmup,
                "save_interval": s.save_interval,
                "max_checkpoints": s.max_checkpoints,
                "status": status,
                "wall_seconds": wall,
                "best_ppl": (r or {}).get("best_ppl"),
                "final_val_ppl": (r or {}).get("final_val_ppl"),
                "best_step": (r or {}).get("best_step"),
                "scorer_overhead_pct": (r or {}).get("scorer_overhead_pct"),
                "max_weight": (r or {}).get("max_weight"),
                "error": (r or {}).get("error"),
            }
            results.append(row)
            best = row["best_ppl"] or "?"
            print(f"[{wall/60:.0f}min] {s.name:60s} best_ppl={best}  "
                  f"status={status}")

    threads = [threading.Thread(target=_run, args=(s,)) for s in specs]
    for t in threads: t.start()
    for t in threads: t.join()

    total_wall = (time.time() - t0_all) / 60
    print(f"\n[vn-ts-sweep] all runs done in {total_wall:.0f} min wall time")

    out_csv = Path("vn_ts_alpha_sweep_results.csv")
    if results:
        keys = list(results[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(sorted(results, key=lambda r: (r["seed"], r["alpha"])))
        print(f"[vn-ts-sweep] aggregate -> {out_csv.resolve()}")

    _print_pareto_table(results)


@app.local_entrypoint()
def run_vn_ts_multiseed(
    dataset: str = "wikitext103",
    alpha: float = 0.7,
    seeds: str = "42,43,44",
    model: str = "gpt2l",
    steps: int = 25000,
    sigma_warmup: int = 2,
    save_interval: int = 1000,
    max_checkpoints: int = 0,
    dry_run: bool = False,
    max_parallel: int = 3,
):
    """Run VN-TS at one alpha across multiple seeds, preserving checkpoints."""
    run_vn_ts_sweep(
        dataset=dataset,
        alphas=str(alpha),
        seeds=seeds,
        model=model,
        steps=steps,
        sigma_warmup=sigma_warmup,
        save_interval=save_interval,
        max_checkpoints=max_checkpoints,
        dry_run=dry_run,
        max_parallel=max_parallel,
    )


def _print_pareto_table(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("α SWEEP RESULTS")
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
