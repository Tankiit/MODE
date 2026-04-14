import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import modal

app = modal.App("sieve-neurips-2026")
volume = modal.Volume.from_name("sieve-results-v1", create_if_missing=True)
RESULTS_DIR = Path("/results")

REPO_ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.3.0",
        "transformers==4.41.0",
        "datasets==2.19.0",
        "accelerate==0.30.0",
        "tokenizers==0.19.1",
        "numpy==1.26.4",
        "scipy==1.13.0",
        "tqdm==4.66.4",
        "liger-kernel==0.3.0",
        "lm-eval==0.4.3",
        "wandb==0.17.0",
        "huggingface-hub",
        "einops",
        "kernels",
    )
    .copy_local_dir(str(REPO_ROOT), "/workspace")
)


@dataclass
class RunSpec:
    baseline: str
    dataset: str
    model: str
    seed: int
    alpha: float = 0.70
    steps: Optional[int] = None
    tag: str = ""

    @property
    def run_id(self) -> str:
        return (
            f"{self.baseline}_{self.dataset}_{self.model}"
            f"_a{int(self.alpha * 100)}_s{self.seed}"
        )

    @property
    def dataset_arg(self) -> str:
        return {
            "wt2": "wikitext2",
            "wt103": "wikitext103",
            "c4_1b": "c4_1b",
        }[self.dataset]

    @property
    def model_arg(self) -> str:
        return self.model

    @property
    def default_steps(self) -> int:
        return {
            "wt2": 10_000,
            "wt103": 25_000,
            "c4_1b": 25_000,
        }[self.dataset]

    @property
    def estimated_hours(self) -> float:
        base = {
            ("gpt2", "wt2"): 0.33,
            ("gpt2", "wt103"): 3.50,
            ("gpt2l", "wt2"): 0.75,
            ("gpt2l", "wt103"): 3.50,
            ("gpt2l", "c4_1b"): 3.50,
        }
        return base.get((self.model, self.dataset), 2.0)

    @property
    def estimated_cost(self) -> float:
        return self.estimated_hours * 0.90


def build_matrix() -> dict[str, list[RunSpec]]:
    gate = [
        RunSpec("sieve", "wt2", "gpt2", 42, tag="gate"),
        RunSpec("rho1", "wt2", "gpt2", 42, tag="gate"),
        RunSpec("clm", "wt2", "gpt2", 42, tag="gate"),
    ]
    ablation = [
        RunSpec("scalarization", "wt2", "gpt2", 42, tag="ablation"),
        RunSpec("su_only", "wt2", "gpt2", 42, tag="ablation"),
        RunSpec("sd_only", "wt2", "gpt2", 42, tag="ablation"),
    ]
    headline = [
        RunSpec(baseline, "wt103", "gpt2l", seed, tag="headline")
        for baseline in ["sieve", "rho1", "clm", "scalarization"]
        for seed in [42, 123, 456]
    ]
    c4 = [
        RunSpec("sieve", "c4_1b", "gpt2l", 42, tag="cross_domain"),
        RunSpec("rho1", "c4_1b", "gpt2l", 42, tag="cross_domain"),
        RunSpec("clm", "c4_1b", "gpt2l", 42, tag="cross_domain"),
    ]
    return {
        "gate": gate,
        "ablation": ablation,
        "headline": headline,
        "cross_domain": c4,
    }


MATRIX = build_matrix()
ALL_RUNS = (
    MATRIX["gate"]
    + MATRIX["ablation"]
    + MATRIX["headline"]
    + MATRIX["cross_domain"]
)


@app.function(
    image=image,
    gpu="A100",
    timeout=6 * 3600,
    volumes={str(RESULTS_DIR): volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    memory=32768,
    cpu=4.0,
    max_containers=4,
    keep_warm=0,
)
def run_experiment(spec_dict: dict) -> dict:
    import subprocess
    import sys

    raw = dict(spec_dict)
    skip_eval = bool(raw.pop("_skip_eval", False))
    spec = RunSpec(**raw)

    out_dir = RESULTS_DIR / spec.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = spec.steps or spec.default_steps

    print(f"\n{'=' * 60}")
    print(f"  RUN: {spec.run_id}")
    print(f"  Steps: {steps}  Alpha: {spec.alpha}  Tag: {spec.tag}")
    print(f"{'=' * 60}\n")

    cmd = [
        sys.executable,
        "/workspace/train_sieve.py",
        "--dataset",
        spec.dataset_arg,
        "--model",
        spec.model_arg,
        "--steps",
        str(steps),
        "--baseline",
        spec.baseline,
        "--alpha",
        str(spec.alpha),
        "--seed",
        str(spec.seed),
        "--eval_interval",
        "500",
    ]
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        cmd += ["--wandb", "sieve-neurips-2026"]

    print(f"Command: {' '.join(cmd)}\n")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd="/workspace",
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0
    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    print(f"\n[{spec.run_id}] {status} in {elapsed / 3600:.2f}h")

    if result.returncode != 0:
        (out_dir / "error.txt").write_text(f"returncode={result.returncode}\n")
        volume.commit()
        return {
            "run_id": spec.run_id,
            "status": "FAILED",
            "elapsed_h": elapsed / 3600,
            "baseline": spec.baseline,
            "dataset": spec.dataset,
            "model": spec.model,
            "seed": spec.seed,
            "alpha": spec.alpha,
            "tag": spec.tag,
        }

    ckpt_dir = Path("/workspace/checkpoints") / spec.run_id
    eval_results: dict = {}

    if not skip_eval and ckpt_dir.exists():
        print(f"\n[eval] Running lm-eval on {ckpt_dir}")
        eval_cmd = [
            sys.executable,
            "-m",
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            f"pretrained={ckpt_dir},dtype=float16",
            "--tasks",
            "wikitext,hellaswag,lambada_openai,winogrande",
            "--batch_size",
            "8",
            "--output_path",
            str(out_dir / "eval"),
            "--log_samples",
        ]
        eval_proc = subprocess.run(
            eval_cmd,
            cwd="/workspace",
            capture_output=True,
            text=True,
        )
        if eval_proc.returncode == 0:
            eval_dir = out_dir / "eval"
            result_files = list(eval_dir.glob("*.json")) if eval_dir.exists() else []
            if result_files:
                with open(result_files[0]) as f:
                    raw_eval = json.load(f)
                results_raw = raw_eval.get("results", {})
                eval_results = {
                    "ppl": results_raw.get("wikitext", {}).get(
                        "word_perplexity,none"
                    ),
                    "hellaswag": results_raw.get("hellaswag", {}).get(
                        "acc_norm,none"
                    ),
                    "lambada": results_raw.get("lambada_openai", {}).get(
                        "acc,none"
                    ),
                    "winogrande": results_raw.get("winogrande", {}).get(
                        "acc,none"
                    ),
                }
                print(
                    f"[eval] PPL={eval_results.get('ppl')}  "
                    f"HellaSwag={eval_results.get('hellaswag')}"
                )
        else:
            print(f"[eval] FAILED: {eval_proc.stderr[:200]}")
    elif skip_eval:
        print("[eval] skipped (--skip-eval)")
    else:
        print(f"[eval] Checkpoint not found at {ckpt_dir} — skipping eval")

    out = {
        "run_id": spec.run_id,
        "status": "SUCCESS",
        "elapsed_h": elapsed / 3600,
        "baseline": spec.baseline,
        "dataset": spec.dataset,
        "model": spec.model,
        "seed": spec.seed,
        "alpha": spec.alpha,
        "tag": spec.tag,
        **eval_results,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    volume.commit()
    return out


@app.local_entrypoint()
def main(
    gate_only: bool = False,
    dataset: str = "",
    baseline: str = "",
    seed: int = -1,
    dry_run: bool = False,
    max_par: int = 4,
    skip_eval: bool = False,
    skip_c4: bool = False,
):
    MAX_CONTAINERS = 4
    actual_par = min(max_par, MAX_CONTAINERS)
    gate_results: dict[str, dict] = {}

    if gate_only:
        runs = list(MATRIX["gate"])
    else:
        runs = list(ALL_RUNS)

    if dataset:
        runs = [r for r in runs if r.dataset == dataset]
    if baseline:
        runs = [r for r in runs if r.baseline == baseline]
    if seed >= 0:
        runs = [r for r in runs if r.seed == seed]

    if skip_c4:
        runs = [r for r in runs if r.dataset != "c4_1b"]
        print(
            "[info] Skipping C4 runs (--skip-c4 set). "
            "Run scripts/prepare_c4.py first to enable them."
        )

    if not runs:
        print("No runs matched filters.")
        return

    total_cost = sum(r.estimated_cost for r in runs)
    total_hours = sum(r.estimated_hours for r in runs)
    wall_hours = total_hours / actual_par

    print(f"\n{'=' * 65}")
    print(f"  SIEVE NeurIPS — Modal experiment launcher")
    print(f"{'=' * 65}")
    print(f"  Runs:          {len(runs)}")
    print(f"  Parallelism:   {actual_par} containers (hard cap: {MAX_CONTAINERS})")
    print(f"  Est. GPU-hrs:  {total_hours:.1f} hrs")
    print(f"  Est. cost:     ~${total_cost:.0f}")
    print(f"  Est. wall:     ~{wall_hours:.1f} hrs")
    print(f"\n  {'ID':<50} {'hrs':>5}  {'$':>5}  {'tag'}")
    print(f"  {'─' * 50} {'─' * 5}  {'─' * 5}  {'─' * 8}")
    for r in runs:
        print(
            f"  {r.run_id:<50} {r.estimated_hours:>5.1f}  "
            f"${r.estimated_cost:>4.2f}  {r.tag}"
        )
    print(f"{'=' * 65}\n")

    if dry_run:
        print("Dry run — not launching.")
        return

    if not gate_only and total_cost > 5.0:
        confirm = input(f"Launch {len(runs)} runs for ~${total_cost:.0f}? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return

    launch_gate_first = gate_only or (
        not dataset and not baseline and seed < 0
    )

    if launch_gate_first:
        print("\n── Phase 1: Gate check (sequential) ──────────────────────")

        for spec in MATRIX["gate"]:
            if spec not in runs:
                continue
            print(f"\n[gate] Launching {spec.run_id}...")
            payload = {**spec.__dict__, "_skip_eval": skip_eval}
            res = run_experiment.remote(payload)
            gate_results[spec.baseline] = res
            print(
                f"[gate] {spec.run_id}: PPL={res.get('ppl', 'N/A')}  "
                f"status={res.get('status')}"
            )

        sieve_ppl = gate_results.get("sieve", {}).get("ppl", float("inf"))
        rho1_ppl = gate_results.get("rho1", {}).get("ppl", float("inf"))
        clm_ppl = gate_results.get("clm", {}).get("ppl", float("inf"))

        try:
            sieve_f = float(sieve_ppl) if sieve_ppl is not None else float("inf")
            rho1_f = float(rho1_ppl) if rho1_ppl is not None else float("inf")
            clm_f = float(clm_ppl) if clm_ppl is not None else float("inf")
        except (TypeError, ValueError):
            sieve_f, rho1_f, clm_f = float("inf"), float("inf"), float("inf")

        gate_pass = sieve_f < rho1_f < clm_f and sieve_f < float("inf")

        print(f"\n── Gate results ───────────────────────────────────────────")
        print(f"  SIEVE PPL:  {sieve_ppl}")
        print(f"  Rho-1 PPL:  {rho1_ppl}")
        print(f"  CLM PPL:    {clm_ppl}")
        print(f"  Gate:       {'PASS' if gate_pass else 'FAIL'}")

        if not gate_pass:
            print("\nGate failed — aborting full matrix.")
            print("  Diagnose: check wandb sieve/dirichlet_entropy")
            print("  If entropy is flat: reward signal not reaching bandit")
            print("  If sieve ≥ rho1: check S_L and S_D scorer implementations")
            _print_summary_table(list(gate_results.values()))
            return

        print("\nGate passed — proceeding to full matrix.\n")

        if gate_only:
            _print_summary_table(list(gate_results.values()))
            return

        runs = [r for r in runs if r not in MATRIX["gate"]]

    print(
        f"\n── Phase 2: Full matrix ({len(runs)} runs, "
        f"{actual_par} parallel) ───────────────"
    )

    sem = threading.Semaphore(actual_par)
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    lock = threading.Lock()

    def submit(spec: RunSpec):
        with sem:
            t0 = time.time()
            print(f"[→] START  {spec.run_id}")
            try:
                payload = {**spec.__dict__, "_skip_eval": skip_eval}
                res = run_experiment.remote(payload)
                elapsed = (time.time() - t0) / 3600
                with lock:
                    results[spec.run_id] = res
                print(
                    f"[✓] DONE   {spec.run_id}  "
                    f"({elapsed:.2f}h)  PPL={res.get('ppl', 'N/A')}"
                )
                return spec.run_id, res
            except Exception as e:
                with lock:
                    errors[spec.run_id] = str(e)
                print(f"[✗] FAIL   {spec.run_id}  {e}")
                return spec.run_id, None

    with ThreadPoolExecutor(max_workers=actual_par) as pool:
        futures = {pool.submit(submit, r): r for r in runs}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"[!] Unhandled: {e}")

    all_results = list(results.values())
    for res in gate_results.values():
        if res and res not in all_results:
            all_results.append(res)

    _print_summary_table(all_results)

    if errors:
        print(f"\n{len(errors)} run(s) failed:")
        for rid, err in errors.items():
            print(f"  {rid}: {err[:80]}")
        print(f"\nRerun failed runs (adjust filters as needed).")

    print(f"\nResults saved to Modal Volume: sieve-results-v1")
    print(f"Download: modal volume get sieve-results-v1 /results ./local_results")


def _print_summary_table(results: list) -> None:
    results = [r for r in results if r and r.get("status") == "SUCCESS"]
    if not results:
        print("\nNo successful results to display.")
        return

    print(f"\n{'=' * 85}")
    print(f"  RESULTS SUMMARY (Paper Table 1)")
    print(f"{'=' * 85}")
    print(
        f"  {'Run':<48} {'PPL':>7} {'HellaSwag':>10} {'Lambada':>8} {'Wino':>6}"
    )
    print(
        f"  {'─' * 48} {'─' * 7} {'─' * 10} {'─' * 8} {'─' * 6}"
    )

    for dataset_label, ds_key in [
        ("WikiText-2 Ablations", "wt2"),
        ("WikiText-103 Headline", "wt103"),
        ("C4 1B Cross-domain", "c4_1b"),
    ]:
        ds_results = [r for r in results if r.get("dataset") == ds_key]
        if not ds_results:
            continue

        print(f"\n  [{dataset_label}]")

        order = {
            "clm": 0,
            "rho1": 1,
            "scalarization": 2,
            "su_only": 3,
            "sd_only": 4,
            "sieve": 5,
        }
        ds_results.sort(
            key=lambda r: (r.get("seed", 0), order.get(r.get("baseline", ""), 99))
        )

        for r in ds_results:
            ppl = r.get("ppl")
            hs = r.get("hellaswag")
            lam = r.get("lambada")
            win = r.get("winogrande")

            ppl_s = f"{ppl:.2f}" if ppl is not None else "—"
            hs_s = f"{hs:.4f}" if hs is not None else "—"
            lam_s = f"{lam:.4f}" if lam is not None else "—"
            win_s = f"{win:.4f}" if win is not None else "—"

            marker = "→ " if r.get("baseline") == "sieve" else "  "
            print(
                f"{marker}{r['run_id']:<48} {ppl_s:>7} "
                f"{hs_s:>10} {lam_s:>8} {win_s:>6}"
            )

    print(f"{'=' * 85}")

    for seed in [42, 123, 456]:
        sieve_r = next(
            (
                r
                for r in results
                if r.get("baseline") == "sieve"
                and r.get("dataset") == "wt103"
                and r.get("seed") == seed
            ),
            None,
        )
        rho1_r = next(
            (
                r
                for r in results
                if r.get("baseline") == "rho1"
                and r.get("dataset") == "wt103"
                and r.get("seed") == seed
            ),
            None,
        )
        if sieve_r and rho1_r:
            s_ppl = sieve_r.get("ppl")
            r_ppl = rho1_r.get("ppl")
            if s_ppl is not None and r_ppl is not None:
                try:
                    delta = float(r_ppl) - float(s_ppl)
                    wins = "SIEVE wins" if delta > 0 else "Rho-1 wins"
                    print(
                        f"  SIEVE vs Rho-1 (wt103, s={seed}): "
                        f"Δ PPL = {delta:+.2f} ({wins})"
                    )
                except (TypeError, ValueError):
                    pass
