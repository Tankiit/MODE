"""
Scorer variance analysis for the "bandit stability bias" hypothesis.

Tests whether S_D has systematically lower across-step variance than
S_E, S_U, and S_L on the same validation tokens.

Usage:
  modal run scorer_variance_analysis.py::variance_analysis --dataset wikitext103 --alpha 0.7
  modal run scorer_variance_analysis.py::variance_analysis --dataset owm --alpha 0.7
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("sieve-variance-analysis")

REPO = Path(__file__).resolve().parent

volume_results = modal.Volume.from_name("sieve-results", create_if_missing=True)
volume_data = modal.Volume.from_name("sieve-data", create_if_missing=True)
volume_hf = modal.Volume.from_name("sieve-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "liger-kernel",
        "torch",
        "transformers>=4.36",
        "accelerate",
        "datasets",
        "tiktoken",
        "numpy",
        "wandb",
        "huggingface-hub",
        "einops",
        "kernels",
    )
    .add_local_dir(
        REPO,
        remote_path="/repo",
        ignore=["**/.git/**", "**/__pycache__/**", "**/*.pyc"],
    )
)

DATASET_VOLUME_MAP = {
    "wikitext103": "wikitext103",
    "wikitext2": "wikitext2",
    "owm": "openwebmath_1b",
    "openwebmath_1b": "openwebmath_1b",
}

MODEL_NAMES = {
    "gpt2": "gpt2",
    "gpt2l": "gpt2-large",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
}


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=15 * 60,
    volumes={
        "/results": volume_results,
        "/data": volume_data,
        "/hf_cache": volume_hf,
    },
)
def _run_variance_analysis(
    dataset: str,
    alpha: float,
    model: str = "gpt2l",
    seed: int = 42,
    run_name: str | None = None,
    early_step: int = 1000,
    late_step: int = 25000,
    batch_size: int = 4,
    seq_len: int = 1024,
) -> dict:
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache"
    sys.path.insert(0, "/repo")

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from sieve.scorers import TokenScorers

    if dataset not in DATASET_VOLUME_MAP:
        raise ValueError(f"Unknown dataset {dataset}; expected one of {sorted(DATASET_VOLUME_MAP)}")
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model}; expected one of {sorted(MODEL_NAMES)}")

    volume_data.reload()
    volume_results.reload()

    device = torch.device("cuda")
    data_dir = Path(f"/data/{DATASET_VOLUME_MAP[dataset]}")
    val_path = data_dir / "val.bin"
    freq_path = data_dir / "freq.pt"
    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation memmap: {val_path}")
    if not freq_path.exists():
        raise FileNotFoundError(f"Missing frequency table: {freq_path}")

    val_mmap = np.memmap(val_path, dtype=np.uint16, mode="r")
    if len(val_mmap) <= seq_len + 1:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}: {len(val_mmap)} tokens")

    rng = np.random.default_rng(seed=42)
    windows = []
    for _ in range(batch_size):
        start = int(rng.integers(0, len(val_mmap) - seq_len - 1))
        windows.append(np.asarray(val_mmap[start : start + seq_len], dtype=np.int64))
    batch_ids = torch.from_numpy(np.stack(windows)).to(device)
    freq_table = torch.load(freq_path, map_location=device)

    print(f"[variance] dataset={dataset} alpha={alpha} model={model} seed={seed}")
    print(f"[variance] val batch shape: {tuple(batch_ids.shape)}")

    alpha_tag = int(alpha * 100)
    candidate_run_names = [
        run_name,
        f"sieve_{dataset}_{model}_a{alpha_tag}_s{seed}",
        f"sieve_{dataset}_{model}_a{alpha_tag}_s{seed}_st{late_step}",
    ]
    candidate_run_names = [x for x in candidate_run_names if x]

    ckpt_root = None
    for name in candidate_run_names:
        p = Path("/results") / name / "checkpoints"
        if p.exists():
            ckpt_root = p
            break
    if ckpt_root is None:
        tried = "\n".join(f"  /results/{name}/checkpoints" for name in candidate_run_names)
        raise FileNotFoundError(f"No checkpoint directory found. Tried:\n{tried}")

    def ckpt_step(path: Path) -> int:
        return int(path.name.replace("ckpt-step", ""))

    all_ckpts = sorted(ckpt_root.glob("ckpt-step*"), key=ckpt_step)
    if not all_ckpts:
        raise FileNotFoundError(f"No ckpt-step* dirs in {ckpt_root}")

    print("[variance] Available checkpoints:")
    for p in all_ckpts:
        print(f"  - {p.name}")

    def closest_to(target: int) -> Path:
        return min(all_ckpts, key=lambda p: abs(ckpt_step(p) - target))

    early_ckpt = closest_to(early_step)
    late_ckpt = closest_to(late_step)
    early_actual = ckpt_step(early_ckpt)
    late_actual = ckpt_step(late_ckpt)

    print(f"[variance] Requested: early={early_step}, late={late_step}")
    print(f"[variance] Using:     early={early_actual}, late={late_actual}")
    if early_actual == late_actual:
        raise ValueError(f"Only one checkpoint available ({early_actual}); cannot compute across-step variance")

    model_name = MODEL_NAMES[model]

    def load_ckpt(ckpt_path: Path):
        """Load either HF checkpoint dirs or this repo's raw state_dict checkpoints."""
        if (ckpt_path / "config.json").exists():
            return AutoModelForCausalLM.from_pretrained(
                str(ckpt_path),
                dtype=torch.bfloat16,
                local_files_only=False,
            ).to(device).eval()

        if (ckpt_root / "config.json").exists():
            config = AutoConfig.from_pretrained(str(ckpt_root))
            model_obj = AutoModelForCausalLM.from_config(config).to(device, dtype=torch.bfloat16).eval()
        else:
            model_obj = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                local_files_only=False,
            ).to(device).eval()

        for model_file in ("pytorch_model.bin", "model.pt", "model.safetensors"):
            fp = ckpt_path / model_file
            if not fp.exists():
                continue
            if fp.suffix == ".safetensors":
                raise FileNotFoundError(
                    f"{fp} exists, but the training image does not include safetensors. "
                    "This repo's checkpoints normally use pytorch_model.bin."
                )
            state = torch.load(fp, map_location=device, weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            elif isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model_obj.load_state_dict(state)
            return model_obj

        contents = [p.name for p in ckpt_path.iterdir()]
        raise FileNotFoundError(f"No recognizable checkpoint file in {ckpt_path}; contents={contents}")

    scorers = TokenScorers(freq_table=freq_table)

    def token_scores(model_obj) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            logits = model_obj(batch_ids, use_cache=False).logits.float()
            scores = {
                "S_E": scorers.score_excess_loss(logits, batch_ids, ref_losses=None),
                "S_U": scorers.score_entropy(logits),
                "S_L": scorers.score_logit_spread(logits),
                "S_D": scorers.score_inv_freq(batch_ids, device=device),
            }
        del logits
        # Drop position 0, which is ignored by CLM loss and padded for S_E.
        return {name: score[:, 1:].contiguous() for name, score in scores.items()}

    print(f"\n[variance] Loading EARLY checkpoint ({early_actual})...")
    model_early = load_ckpt(early_ckpt)
    scores_early = token_scores(model_early)
    del model_early
    torch.cuda.empty_cache()
    print("[variance] Early scores computed.")

    print(f"\n[variance] Loading LATE checkpoint ({late_actual})...")
    model_late = load_ckpt(late_ckpt)
    scores_late = token_scores(model_late)
    del model_late
    torch.cuda.empty_cache()
    print("[variance] Late scores computed.")

    intra = {}
    for name, scores in scores_late.items():
        flat = scores.flatten().cpu().float()
        intra[name] = {
            "mean": float(flat.mean()),
            "std": float(flat.std(unbiased=False)),
            "variance": float(flat.var(unbiased=False)),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "range": float(flat.max() - flat.min()),
        }

    across = {}
    for name in scores_early:
        delta = (scores_late[name] - scores_early[name]).cpu().float().flatten()
        across[name] = {
            "mean_abs_delta": float(delta.abs().mean()),
            "std_delta": float(delta.std(unbiased=False)),
            "max_abs_delta": float(delta.abs().max()),
            "median_abs_delta": float(delta.abs().median()),
            "variance_of_delta": float(delta.var(unbiased=False)),
        }

    flat_late = {
        name: scores.flatten().cpu().float().numpy()
        for name, scores in scores_late.items()
    }

    def rankdata(values: np.ndarray) -> np.ndarray:
        """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
        values = np.asarray(values)
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranks = np.empty(len(values), dtype=np.float64)
        i = 0
        while i < len(values):
            j = i + 1
            while j < len(values) and sorted_values[j] == sorted_values[i]:
                j += 1
            avg_rank = 0.5 * (i + j - 1) + 1.0
            ranks[order[i:j]] = avg_rank
            i = j
        return ranks

    def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
        ra = rankdata(a)
        rb = rankdata(b)
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        denom = np.sqrt(np.sum(ra * ra) * np.sum(rb * rb))
        if denom == 0.0:
            return float("nan")
        return float(np.sum(ra * rb) / denom)

    pairwise = {}
    pairs = [
        ("S_E", "S_U"),
        ("S_E", "S_L"),
        ("S_E", "S_D"),
        ("S_U", "S_L"),
        ("S_U", "S_D"),
        ("S_L", "S_D"),
    ]
    for a, b in pairs:
        pairwise[f"{a}_vs_{b}"] = spearman_rho(flat_late[a], flat_late[b])

    return {
        "dataset": dataset,
        "alpha": alpha,
        "model": model,
        "seed": seed,
        "run_name": ckpt_root.parent.name,
        "checkpoint_root": str(ckpt_root),
        "early_step": early_actual,
        "late_step": late_actual,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "intra_batch_variance": intra,
        "across_step_variance": across,
        "pairwise_spearman_late": pairwise,
    }


def _print_report(results: dict) -> None:
    print("\n" + "=" * 72)
    print(f"REPORT - {results['dataset']} @ alpha={results['alpha']}")
    print(f"Run: {results['run_name']}")
    print(f"Early step {results['early_step']} -> Late step {results['late_step']}")
    print("=" * 72)

    print("\n-- TEST 1: Intra-batch variance (LATE checkpoint) --")
    print(f"{'Scorer':<6} {'mean':>10} {'std':>10} {'var':>10} {'range':>10}")
    for name in ["S_E", "S_U", "S_L", "S_D"]:
        s = results["intra_batch_variance"][name]
        print(
            f"{name:<6} {s['mean']:>10.4f} {s['std']:>10.4f} "
            f"{s['variance']:>10.4f} {s['range']:>10.4f}"
        )

    print("\n-- TEST 2: Across-step variance (EARLY -> LATE) --")
    print(f"{'Scorer':<6} {'mean|d|':>12} {'std(d)':>10} {'var(d)':>10} {'max|d|':>10} {'med|d|':>10}")
    for name in ["S_E", "S_U", "S_L", "S_D"]:
        s = results["across_step_variance"][name]
        print(
            f"{name:<6} {s['mean_abs_delta']:>12.4f} {s['std_delta']:>10.4f} "
            f"{s['variance_of_delta']:>10.4f} {s['max_abs_delta']:>10.4f} "
            f"{s['median_abs_delta']:>10.4f}"
        )

    print("\n-- TEST 3: Pairwise Spearman rank correlations (LATE) --")
    for pair, rho in results["pairwise_spearman_late"].items():
        print(f"  {pair:<18} rho = {rho:+.3f}")

    across = results["across_step_variance"]
    sd_delta = across["S_D"]["mean_abs_delta"]
    others = [across[k]["mean_abs_delta"] for k in ("S_E", "S_U", "S_L")]
    max_other = max(others)
    mean_other = sum(others) / len(others)
    ratio_max = max_other / (sd_delta + 1e-9)
    ratio_mean = mean_other / (sd_delta + 1e-9)

    if sd_delta < 0.001 and ratio_mean > 20:
        verdict = "STRONGLY SUPPORTED"
        msg = "Other scorers are more than 20x more variable across training than S_D."
    elif ratio_mean > 5:
        verdict = "MODERATELY SUPPORTED"
        msg = "Other scorers are 5-20x more variable than S_D."
    elif ratio_mean > 2:
        verdict = "WEAKLY SUPPORTED"
        msg = "Other scorers are only about 2-5x more variable than S_D."
    else:
        verdict = "NOT CLEARLY SUPPORTED"
        msg = "Other scorers have similar across-step variation to S_D."

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print(f"\nS_D across-step mean|d|:          {sd_delta:.4f}")
    print(f"Other scorers mean mean|d|:       {mean_other:.4f}")
    print(f"Max other scorer mean|d|:         {max_other:.4f}")
    print(f"Ratio max-other / S_D:            {ratio_max:.1f}x")
    print(f"Ratio mean-other / S_D:           {ratio_mean:.1f}x")
    print(f"\nSTABILITY BIAS HYPOTHESIS: {verdict}")
    print(f"  {msg}")

    pairwise = results["pairwise_spearman_late"]
    abs_corrs = [abs(v) for v in pairwise.values()]
    mean_abs_corr = sum(abs_corrs) / len(abs_corrs)
    max_abs_corr = max(abs_corrs)
    print(f"\nMean |Spearman rho| across all pairs: {mean_abs_corr:.3f}")
    print(f"Max  |Spearman rho|:                  {max_abs_corr:.3f}")
    if mean_abs_corr > 0.7:
        print("  -> Scorers are highly redundant.")
    elif mean_abs_corr > 0.4:
        print("  -> Moderate scorer overlap.")
    else:
        print("  -> Scorers capture largely distinct token properties.")
    print("=" * 72)


@app.local_entrypoint()
def variance_analysis(
    dataset: str = "wikitext103",
    alpha: float = 0.7,
    model: str = "gpt2l",
    seed: int = 42,
    run_name: str | None = None,
    early_step: int = 1000,
    late_step: int = 25000,
    batch_size: int = 4,
    seq_len: int = 1024,
) -> None:
    """Run analysis and print an interpretation report."""
    print("\n" + "=" * 72)
    print(f"Variance analysis: dataset={dataset}, alpha={alpha}, model={model}, seed={seed}")
    print("=" * 72 + "\n")

    results = _run_variance_analysis.remote(
        dataset=dataset,
        alpha=alpha,
        model=model,
        seed=seed,
        run_name=run_name,
        early_step=early_step,
        late_step=late_step,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    _print_report(results)

    out_path = Path(f"variance_analysis_{dataset}_{model}_a{int(alpha * 100)}_s{seed}.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[variance] Saved JSON -> {out_path}")
