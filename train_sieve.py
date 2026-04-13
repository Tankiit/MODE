"""
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 500
python train_sieve.py --dataset wikitext2 --model gpt2 --epochs 3 --data_fraction 0.5 --profile_data
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 500 --baseline rho1
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 500 --baseline clm
python train_sieve.py --dataset owm       --model tinyllama --steps 200

Multi-seed gate run (run these three first before Modal):
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 500 --baseline sieve --seed 42
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 500 --baseline rho1  --seed 42
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 500 --baseline clm   --seed 42

Gate condition: sieve PPL < rho1 PPL < clm PPL
If gate fails: check wandb sieve/dirichlet_entropy — if flat, reward signal broken.
"""
import argparse, math, random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sieve.config import SieveConfig
from sieve.device import get_cfg
from sieve.data import MemmapDataset
from sieve.state import SieveState
from sieve.loop import train
from sieve.rho1 import Rho1Baseline    # now exists

MODEL_NAMES = {
    "gpt2":      "gpt2",
    "gpt2l":     "gpt2-large",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
}

DATASET_DIRS = {
    "wikitext2":  "./data/wikitext2",
    "wikitext103": "./data/wikitext103",
    "owm":        "./data/owm",
}

BATCH_SIZES = {
    "gpt2":      8,
    "gpt2l":     4,   # GPT-2 Large needs smaller batch on MPS/32GB GPU
    "tinyllama": 2,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  default="wikitext2",
                   choices=list(DATASET_DIRS.keys()))
    p.add_argument("--model",    default="gpt2",
                   choices=list(MODEL_NAMES.keys()))
    p.add_argument("--steps",    type=int, default=500)
    p.add_argument("--epochs",   type=float, default=None)
    p.add_argument("--profile_data", action="store_true")
    p.add_argument("--eval_interval", type=int, default=None)
    p.add_argument("--alpha",    type=float, default=0.70,
                   help="SIEVE token selection ratio")
    p.add_argument("--data_fraction", type=float, default=1.0)
    p.add_argument("--baseline", default="sieve",
                   choices=["sieve", "rho1", "clm", "scalarization",
                            "su_only", "sd_only"],
                   help="sieve=full SIEVE  rho1=S_E only  clm=no selection  "
                        "scalarization=fixed equal weights  su/sd_only=ablations")
    p.add_argument("--wandb",    default=None)
    p.add_argument("--seed",     type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dev_cfg = get_cfg()
    device  = dev_cfg.device
    print(f"backend={dev_cfg.backend}  dtype={dev_cfg.dtype}  seed={args.seed}")

    cfg = SieveConfig(
        model_name         = MODEL_NAMES[args.model],
        data_dir           = DATASET_DIRS[args.dataset],
        max_steps          = args.steps,
        batch_size         = BATCH_SIZES[args.model],
        selection_ratio    = args.alpha,
        data_fraction      = args.data_fraction,
        seed               = args.seed,
        wandb_project      = args.wandb,
        profile_dataloader = args.profile_data,
        run_name           = (
            f"{args.baseline}_{args.dataset}_{args.model}"
            f"_a{int(args.alpha*100)}_s{args.seed}"
        ),
    )
    if args.eval_interval is not None:
        cfg.eval_interval = args.eval_interval

    # ── Data ──────────────────────────────────────────────────────────
    train_ds = MemmapDataset(cfg.data_dir, "train", cfg)
    val_ds   = MemmapDataset(cfg.data_dir, "val",   cfg)

    if args.epochs is not None:
        spe = train_ds.steps_per_epoch(cfg.batch_size)
        cfg.max_steps = max(1, int(math.ceil(args.epochs * spe)))
        print(f"[train] epochs={args.epochs}  steps_per_epoch≈{spe}  "
              f"max_steps={cfg.max_steps}")

    cfg.freq_table_path = _find_freq_table(cfg.data_dir)

    # ── Model ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype            = dev_cfg.dtype,
        attn_implementation    = "sdpa",
    ).to(device).train()

    # Optional: apply Liger fused kernels on CUDA for faster backward.
    # Safe no-op on non-CUDA or if package unavailable.
    if dev_cfg.backend == "cuda":
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_gpt2
            apply_liger_kernel_to_gpt2(model)
            print("[liger] Applied fused kernels to GPT-2 (CUDA)")
        except Exception as e:
            print(f"[liger] Not enabled: {e}")

    # ── Baseline construction ─────────────────────────────────────────
    sieve = _build_selector(args.baseline, cfg, device)

    # ── Train ─────────────────────────────────────────────────────────
    history = train(model, train_ds, val_ds, cfg, dev_cfg, sieve=sieve)

    # ── Summary ───────────────────────────────────────────────────────
    _print_summary(history, args.baseline)


def _build_selector(baseline: str, cfg: SieveConfig, device):
    """
    Build the appropriate selector object.
    All selectors are duck-typed to SieveState's interface:
      step_begin / score_and_mask / on_eval / log_dict
    """
    if baseline == "sieve":
        return SieveState(cfg, device)

    elif baseline == "rho1":
        # Rho-1: S_E only, same MaskCache, same SelectiveLoss, no bandit.
        # Rho1Baseline is duck-typed — training loop sees no difference.
        freq_table = None
        if cfg.freq_table_path:
            freq_table = torch.load(cfg.freq_table_path, map_location="cpu")
        rho1 = Rho1Baseline(
            selection_ratio  = cfg.selection_ratio,
            rescore_interval = cfg.rescore_interval,
        )
        return rho1   # direct — no adapter needed

    elif baseline == "clm":
        return None   # full CLM: loop uses mask=None → no selection

    elif baseline == "scalarization":
        # Fixed equal weights across all four strategies — no bandit
        from sieve.state import SieveState as _S
        state = _S(cfg, device)
        # Override: fix weights to uniform, disable bandit sampling
        state._fixed_weights = torch.ones(4) / 4.0
        _orig_step_begin = state.step_begin
        def _fixed_step_begin(step, max_steps, prev_loss, prev_grad):
            state.step = step
            state.weights = state._fixed_weights
        state.step_begin = _fixed_step_begin
        return state

    elif baseline in ("su_only", "sd_only"):
        # Single-strategy ablations
        from sieve.state import SieveState as _S
        state = _S(cfg, device)
        w_map = {
            "su_only": torch.tensor([0., 1., 0., 0.]),
            "sd_only": torch.tensor([0., 0., 0., 1.]),
        }
        w = w_map[baseline]
        state._fixed_weights = w
        def _fixed_step_begin(step, max_steps, prev_loss, prev_grad):
            state.step = step
            state.weights = state._fixed_weights
        state.step_begin = _fixed_step_begin
        return state

    else:
        raise ValueError(f"Unknown baseline: {baseline}")


def _find_freq_table(data_dir: str) -> str | None:
    """Look for precomputed corpus-level IDF table."""
    import pathlib
    p = pathlib.Path(data_dir) / "freq.pt"
    return str(p) if p.exists() else None


def _print_summary(history: dict, baseline: str):
    if history["val_ppl"]:
        final = history["val_ppl"][-1]
        print(f"\nFinal val PPL ({baseline}): {final:.2f}")

    if history.get("weights"):
        w = history["weights"][-1]
        print(f"Final bandit weights: "
              f"S_E={w[0]:.3f}  S_U={w[1]:.3f}  "
              f"S_L={w[2]:.3f}  S_D={w[3]:.3f}")
        max_w = max(w)
        if max_w > 0.35:
            print(f"✓ Bandit differentiated (max={max_w:.3f})")
        else:
            print(f"✗ Weights near-uniform (max={max_w:.3f}) "
                  f"— check eval_interval and reward signal")

    if history.get("timing"):
        T = history["timing"]
        total = sum(T.values())
        if total > 0:
            score_pct = 100.0 * T.get("score", 0) / total
            print(f"Scorer overhead: {score_pct:.2f}% of total wall time")
            if score_pct < 2.0:
                print(f"✓ Within paper's ~1% claim")
            else:
                print(f"✗ Overhead too high — check rescore_interval")


if __name__ == "__main__":
    main()
