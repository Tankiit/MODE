"""
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 500
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 500 --baseline rho1
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 500 --baseline clm
python train_sieve.py --dataset owm       --model tinyllama --steps 200
"""
import argparse, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sieve.config import SieveConfig
from sieve.device import get_cfg
from sieve.data import MemmapDataset
from sieve.state import SieveState
from sieve.loop import train
from sieve.rho1 import Rho1Baseline

MODEL_NAMES = {
    "gpt2":      "gpt2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
}

DATASET_DIRS = {
    "wikitext2": "./data/wikitext2",
    "owm":       "./data/owm",
}

BATCH_SIZES = {"gpt2": 8, "tinyllama": 2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  default="wikitext2",
                   choices=list(DATASET_DIRS.keys()))
    p.add_argument("--model",    default="gpt2",
                   choices=list(MODEL_NAMES.keys()))
    p.add_argument("--steps",    type=int, default=500)
    p.add_argument("--alpha",    type=float, default=0.70)
    p.add_argument("--baseline", default="sieve",
                   choices=["sieve", "rho1", "clm"],
                   help="sieve=full SIEVE  rho1=S_E only  clm=no selection")
    p.add_argument("--wandb",    default=None,
                   help="wandb project name (omit to disable)")
    p.add_argument("--seed",     type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dev_cfg = get_cfg()
    device  = dev_cfg.device
    print(f"backend={dev_cfg.backend}  dtype={dev_cfg.dtype}")

    cfg = SieveConfig(
        model_name      = MODEL_NAMES[args.model],
        data_dir        = DATASET_DIRS[args.dataset],
        max_steps       = args.steps,
        batch_size      = BATCH_SIZES[args.model],
        selection_ratio = args.alpha,
        seed            = args.seed,
        wandb_project   = args.wandb,
        run_name        = f"{args.baseline}_{args.dataset}_{args.model}"
                          f"_a{int(args.alpha*100)}_s{args.seed}",
    )

    # ── Data ──────────────────────────────────────────────────────────
    train_ds = MemmapDataset(cfg.data_dir, "train", cfg)
    val_ds   = MemmapDataset(cfg.data_dir, "val",   cfg)
    cfg.freq_table_path = str(
        __import__("pathlib").Path(cfg.data_dir) / "freq.pt"
    ) if (__import__("pathlib").Path(cfg.data_dir) / "freq.pt").exists() else None

    # ── Model ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dev_cfg.dtype,
        attn_implementation="sdpa",   # works on CUDA, MPS, CPU
    ).to(device).train()

    # ── Baseline selection ─────────────────────────────────────────────
    sieve = None

    if args.baseline == "sieve":
        sieve = SieveState(cfg, device)

    elif args.baseline == "rho1":
        # Load a separate frozen reference model (same architecture)
        ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        ).to(device)
        rho1 = Rho1Baseline(ref_model, cfg.selection_ratio, cfg.rescore_interval)

        # Wrap into a minimal SieveState-compatible object
        # by monkey-patching the training loop hooks
        class Rho1Adapter:
            def step_begin(self, *a, **kw): rho1.step += 1
            def score_and_mask(self, logits, input_ids, ref_losses=None, attn_mask=None):
                if rho1.cache.needs_rescore(rho1.step):
                    sc = rho1._score(logits, input_ids)
                    rho1.cache.update(sc, rho1.step)
                return rho1.cache.mask
            def on_eval(self, val_loss): pass
            def log_dict(self): return {}

        sieve = Rho1Adapter()   # duck-typed, loop doesn't care

    # args.baseline == "clm" → sieve stays None → full CLM

    # ── Train ─────────────────────────────────────────────────────────
    history = train(model, train_ds, val_ds, cfg, dev_cfg, sieve=sieve)

    # ── Summary ───────────────────────────────────────────────────────
    if history["val_ppl"]:
        final = history["val_ppl"][-1]
        print(f"\nFinal val PPL: {final:.2f}")

    if history.get("weights"):
        w = history["weights"][-1]
        print(f"Final bandit weights: "
              f"S_E={w[0]:.3f}  S_U={w[1]:.3f}  "
              f"S_L={w[2]:.3f}  S_D={w[3]:.3f}")
        max_w = max(w)
        if max_w > 0.35:
            print(f"✓ Bandit differentiated (max={max_w:.3f})")
        else:
            print(f"✗ Weights near-uniform — check eval_interval "
                  f"and reward signal")


if __name__ == "__main__":
    main()
