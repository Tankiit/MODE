"""Local training entrypoint for SIEVE and baseline comparisons."""
import argparse, json, math, random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from sieve.config import SieveConfig
from sieve.device import get_cfg
from sieve.data import MemmapDataset
from sieve.state import SieveState
from sieve.loop import train
from sieve.rho1 import Rho1Baseline
from sieve.reference import load_reference

MODEL_NAMES = {
    "gpt2":      "gpt2",
    "gpt2l":     "gpt2-large",
    "nanogpt":   "nanogpt",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
}

TOKENIZER_NAMES = {
    "nanogpt": "gpt2",
}

DATASET_DIRS = {
    "wikitext2":   "./data/wikitext2",
    "wikitext103": "./data/wikitext103",
    "c4_1b":       "./data/c4_1b",
    "owt":         "./data/owt",
    "owm":         "./data/openwebmath_1b",
    "openwebmath_1b": "./data/openwebmath_1b",
}

BATCH_SIZES = {
    "gpt2":      8,
    "gpt2l":     4,   # GPT-2 Large needs smaller batch on MPS/32GB GPU
    "nanogpt":   16,
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
                            "sieve_no_se", "su_only", "sl_only", "sd_only"],
                   help="sieve=full SIEVE  rho1=S_E only  clm=no selection  "
                        "scalarization=fixed equal weights  su/sd_only=ablations")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument(
        "--patience",
        action="store_true",
        help="Enable patience-based early stopping (sieve/patience.py).",
    )
    p.add_argument("--patience_window", type=int, default=5)
    p.add_argument("--patience_min_steps", type=int, default=5000)
    p.add_argument("--patience_ppl_tol", type=float, default=0.005)
    p.add_argument("--patience_weight_tol", type=float, default=0.05)
    p.add_argument("--patience_entropy_tol", type=float, default=0.05)
    p.add_argument("--run_name", default=None, help="Optional display name for this run")
    p.add_argument(
        "--variance_normalized_bandit",
        action="store_true",
        help="VN-Mix: variance-normalized pseudo-count update (paper Eq. vnts).",
    )
    p.add_argument("--sigma_warmup", type=int, default=2,
                   help="VN-Mix: sigma = 1 until this many rewards are observed")
    p.add_argument(
        "--sd_scorer",
        default="inv_freq",
        choices=["inv_freq", "bigram_freq", "pos_inv_freq", "random_noise"],
        help="Fourth SIEVE scorer variant for controls.",
    )
    p.add_argument(
        "--bigram_table_path",
        default=None,
        help="Path to cached exact bigram log-frequency table for --sd_scorer bigram_freq.",
    )
    p.add_argument(
        "--patience_mode",
        default="2_of_3",
        choices=["2_of_3", "ppl_only", "weights_only", "entropy_only"],
    )
    p.add_argument("--ref_model", default="gpt2",
                   help="Frozen reference model for S_E (paper: GPT-2 Small)")
    p.add_argument("--log_interval", type=int, default=None,
                   help="Steps between log lines / history points (default: config)")
    p.add_argument("--out_dir", default=None,
                   help="Write history.json (and final.pt with --save_final) here")
    p.add_argument("--save_final", action="store_true",
                   help="Save the final model state_dict to <out_dir>/final.pt")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override the per-model batch size (tests only; the grid uses the defaults)")
    p.add_argument("--seq_len", type=int, default=None,
                   help="Override the sequence length (tests only; the grid uses 1024)")
    args = p.parse_args()
    if args.save_final and not args.out_dir:
        p.error("--save_final requires --out_dir")

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
        batch_size         = args.batch_size or BATCH_SIZES[args.model],
        selection_ratio    = args.alpha,
        patience_enable      = args.patience,
        patience_window      = args.patience_window,
        patience_min_steps   = args.patience_min_steps,
        patience_ppl_tol     = args.patience_ppl_tol,
        patience_weight_tol  = args.patience_weight_tol,
        patience_entropy_tol = args.patience_entropy_tol,
        patience_mode        = args.patience_mode,
        data_fraction      = args.data_fraction,
        sd_scorer          = args.sd_scorer,
        bigram_table_path  = args.bigram_table_path,
        variance_normalized_bandit = args.variance_normalized_bandit,
        sigma_warmup       = args.sigma_warmup,
        seed               = args.seed,
        profile_dataloader = args.profile_data,
        run_name           = args.run_name or (
            f"{args.baseline}_{args.dataset}_{args.model}"
            f"_a{int(args.alpha*100)}_s{args.seed}"
        ),
    )
    if args.eval_interval is not None:
        cfg.eval_interval = args.eval_interval
    if args.log_interval is not None:
        cfg.log_interval = args.log_interval
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.baseline == "sieve_no_se":
        cfg.scorer_names = ["S_U", "S_L", "S_D"]
        cfg.n_strategies = 3
    elif args.baseline == "su_only":
        cfg.scorer_names = ["S_U"]
        cfg.n_strategies = 1
    elif args.baseline == "sl_only":
        cfg.scorer_names = ["S_L"]
        cfg.n_strategies = 1
    elif args.baseline == "sd_only":
        cfg.scorer_names = ["S_D"]
        cfg.n_strategies = 1

    # ── Data ──────────────────────────────────────────────────────────
    train_ds = MemmapDataset(cfg.data_dir, "train", cfg)
    val_ds   = MemmapDataset(cfg.data_dir, "val",   cfg)

    if args.epochs is not None:
        spe = train_ds.steps_per_epoch(cfg.batch_size)
        cfg.max_steps = max(1, int(math.ceil(args.epochs * spe)))
        print(f"[train] epochs={args.epochs}  steps_per_epoch≈{spe}  "
              f"max_steps={cfg.max_steps}")

    cfg.freq_table_path = _find_freq_table(cfg.data_dir)
    if cfg.sd_scorer == "bigram_freq" and cfg.bigram_table_path is None:
        cfg.bigram_table_path = _find_bigram_table(cfg.data_dir)
        if cfg.bigram_table_path is None:
            raise FileNotFoundError(
                f"--sd_scorer bigram_freq requires a cached table at "
                f"{cfg.data_dir}/bigram_freq.pt or --bigram_table_path"
            )

    # ── Model ─────────────────────────────────────────────────────────
    # float32 master weights; the forward runs under bf16 autocast on CUDA
    # (DeviceCfg.autocast), i.e. bf16 activations + fp32 weights (App. B).
    model = build_model(args.model, cfg.seq_len, device).train()
    print(f"[model] {args.model}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"weights {next(model.parameters()).dtype}")

    # Liger: fused LM-head / CE backward on CUDA for GPT-2 (often ~40–60% faster).
    # Scoring (SIEVE hooks) unchanged — same logits path. Install: pip install liger-kernel
    # or pyproject extra: pip install -e ".[liger]". Not applicable to TinyLlama.
    if dev_cfg.backend == "cuda" and args.model in ("gpt2", "gpt2l"):
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_gpt2

            apply_liger_kernel_to_gpt2(model)
            print("[liger] Fused kernels applied to GPT-2 (CUDA)")
        except Exception as e:
            print(f"[liger] Not enabled: {e}")

    # ── Baseline construction ─────────────────────────────────────────
    sieve = _build_selector(args.baseline, cfg, device)

    # ── Reference model for S_E (only for selectors that use S_E) ─────
    ref_model = None
    uses_se = args.baseline == "rho1" or (
        sieve is not None and "S_E" in getattr(sieve, "scorer_names", []))
    if uses_se:
        ref_model = load_reference(args.ref_model, device)
        print(f"[ref] S_E reference model: {args.ref_model} (frozen, run on each batch)")

    # ── Train ─────────────────────────────────────────────────────────
    history = train(model, train_ds, val_ds, cfg, dev_cfg, sieve=sieve,
                    ref_model=ref_model)

    # ── Outputs ───────────────────────────────────────────────────────
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if args.save_final:
            torch.save(model.state_dict(), out / "final.pt")
        record = {"args": vars(args), "config": asdict(cfg), "history": _jsonable(history)}
        # history.json is written last and atomically: run_array.slurm uses
        # its presence to skip finished runs.
        tmp = out / "history.json.tmp"
        tmp.write_text(json.dumps(record, indent=1))
        tmp.replace(out / "history.json")
        print(f"[out] wrote {out / 'history.json'}"
              + (f" and {out / 'final.pt'}" if args.save_final else ""))

    # ── Summary ───────────────────────────────────────────────────────
    _print_summary(history, args.baseline)


def build_model(model_key: str, seq_len: int, device) -> torch.nn.Module:
    """
    Build the model with float32 weights (also used by scripts/analyze_selection.py).
    gpt2 / gpt2l start from the public pretrained checkpoints; nanogpt is a
    small randomly initialised GPT-2 used for tests.
    """
    if model_key == "nanogpt":
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAMES["nanogpt"])
        model_cfg = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=seq_len,
            n_ctx=seq_len,
            n_embd=384,
            n_layer=6,
            n_head=6,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        return GPT2LMHeadModel(model_cfg).to(device=device, dtype=torch.float32)
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAMES[model_key],
        dtype               = torch.float32,
        attn_implementation = "sdpa",
    ).to(device)


def _jsonable(x):
    """Recursively convert numpy / torch values so that json.dumps works."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _build_selector(baseline: str, cfg: SieveConfig, device):
    """
    Build the appropriate selector object.
    All selectors are duck-typed to SieveState's interface:
      step_begin / score_and_mask / on_eval / log_dict
    """
    if baseline in ("sieve", "sieve_no_se"):
        return SieveState(cfg, device)

    elif baseline == "rho1":
        # Rho-1: S_E only, same MaskCache, same SelectiveLoss, no bandit.
        # Rho1Baseline is duck-typed — training loop sees no difference.
        freq_table = None
        if cfg.freq_table_path:
            freq_table = torch.load(cfg.freq_table_path, map_location="cpu")
        rho1 = Rho1Baseline(
            selection_ratio  = cfg.selection_ratio,
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

    elif baseline in ("su_only", "sl_only", "sd_only"):
        # Single-strategy ablations
        state = SieveState(cfg, device)
        state._fixed_weights = torch.ones(len(cfg.scorer_names)) / len(cfg.scorer_names)
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


def _find_bigram_table(data_dir: str) -> str | None:
    """Look for cached exact bigram log-frequency table."""
    import pathlib
    p = pathlib.Path(data_dir) / "bigram_freq.pt"
    return str(p) if p.exists() else None


def _print_summary(history: dict, baseline: str):
    if "final_val_ppl" in history:
        print(f"\nFinal val PPL ({baseline}, full validation set): {history['final_val_ppl']:.3f}")

    if history.get("weights"):
        w = history["weights"][-1]
        print("Final bandit weights: " + "  ".join(f"w{i}={float(x):.3f}" for i, x in enumerate(w)))
        max_w = max(w)
        if max_w > 0.35:
            print(f"✓ Bandit differentiated (max={max_w:.3f})")
        else:
            print(f"✗ Weights near-uniform (max={max_w:.3f}) "
                  f"— check eval_interval and reward signal")

    if history.get("variance"):
        v = history["variance"][-1]
        print(
            "Final variance analysis: "
            f"sum={v['variance_sum']:.6f}  "
            f"weight_var={v['weight_variance']:.6f}  "
            f"gap={v['weight_gap']:.6f}"
        )

    if history.get("timing"):
        T = history["timing"]
        total = sum(T.values())
        if total > 0:
            score_pct = 100.0 * T.get("score", 0) / total
            print(f"Scorer overhead: {score_pct:.2f}% of total wall time")
            if score_pct < 2.0:
                print(f"✓ Within paper's ~1% claim")
            else:
                print(f"✗ Scorer overhead above 2% (scoring now runs at every step)")


if __name__ == "__main__":
    main()
