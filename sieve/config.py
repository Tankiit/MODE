"""Single source of truth for all hyperparameters."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class SieveConfig:
    # ── Selection ─────────────────────────────────────────────────────
    selection_ratio:    float = 0.70   # α — fraction of tokens kept
    rescore_interval:   int   = 50     # Δ — steps between mask refreshes

    # ── Bandit ────────────────────────────────────────────────────────
    n_strategies:       int   = 4      # K — S_E, S_U, S_L, S_D
    n_context_bits:     int   = 16     # binary state encoding width
    n_bins:             int   = 16     # context bins (2^4 phase bits)
    discount:           float = 0.95   # γ — posterior discount per update
    ema_alpha:          float = 0.99   # EMA for reward normalisation

    # ── Training ──────────────────────────────────────────────────────
    max_steps:          int   = 10_000
    eval_interval:      int   = 500    # steps between val PPL evals
    log_interval:       int   = 50
    profile_dataloader: bool  = False  # time sample_batch vs compute (syncs CUDA when True)
    batch_size:         int   = 8
    seq_len:            int   = 1024
    lr:                 float = 3e-4
    grad_clip:          float = 1.0
    seed:               int   = 42

    # ── Data ──────────────────────────────────────────────────────────
    data_dir:           str   = "./data/wikitext2"
    data_fraction:      float = 1.0            # train only: fraction of memmap tokens to use
    freq_table_path:    Optional[str] = None   # set by prepare.py
    ref_losses_path:    Optional[str] = None   # S_E offline cache

    # ── Model ─────────────────────────────────────────────────────────
    model_name:         str   = "gpt2"

    # ── Logging ───────────────────────────────────────────────────────
    wandb_project:      Optional[str] = None   # None = no wandb
    run_name:           Optional[str] = None

    # ── Checkpoints (optional; sieve/loop.py) ─────────────────────────
    save_dir:           Optional[str] = None
    save_interval:      int   = 0       # 0 = only periodic-off; final ckpt still when save_dir set
    max_checkpoints:    int   = 0       # 0 = keep all

    # ── Early stopping (sieve/patience.PatienceMonitor) ────────────────
    patience_enable:        bool   = False
    patience_window:        int    = 5
    patience_min_steps:     int    = 5000
    patience_ppl_tol:       float  = 0.005
    patience_weight_tol:   float  = 0.05
    patience_entropy_tol: float  = 0.05
    patience_mode:          str    = "2_of_3"  # 2_of_3 | ppl_only | weights_only | entropy_only
