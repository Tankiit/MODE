"""Single source of truth for all hyperparameters (defaults follow the paper, §3 and App. B)."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SieveConfig:
    # ── Selection ─────────────────────────────────────────────────────
    selection_ratio:    float = 0.70   # α — fraction of tokens kept per sequence
    rescore_interval:   int   = 100    # T_rescore — steps between mixture-weight samples
                                       # (scores and mask are recomputed at every step)

    # ── Mixture (single Dirichlet, paper Eq. update / reward) ──────────
    n_strategies:       int   = 4      # K — S_E, S_U, S_L, S_D
    scorer_names:       list[str] = field(default_factory=lambda: ["S_E", "S_U", "S_L", "S_D"])
    variance_normalized_bandit: bool = False   # VN-Mix (paper Eq. vnts)
    vn_eps:             float = 1e-3   # ε in Eq. vnts
    sigma_warmup:       int   = 2      # VN-Mix: σ̂ = 1 until this many rewards are observed

    # ── Training ──────────────────────────────────────────────────────
    max_steps:          int   = 25_000
    eval_interval:      int   = 1000   # T_val — steps between validation evaluations
    eval_batches:       int   = 50     # batches per intermediate evaluation (final eval uses all)
    log_interval:       int   = 100
    profile_dataloader: bool  = False  # time sample_batch vs compute (syncs CUDA when True)
    batch_size:         int   = 8
    seq_len:            int   = 1024
    lr:                 float = 3e-4
    beta1:              float = 0.9
    beta2:              float = 0.95
    weight_decay:       float = 0.1
    warmup_steps:       int   = 500    # linear warmup, then cosine decay
    min_lr_ratio:       float = 0.1    # cosine decays to 10% of peak at max_steps
    grad_clip:          float = 1.0
    seed:               int   = 42

    # ── Data ──────────────────────────────────────────────────────────
    data_dir:           str   = "./data/wikitext2"
    data_fraction:      float = 1.0            # train only: fraction of memmap tokens to use
    freq_table_path:    Optional[str] = None   # freq.pt, written by scripts/prepare_data.py
    ref_losses_path:    Optional[str] = None   # S_E offline cache
    sd_scorer:          str   = "inv_freq"     # inv_freq | bigram_freq | pos_inv_freq | random_noise
    bigram_table_path:  Optional[str] = None   # cached bigram log-probability table

    # ── Model ─────────────────────────────────────────────────────────
    model_name:         str   = "gpt2"

    # ── Logging ───────────────────────────────────────────────────────
    run_name:           Optional[str] = None

    # ── Early stopping (sieve/patience.PatienceMonitor) ────────────────
    patience_enable:        bool   = False
    patience_window:        int    = 5
    patience_min_steps:     int    = 5000
    patience_ppl_tol:       float  = 0.005
    patience_weight_tol:   float  = 0.05
    patience_entropy_tol: float  = 0.05
    patience_mode:          str    = "2_of_3"  # 2_of_3 | ppl_only | weights_only | entropy_only
