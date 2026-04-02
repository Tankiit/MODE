"""
Shared mutable SIEVE state passed by reference between
the training loop and any callback/hook layer.
"""
from __future__ import annotations
import torch
from typing import Optional
from .config import SieveConfig
from .scorers import TokenScorers
from .bandit import DirichletBandit
from .mask import MaskCache


class SieveState:
    """
    Single object that owns all SIEVE components.
    Pass one instance to both the trainer and any callback.
    Mutations in compute_loss() are visible in on_evaluate().
    """

    def __init__(self, cfg: SieveConfig, device: torch.device):
        self.cfg    = cfg
        self.device = device
        self.step   = 0

        # Load optional tables
        freq_table = None
        if cfg.freq_table_path:
            freq_table = torch.load(cfg.freq_table_path, map_location="cpu")

        self.scorers = TokenScorers(freq_table=freq_table)
        self.bandit  = DirichletBandit(
            n_strategies=cfg.n_strategies,
            n_bins=cfg.n_bins,
            discount=cfg.discount,
            seed=cfg.seed,
        )
        self.mask_cache = MaskCache(cfg.selection_ratio, cfg.rescore_interval)

        # EMA state for reward normalisation
        self._ema:       float = 0.0
        self._prev_loss: Optional[float] = None

        # Current sampled weights (updated at each rescore)
        self.weights = torch.ones(cfg.n_strategies) / cfg.n_strategies

    # ── Hook 1 ────────────────────────────────────────────────────────
    def step_begin(self, step: int, max_steps: int,
                   prev_loss: float, prev_grad: float):
        """Update step counter. Sample bandit weights if rescore due."""
        self.step = step
        if self.mask_cache.needs_rescore(step):
            bin_idx      = self.bandit.encode_context(
                step, max_steps, prev_loss, prev_grad
            )
            self.weights = self.bandit.sample(bin_idx)  # [K] on CPU

    # ── Hook 2 (inside compute_loss) ─────────────────────────────────
    def score_and_mask(
        self,
        logits:     torch.Tensor,               # [B, T, V] float32
        input_ids:  torch.Tensor,               # [B, T]
        ref_losses: Optional[torch.Tensor] = None,
        attn_mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                          # [B, T] bool
        """Score tokens and update mask cache if rescore interval reached."""
        if not self.mask_cache.needs_rescore(self.step):
            return self.mask_cache.mask

        w = self.weights.to(self.device)
        scores = self.scorers.score_all(
            logits.detach().float(),   # detach: no grad through scoring
            input_ids,
            w,
            ref_losses=ref_losses,
        )
        return self.mask_cache.update(scores, self.step, attn_mask)

    # ── Hook 3 ────────────────────────────────────────────────────────
    def on_eval(self, val_loss: float):
        """Compute EMA-normalised reward and update bandit posterior."""
        reward = self._ema_reward(val_loss)
        self.bandit.update(reward)

    def _ema_reward(self, val_loss: float) -> float:
        if self._prev_loss is None:
            self._prev_loss = val_loss
            return 0.0
        raw             = self._prev_loss - val_loss   # +ve = improvement
        self._ema       = self.cfg.ema_alpha * self._ema + \
                          (1 - self.cfg.ema_alpha) * abs(raw)
        reward          = raw / max(self._ema, 1e-6)
        self._prev_loss = val_loss
        return float(reward)

    # ── Diagnostics ───────────────────────────────────────────────────
    def log_dict(self) -> dict:
        """Return a flat dict suitable for wandb.log() or print."""
        b   = self.bandit._current_bin
        ew  = self.bandit.expected_weights(b)
        ent = self.bandit.posterior_entropy(b)
        return {
            "sieve/w_S_E":            float(ew[0]),
            "sieve/w_S_U":            float(ew[1]),
            "sieve/w_S_L":            float(ew[2]),
            "sieve/w_S_D":            float(ew[3]),
            "sieve/dirichlet_entropy":ent,
            "sieve/n_selected":       self.mask_cache.n_selected,
            "sieve/context_bin":      b,
        }
