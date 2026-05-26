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
        self.scorer_names = list(cfg.scorer_names)

        # Load optional tables
        freq_table = None
        if cfg.freq_table_path:
            freq_table = torch.load(cfg.freq_table_path, map_location="cpu")
        bigram_table = None
        if cfg.bigram_table_path:
            bigram_table = torch.load(cfg.bigram_table_path, map_location="cpu", weights_only=False)

        self.scorers = TokenScorers(
            freq_table=freq_table,
            bigram_table=bigram_table,
            sd_scorer=cfg.sd_scorer,
            scorer_names=self.scorer_names,
        )
        self.bandit  = DirichletBandit(
            n_strategies=len(self.scorer_names),
            n_bins=cfg.n_bins,
            discount=cfg.discount,
            seed=cfg.seed,
            variance_normalized=cfg.variance_normalized_bandit,
            var_history_window=cfg.var_history_window,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            sigma_warmup=cfg.sigma_warmup,
        )
        self.mask_cache = MaskCache(cfg.selection_ratio, cfg.rescore_interval)

        # EMA state for reward normalisation
        self._ema:       float = 0.0
        self._prev_loss: Optional[float] = None

        # Current sampled weights (updated at each rescore)
        self.weights = torch.ones(len(self.scorer_names)) / len(self.scorer_names)

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
        """Return a flat dict suitable for logging."""
        b   = self.bandit._current_bin
        ew  = self.bandit.expected_weights(b)
        var = self.bandit.posterior_variance(b)
        var_sum = float(var.sum())
        top2 = sorted((float(x) for x in ew), reverse=True)[:2]
        weight_by_name = {name: float(ew[i]) for i, name in enumerate(self.scorer_names)}
        var_by_name = {name: float(var[i]) for i, name in enumerate(self.scorer_names)}
        out = {
            "sieve/w_S_E":            weight_by_name.get("S_E", 0.0),
            "sieve/w_S_U":            weight_by_name.get("S_U", 0.0),
            "sieve/w_S_L":            weight_by_name.get("S_L", 0.0),
            "sieve/w_S_D":            weight_by_name.get("S_D", 0.0),
            # Backward-compatible name used by older dashboards.
            "sieve/dirichlet_entropy":var_sum,
            "sieve/variance_sum":     var_sum,
            "sieve/variance_S_E":     var_by_name.get("S_E", 0.0),
            "sieve/variance_S_U":     var_by_name.get("S_U", 0.0),
            "sieve/variance_S_L":     var_by_name.get("S_L", 0.0),
            "sieve/variance_S_D":     var_by_name.get("S_D", 0.0),
            "sieve/weight_variance":  float(ew.var()),
            "sieve/weight_gap":       top2[0] - top2[1] if len(top2) > 1 else 0.0,
            "sieve/n_selected":       self.mask_cache.n_selected,
            "sieve/context_bin":      b,
            "sieve/sd_scorer":        self.cfg.sd_scorer,
            "sieve/scorer_names":     ",".join(self.scorer_names),
        }
        if getattr(self.bandit, "variance_normalized", False):
            sigma = self.bandit._estimate_sigma(b)
            out.update({
                "sieve/sigma_S_E": float(sigma[0]),
                "sieve/sigma_S_U": float(sigma[1]),
                "sieve/sigma_S_L": float(sigma[2]),
                "sieve/sigma_S_D": float(sigma[3]),
                "sieve/vnts_active": 1,
            })
        return out
