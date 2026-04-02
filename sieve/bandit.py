"""
Dirichlet Thompson Sampling contextual bandit.
One Dirichlet posterior per context bin (indexed by 4 phase bits).
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Optional


class DirichletBandit:
    """
    K=4 arms (S_E, S_U, S_L, S_D).
    B=16 context bins (4 phase indicator bits → 16 combinations).
    Each bin maintains its own Dirichlet posterior α^(b) ∈ R^K_{>0}.

    Update rule (from paper eq. 8):
        α^(b) ← γ · α^(b) + 2·r·w    if r > 0
        α^(b) ← γ · α^(b) + 0.1·1_K  if r ≤ 0
    """

    def __init__(
        self,
        n_strategies: int   = 4,
        n_bins:       int   = 16,
        discount:     float = 0.95,
        seed:         int   = 42,
    ):
        self.K        = n_strategies
        self.B        = n_bins
        self.gamma    = discount
        self.rng      = np.random.default_rng(seed)

        # Posteriors: [B, K], uniform initialisation
        self.alpha = np.ones((n_bins, n_strategies), dtype=np.float64)

        # Track current bin and sampled weights for update
        self._current_bin:     int              = 0
        self._current_weights: Optional[np.ndarray] = None

    # ── Context encoding ─────────────────────────────────────────────
    def encode_context(
        self,
        step:      int,
        max_steps: int,
        val_loss:  float,
        grad_norm: float,
    ) -> int:
        """
        16-bit binary state → 4-bit bin index (first 4 bits = phase).
        Returns bin index b ∈ {0, …, 15}.

        Phase bits (4): fractional training progress quartiles
        The bin index is b = Σ c_j · 2^(3-j) for j=0..3
        """
        p = step / max(max_steps, 1)
        # Phase indicator bits: thermometer encoding
        c0 = int(p > 0.00)
        c1 = int(p > 0.25)
        c2 = int(p > 0.50)
        c3 = int(p > 0.75)
        return (c0 << 3) | (c1 << 2) | (c2 << 1) | c3

    # ── Sample ───────────────────────────────────────────────────────
    def sample(self, bin_idx: int) -> torch.Tensor:
        """
        Sample weight vector from Dirichlet posterior for this bin.
        Returns [K] float32 tensor on CPU — weights.to(device) at call site.
        """
        self._current_bin = bin_idx
        sample = self.rng.dirichlet(self.alpha[bin_idx])
        self._current_weights = sample
        return torch.tensor(sample, dtype=torch.float32)

    # ── Update ───────────────────────────────────────────────────────
    def update(self, reward: float):
        """
        Bayesian posterior update after observing val loss improvement.
        reward > 0 means val loss improved — reinforce current weights.
        reward ≤ 0 means no improvement — add small uniform pseudocount.
        """
        b = self._current_bin
        w = self._current_weights
        if w is None:
            return

        self.alpha[b] *= self.gamma                      # temporal discount

        if reward > 0:
            self.alpha[b] += 2.0 * reward * w
        else:
            self.alpha[b] += 0.1                         # uniform floor

        self.alpha[b] = np.clip(self.alpha[b], 0.01, None)  # prevent collapse

    # ── Diagnostics ─────────────────────────────────────────────────
    def posterior_entropy(self, bin_idx: int) -> float:
        """
        Sum of marginal variances as Dirichlet entropy proxy.
        Should decrease within a training phase (exploitation increasing).
        Should spike at phase transitions.
        """
        a    = self.alpha[bin_idx]
        a0   = a.sum()
        var  = a * (a0 - a) / (a0**2 * (a0 + 1))
        return float(var.sum())

    def expected_weights(self, bin_idx: int) -> np.ndarray:
        """Expected value of Dirichlet = α / Σα. For logging."""
        a = self.alpha[bin_idx]
        return a / a.sum()
