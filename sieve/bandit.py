"""
sieve/bandit.py

Dirichlet mixture over the K token scorers (paper §3, "Adaptive mixture
over scorers").

A single Dirichlet with pseudo-counts α ∈ R^K_{>0}, initialised to 1_K.
Every T_rescore steps a weight vector w ~ Dir(α) is sampled; at each
validation event the active w receives the clipped reward
r = max{0, L_prev − L_val} (computed by SieveState) and

    α ← α + r · w                      (paper Eq. update)
    α ← α + r · w / (σ̂ + ε)            (VN-Mix, paper Eq. vnts)

where σ̂_k is the running standard deviation of r_j · w_j^(k) over all
past validation events. This is a heuristic, not exact Bayesian inference.

The interface keeps a `bin_idx` argument (always 0) so that logging code
written for the earlier contextual version keeps working.
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Optional


class DirichletBandit:

    def __init__(
        self,
        n_strategies: int   = 4,
        seed:         int   = 42,
        variance_normalized: bool = False,
        vn_eps:       float = 1e-3,
        sigma_warmup: int   = 2,
    ):
        self.K   = n_strategies
        self.rng = np.random.default_rng(seed)

        # Pseudo-counts, shape [1, K] (one "bin"), uniform initialisation
        self.alpha = np.ones((1, n_strategies), dtype=np.float64)

        # Active sampled weights, credited at the next update()
        self._current_bin:     int                  = 0
        self._current_weights: Optional[np.ndarray] = None

        # VN-Mix: history of r_j · w_j over past validation events, [n, K]
        self.variance_normalized = variance_normalized
        self.vn_eps       = vn_eps
        self.sigma_warmup = sigma_warmup
        self._contrib: list[np.ndarray] = []

    # ── Context (single Dirichlet: always bin 0) ─────────────────────
    def encode_context(self, step: int, max_steps: int,
                       val_loss: float, grad_norm: float) -> int:
        return 0

    # ── Sample ───────────────────────────────────────────────────────
    def sample(self, bin_idx: int = 0) -> torch.Tensor:
        """Sample w ~ Dir(α); store it as the active mixture. Returns [K] float32 on CPU."""
        self._current_bin = 0
        sample = self.rng.dirichlet(self.alpha[0])
        self._current_weights = sample
        return torch.tensor(sample, dtype=torch.float32)

    # ── VN-Mix variance estimate ─────────────────────────────────────
    def _estimate_sigma(self, bin_idx: int = 0) -> np.ndarray:
        """
        Per-arm running std of r_j · w_j^(k) over all past validation events.
        Until sigma_warmup events have been observed, σ̂ = 1 (documented
        choice; the paper does not specify the cold start).
        """
        if len(self._contrib) < self.sigma_warmup:
            return np.ones(self.K, dtype=np.float64)
        return np.std(np.stack(self._contrib), axis=0)

    # ── Update ───────────────────────────────────────────────────────
    def update(self, reward: float):
        """Credit the active mixture with a non-negative reward r."""
        w = self._current_weights
        if w is None:
            return
        if reward < 0:
            raise ValueError(f"reward must be clipped to >= 0, got {reward}")

        if self.variance_normalized:
            self._contrib.append(reward * w)
            sigma = self._estimate_sigma()
            self.alpha[0] += reward * w / (sigma + self.vn_eps)
        else:
            self.alpha[0] += reward * w

    # ── Diagnostics ──────────────────────────────────────────────────
    def posterior_variance(self, bin_idx: int = 0) -> np.ndarray:
        """Marginal variances Var[w_i] = α_i(α_0 − α_i) / (α_0²(α_0 + 1))."""
        a = self.alpha[0]
        a0 = a.sum()
        return a * (a0 - a) / (a0**2 * (a0 + 1))

    def posterior_entropy(self, bin_idx: int = 0) -> float:
        """Sum of marginal variances (cheap concentration proxy)."""
        return float(self.posterior_variance().sum())

    def expected_weights(self, bin_idx: int = 0) -> np.ndarray:
        """E[w] = α / Σα (paper App. Bandit Dynamics)."""
        a = self.alpha[0]
        return a / a.sum()
