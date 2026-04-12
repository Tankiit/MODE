"""
sieve/bandit.py

Dirichlet Thompson Sampling contextual bandit.
One Dirichlet posterior per context bin (indexed by 4 phase bits).

FIX applied: encode_context now uses OVERLAPPING phase bits as specified
in sieve_method_section_v2.tex eq. 3-4, not thermometer encoding.
Thermometer (old): c0=1[p>0.00] always fires, gives bins {8,12,14,15}.
Overlapping (correct): non-exclusive windows, gives bins {12,4,2,1}.
Both give ~4 active bins per run, but only overlapping matches the paper.
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

    Update rule (paper eq. 8):
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
        self.K     = n_strategies
        self.B     = n_bins
        self.gamma = discount
        self.rng   = np.random.default_rng(seed)

        # Posteriors: [B, K], uniform initialisation
        self.alpha = np.ones((n_bins, n_strategies), dtype=np.float64)

        # Track current bin and sampled weights for update()
        self._current_bin:     int               = 0
        self._current_weights: Optional[np.ndarray] = None

    # ── Context encoding ─────────────────────────────────────────────
    def encode_context(
        self,
        step:      int,
        max_steps: int,
        val_loss:  float,   # prev_loss in state.py — positional, name ignored
        grad_norm: float,
    ) -> int:
        """
        16-bit binary state → 4-bit bin index.
        Uses only the 4 phase bits for binning — bits [0-3].

        OVERLAPPING phase encoding (paper eq. 3-4):
          c0 = 1[p < 0.10]           early phase
          c1 = 1[p < 0.30]           pre-mid phase  (overlaps with c0)
          c2 = 1[0.30 ≤ p < 0.70]   mid phase
          c3 = 1[p ≥ 0.70]           late phase

        Active bins per run (non-exclusive windows):
          p ∈ [0.00, 0.10):  c0=1, c1=1, c2=0, c3=0  → bin 12
          p ∈ [0.10, 0.30):  c0=0, c1=1, c2=0, c3=0  → bin 4
          p ∈ [0.30, 0.70):  c0=0, c1=0, c2=1, c3=0  → bin 2
          p ∈ [0.70, 1.00]:  c0=0, c1=0, c2=0, c3=1  → bin 1

        bin = c0·2³ + c1·2² + c2·2¹ + c3·2⁰
        """
        p  = step / max(max_steps, 1)
        c0 = int(p < 0.10)
        c1 = int(p < 0.30)
        c2 = int(0.30 <= p < 0.70)
        c3 = int(p >= 0.70)
        return (c0 << 3) | (c1 << 2) | (c2 << 1) | c3

    # ── Sample ───────────────────────────────────────────────────────
    def sample(self, bin_idx: int) -> torch.Tensor:
        """
        Sample weight vector from Dirichlet posterior for this bin.
        Stores bin_idx and weights for the subsequent update() call.
        Returns [K] float32 tensor on CPU.
        """
        self._current_bin = bin_idx
        alpha  = np.clip(self.alpha[bin_idx], 0.01, None)
        sample = self.rng.dirichlet(alpha)
        self._current_weights = sample
        return torch.tensor(sample, dtype=torch.float32)

    # ── Update ───────────────────────────────────────────────────────
    def update(self, reward: float):
        """
        Bayesian posterior update after observing val loss improvement.
        Called from SieveState.on_eval() with EMA-normalised reward.
        Uses stored _current_bin and _current_weights from last sample().
        """
        b = self._current_bin
        w = self._current_weights
        if w is None:
            return

        self.alpha[b] *= self.gamma          # temporal discount

        if reward > 0:
            self.alpha[b] += 2.0 * reward * w
        else:
            self.alpha[b] += 0.1             # uniform floor — keeps all arms alive

        self.alpha[b] = np.clip(self.alpha[b], 0.01, None)

    # ── Diagnostics ──────────────────────────────────────────────────
    def posterior_entropy(self, bin_idx: int) -> float:
        """
        Sum of marginal variances as Dirichlet entropy proxy.
        Decreases within a training phase (exploitation increasing).
        Spikes at phase transitions (bandit becomes uncertain again).
        If flat across all training: reward signal not reaching bandit.
        """
        a   = self.alpha[bin_idx]
        a0  = a.sum()
        var = a * (a0 - a) / (a0**2 * (a0 + 1))
        return float(var.sum())

    def expected_weights(self, bin_idx: int) -> np.ndarray:
        """
        Expected value of Dirichlet(α) = α / Σα.
        Used for logging (wandb Figure 2) — not for sampling.
        Returns numpy array; .tolist() works in loop.py.
        """
        a = self.alpha[bin_idx]
        return a / a.sum()

    def summary(self) -> dict:
        """Flat dict for wandb.log()."""
        b  = self._current_bin
        ew = self.expected_weights(b)
        return {
            "bandit/bin":              b,
            "bandit/posterior_entropy": self.posterior_entropy(b),
            "bandit/w_S_E":            float(ew[0]),
            "bandit/w_S_U":            float(ew[1]),
            "bandit/w_S_L":            float(ew[2]),
            "bandit/w_S_D":            float(ew[3]),
        }