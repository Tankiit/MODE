"""
sieve/rho1.py

Rho-1 baseline: S_E-only token selection.
Duck-typed to match SieveState's interface so train_sieve.py works
without modification to the training loop.

Comparison fairness guarantees:
  - Same codebase, same training loop, same optimizer
  - Same eval harness (lm-evaluation-harness)
  - Same MaskCache (per-sequence topk, BOS-position skip)
  - Same selective_loss (gradient-invariant normalisation)
  - Only difference: weights = [1, 0, 0, 0] (S_E only, no bandit)

Reference model strategy:
  Rho-1 original: separate reference model trained on curated corpus.
  Our implementation: if ref_losses provided in batch (precomputed
  memmap), use them. Otherwise fall back to raw training NLL, which
  selects the hardest tokens by current model loss. This is validated
  as the "self-reference" variant in ESLM 2025.
"""
from __future__ import annotations
import torch
from typing import Optional
from .mask import MaskCache
from .scorers import TokenScorers


class Rho1Baseline:
    """
    Rho-1 selective LM — S_E only, no bandit, no adaptive weighting.

    Interface matches SieveState duck-typing in train_sieve.py:
      step_begin(step, max_steps, prev_loss, prev_grad)
      score_and_mask(logits, input_ids, ref_losses, attn_mask) → [B,T] bool
      on_eval(val_loss)
      log_dict() → dict
    """

    # Fixed weights: full weight on S_E, zero on S_U, S_L, S_D
    _W = torch.tensor([1.0, 0.0, 0.0, 0.0])

    def __init__(
        self,
        selection_ratio:  float = 0.70,
        rescore_interval: int   = 50,
        freq_table:       Optional[torch.Tensor] = None,
    ):
        # freq_table not used for S_E, but TokenScorers expects it
        self.scorers    = TokenScorers(freq_table=None)
        self.mask_cache = MaskCache(selection_ratio, rescore_interval)
        self._step      = 0

    # ── SieveState interface ──────────────────────────────────────────

    def step_begin(
        self,
        step:      int,
        max_steps: int,
        prev_loss: float,
        prev_grad: float,
    ):
        """Update step counter — no bandit to sample."""
        self._step = step

    @torch.no_grad()
    def score_and_mask(
        self,
        logits:     torch.Tensor,
        input_ids:  torch.Tensor,
        ref_losses: Optional[torch.Tensor] = None,
        attn_mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Select top-α tokens by S_E score.
        Reuses the mask for rescore_interval steps (same as SIEVE).
        """
        if not self.mask_cache.needs_rescore(self._step):
            return self.mask_cache.mask

        device = logits.device
        w      = self._W.to(device)

        scores = self.scorers.score_all(
            logits.detach().float(),
            input_ids,
            w,
            ref_losses=ref_losses,
        )
        return self.mask_cache.update(scores, self._step, attn_mask)

    def on_eval(self, val_loss: float):
        """No-op: Rho-1 has no bandit posterior to update."""
        pass

    def log_dict(self) -> dict:
        return {
            "sieve/method":     "rho1",
            "sieve/w_S_E":      1.0,
            "sieve/w_S_U":      0.0,
            "sieve/w_S_L":      0.0,
            "sieve/w_S_D":      0.0,
            "sieve/n_selected": self.mask_cache.n_selected,
        }