"""
sieve/rho1.py

S_E-only baseline (the paper's "excess-loss-only" baseline; not a
reproduction of Rho-1, whose reference model is trained on curated data).
Duck-typed to match SieveState's interface so train_sieve.py works
without modification to the training loop.

Comparison fairness guarantees:
  - Same codebase, same training loop, same optimizer
  - Same MaskCache (per-sequence top-k, recomputed at every step)
  - Same selective_loss (mean over selected tokens)
  - Only difference: utility = S_E alone (no mixture, no bandit)

Reference model: a frozen off-the-shelf checkpoint (train_sieve.py
--ref_model, default GPT-2 Small), run on each training batch
(sieve/reference.py). S_E raises an error if its losses are missing.
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

    # Only S_E is computed, with weight 1
    _W = torch.tensor([1.0])

    def __init__(
        self,
        selection_ratio:  float = 0.70,
        freq_table:       Optional[torch.Tensor] = None,
    ):
        # freq_table is not used for S_E
        self.scorers    = TokenScorers(freq_table=None, scorer_names=["S_E"])
        self.mask_cache = MaskCache(selection_ratio)
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
        Select top-α tokens of the current batch by S_E score.
        Recomputed at every step (same as SIEVE).
        """
        device = logits.device
        w      = self._W.to(device)

        scores = self.scorers.score_all(
            logits.detach().float(),
            input_ids,
            w,
            ref_losses=ref_losses,
        )
        return self.mask_cache.update(scores, self._step, attn_mask)

    def on_eval(self, val_loss: float, step: Optional[int] = None):
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