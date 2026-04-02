"""
Rho-1 SLM baseline — re-implemented for fair comparison.
This is SIEVE with K=1, w=[1,0,0,0], no bandit, static mask.
Uses same selective_loss(), same per-sequence top-k masking.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Optional
from .mask import MaskCache
from .loss import selective_loss


class Rho1Baseline:
    """
    Token selection via excess loss only.
    Reference model: frozen pretrained model passed at init.
    """

    def __init__(
        self,
        ref_model:       torch.nn.Module,
        selection_ratio: float = 0.70,
        rescore_interval: int  = 50,
    ):
        # Freeze reference model completely
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()
        self.ref   = ref_model
        self.cache = MaskCache(selection_ratio, rescore_interval)
        self.step  = 0

    @torch.no_grad()
    def _score(
        self,
        logits_train: torch.Tensor,   # [B, T, V] float32, from training fwd
        input_ids:    torch.Tensor,   # [B, T]
    ) -> torch.Tensor:                # [B, T]
        B, T, V = logits_train.shape

        train_nll = F.cross_entropy(
            logits_train[:, :-1].reshape(-1, V),
            input_ids[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(B, T-1)

        ref_out = self.ref(
            input_ids=input_ids, labels=None, use_cache=False
        )
        ref_nll = F.cross_entropy(
            ref_out.logits[:, :-1].float().reshape(-1, V),
            input_ids[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(B, T-1)

        excess = (train_nll - ref_nll).clamp(min=0.0)
        return F.pad(excess, (1, 0), value=0.0)   # [B, T]

    def compute_loss(
        self,
        logits:    torch.Tensor,   # [B, T, V] float32
        input_ids: torch.Tensor,   # [B, T]
    ) -> torch.Tensor:
        self.step += 1
        if self.cache.needs_rescore(self.step):
            scores = self._score(logits, input_ids)
            self.cache.update(scores, self.step)
        return selective_loss(logits, input_ids, self.cache.mask)
