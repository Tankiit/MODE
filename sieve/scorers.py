"""
sieve/scorers.py

Four token-level scorers. All computed in-place from training logits —
zero extra forward passes for S_U, S_L, S_D.
S_E needs either ref_losses (precomputed) or falls back to raw NLL.

FIX applied: score_entropy now uses exp_() in-place to avoid
materialising a second [B,T,V] tensor. For GPT-2 Large (B=8, T=512,
V=50257) this saves ~820 MB peak GPU memory during scoring.

Original:  p = log_p.exp()          ← 820 MB extra alloc
Fixed:     entropy = -(log_p.exp_() * log_p).sum(-1)  ← reuses log_p buffer
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Optional


class TokenScorers:
    """
    Stateless scorer bank.
    Call score_all() inside compute_loss, after forward, before backward.
    logits must be float32 — cast before passing (loop.py does this).
    """

    def __init__(
        self,
        freq_table:  Optional[torch.Tensor] = None,   # [V] log-IDF on CPU
        topk_spread: int = 10,
    ):
        self.freq_table  = freq_table
        self.topk_spread = topk_spread

    # ── S_E: Excess Loss ─────────────────────────────────────────────
    @torch.no_grad()
    def score_excess_loss(
        self,
        logits:     torch.Tensor,               # [B, T, V] float32
        input_ids:  torch.Tensor,               # [B, T]
        ref_losses: Optional[torch.Tensor],     # [B, T] float32, or None
    ) -> torch.Tensor:                          # [B, T]
        B, T, V = logits.shape
        train_nll = F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            input_ids[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(B, T - 1)                    # [B, T-1]

        if ref_losses is not None:
            excess = (train_nll - ref_losses[:, 1:]).clamp(min=0.0)
        else:
            excess = train_nll                 # fallback: raw training loss

        return F.pad(excess, (1, 0), value=0.0)   # [B, T]

    # ── S_U: Prediction Entropy — FIXED ─────────────────────────────
    @torch.no_grad()
    def score_entropy(
        self,
        logits: torch.Tensor,   # [B, T, V] float32
    ) -> torch.Tensor:          # [B, T]
        """
        Shannon entropy H = -Σ p·log p.

        Memory: F.log_softmax uses a fused kernel — allocates ONE [B,T,V]
        buffer beyond the input logits. Then exp_() converts log_p → p
        IN PLACE, reusing that buffer. No second [B,T,V] allocation.

        Original code: p = log_p.exp()  ← separate alloc, 820 MB extra
        Fixed code:    log_p.exp_()     ← in-place, 0 MB extra
        """
        log_p   = F.log_softmax(logits, dim=-1)      # [B, T, V] — one alloc
        entropy = -(log_p.exp_() * log_p).sum(dim=-1) # exp_ in-place [B, T]
        return entropy

    # ── S_L: Inverted Logit Spread ───────────────────────────────────
    @torch.no_grad()
    def score_logit_spread(
        self,
        logits: torch.Tensor,   # [B, T, V] float32
    ) -> torch.Tensor:          # [B, T]
        # topk(10) only loads 10/50257 values — no V-dim allocation
        topk, _ = logits.topk(self.topk_spread, dim=-1)   # [B, T, k]
        spread   = topk[..., 0] - topk[..., -1]
        return 1.0 / (spread + 1e-6)

    # ── S_D: Inverse Token Frequency ────────────────────────────────
    @torch.no_grad()
    def score_inv_freq(
        self,
        input_ids: torch.Tensor,   # [B, T]
        device:    torch.device,
    ) -> torch.Tensor:             # [B, T]
        if self.freq_table is not None:
            # Corpus-level log-IDF — O(B*T) lookup, no recomputation.
            # Keep table on the same device to avoid D2H/H2D thrash.
            if self.freq_table.device != device:
                self.freq_table = self.freq_table.to(device, dtype=torch.float32)
            return self.freq_table[input_ids]
        else:
            # Within-batch fallback (high variance — use corpus table in prod)
            B, T   = input_ids.shape
            V      = 50257   # GPT-2 default; harmless for TinyLlama (rounds down)
            flat   = input_ids.reshape(-1)
            counts = torch.bincount(flat, minlength=V).float()
            freq   = counts[flat] / (B * T)
            return (1.0 / (freq + 1e-8)).reshape(B, T)

    # ── Combined ─────────────────────────────────────────────────────
    @torch.no_grad()
    def score_all(
        self,
        logits:     torch.Tensor,               # [B, T, V] float32
        input_ids:  torch.Tensor,               # [B, T]
        weights:    torch.Tensor,               # [4] simplex
        ref_losses: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                          # [B, T]
        device = logits.device

        s_e = self.score_excess_loss(logits, input_ids, ref_losses)
        s_u = self.score_entropy(logits)       # fixed — no extra V-dim alloc
        s_l = self.score_logit_spread(logits)
        s_d = self.score_inv_freq(input_ids, device)

        # Stack → [B, T, 4]
        raw  = torch.stack([s_e, s_u, s_l, s_d], dim=-1)

        # Z-score normalise each scorer across (batch, time)
        # Prevents high-variance scorers from dominating weighted sum
        mean = raw.mean(dim=(0, 1), keepdim=True)         # [1, 1, 4]
        std  = raw.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        norm = (raw - mean) / std

        # Weighted sum → [B, T]
        return (norm * weights.to(device).view(1, 1, 4)).sum(dim=-1)
