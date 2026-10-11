"""
sieve/scorers.py

Four token-level scorers (paper §3), computed from the training logits.
S_E also needs the per-token losses of the frozen reference model on the
same batch (sieve/reference.py).

Alignment convention: every scorer returns a [B, T] tensor whose column t
scores token t, i.e. the loss term (logits[t-1], input_ids[t]) that
mask[t] selects. Model-state scorers are therefore computed from
logits[:, :-1] and padded on the left; column 0 is never selected.
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
        freq_table:  Optional[torch.Tensor] = None,   # [V] −log corpus frequency, on CPU
        bigram_table: Optional[dict] = None,
        sd_scorer:   str = "inv_freq",
        scorer_names: Optional[list[str]] = None,
    ):
        self.freq_table  = freq_table
        self.bigram_table = bigram_table
        self.sd_scorer   = sd_scorer
        self.scorer_names = scorer_names or ["S_E", "S_U", "S_L", "S_D"]
        self._noise_gen: Optional[torch.Generator] = None

    # ── S_E: Excess Loss ─────────────────────────────────────────────
    @torch.no_grad()
    def score_excess_loss(
        self,
        logits:     torch.Tensor,               # [B, T, V] float32
        input_ids:  torch.Tensor,               # [B, T]
        ref_losses: Optional[torch.Tensor],     # [B, T] float32
    ) -> torch.Tensor:                          # [B, T]
        """S_E = ℓ_θ − ℓ_ref (paper §3), column t scores token t."""
        if ref_losses is None:
            raise ValueError(
                "S_E needs reference losses: pass ref_model to sieve.loop.train "
                "(train_sieve.py loads --ref_model for selectors that use S_E)"
            )
        B, T, V = logits.shape
        train_nll = F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            input_ids[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(B, T - 1)                    # [B, T-1]
        excess = train_nll - ref_losses[:, 1:]
        return F.pad(excess, (1, 0), value=0.0)   # [B, T]

    # ── S_U: Predictive Entropy ──────────────────────────────────────
    @torch.no_grad()
    def score_entropy(
        self,
        logits: torch.Tensor,   # [B, T, V] float32
    ) -> torch.Tensor:          # [B, T]
        """
        Shannon entropy H = -Σ p·log p of the distribution that predicts
        token t (logits[t-1]); column 0 is padding.

        Note: the earlier `-(log_p.exp_() * log_p)` overwrote log_p in place
        and therefore computed -Σ p², not the entropy. The einsum avoids a
        third [B,T,V] temporary.
        """
        log_p   = F.log_softmax(logits[:, :-1], dim=-1)   # [B, T-1, V]
        p       = log_p.exp()
        entropy = -torch.einsum("btv,btv->bt", p, log_p)  # [B, T-1]
        return F.pad(entropy, (1, 0), value=0.0)          # [B, T]

    # ── S_L: Negative Logit Margin ───────────────────────────────────
    @torch.no_grad()
    def score_logit_spread(
        self,
        logits: torch.Tensor,   # [B, T, V] float32
    ) -> torch.Tensor:          # [B, T]
        """
        S_L = −(largest logit − second-largest logit) of the distribution
        that predicts token t (paper §3); column 0 is padding.
        """
        top2, _ = logits[:, :-1].topk(2, dim=-1)          # [B, T-1, 2]
        margin  = top2[..., 0] - top2[..., 1]
        return F.pad(-margin, (1, 0), value=0.0)          # [B, T]

    # ── S_D: Inverse Token Frequency ────────────────────────────────
    @torch.no_grad()
    def score_inv_freq(
        self,
        input_ids: torch.Tensor,   # [B, T]
        device:    torch.device,
    ) -> torch.Tensor:             # [B, T]
        """S_D = −log of the token's empirical corpus frequency (paper §3)."""
        if self.freq_table is None:
            raise ValueError(
                "S_D needs the corpus table <data_dir>/freq.pt "
                "(written by scripts/prepare_data.py)"
            )
        # Keep table on the same device to avoid D2H/H2D thrash.
        if self.freq_table.device != device:
            self.freq_table = self.freq_table.to(device, dtype=torch.float32)
        return self.freq_table[input_ids]

    # ── S_D variants for constant-arm controls ───────────────────────
    @torch.no_grad()
    def score_pos_inv_freq(
        self,
        input_ids: torch.Tensor,
        device:    torch.device,
    ) -> torch.Tensor:
        """
        Position-conditioned corpus-frequency statistic.

        This keeps the fourth arm deterministic and corpus-statistic based,
        but changes it from pure unigram rarity to a position-modulated rarity
        score. It is cheap because it reuses freq.pt and needs no extra table.
        """
        base = self.score_inv_freq(input_ids, device)
        T = input_ids.shape[1]
        pos = torch.arange(T, device=device, dtype=base.dtype)
        pos_scale = torch.log1p(pos) / torch.log(torch.tensor(float(T), device=device, dtype=base.dtype))
        pos_scale = 0.5 + pos_scale.clamp(0.0, 1.0)
        return base * pos_scale.view(1, T)

    @torch.no_grad()
    def score_random_noise(
        self,
        input_ids: torch.Tensor,
        device:    torch.device,
    ) -> torch.Tensor:
        """
        Non-corpus-statistic negative control: intentionally noisy Gaussian arm.
        """
        if self._noise_gen is None or self._noise_gen.device != device:
            self._noise_gen = torch.Generator(device=device)
            self._noise_gen.manual_seed(12345)
        return torch.randn(input_ids.shape, device=device, generator=self._noise_gen)

    @torch.no_grad()
    def score_bigram_freq(
        self,
        input_ids: torch.Tensor,
        device:    torch.device,
    ) -> torch.Tensor:
        """
        Exact corpus bigram rarity: -log f(prev_token, token).

        The cached table stores sorted integer keys prev * vocab_size + token
        and log frequencies. Lookup uses torch.searchsorted, so the scorer stays
        vectorized on GPU for the sampled batch.
        """
        if self.bigram_table is None:
            raise ValueError("sd_scorer='bigram_freq' requires bigram_table_path")

        keys = self.bigram_table["keys"].cpu()
        log_freq = self.bigram_table["log_freq"].cpu()
        vocab_size = int(self.bigram_table["vocab_size"])
        fallback_log_freq = float(self.bigram_table.get("fallback_log_freq", -20.0))

        out_cpu = torch.full(input_ids.shape, -fallback_log_freq, device="cpu", dtype=torch.float32)
        if input_ids.shape[1] < 2:
            return out_cpu.to(device)

        ids_cpu = input_ids.detach().cpu().long()
        prev_ids = ids_cpu[:, :-1]
        target_ids = ids_cpu[:, 1:]
        query = prev_ids * vocab_size + target_ids
        idx = torch.searchsorted(keys, query.reshape(-1))
        valid = idx < keys.numel()
        flat_query = query.reshape(-1)
        safe_idx = idx.clamp(max=max(keys.numel() - 1, 0))
        valid &= keys[safe_idx] == flat_query

        vals = torch.full(flat_query.shape, fallback_log_freq, device="cpu", dtype=torch.float32)
        vals[valid] = log_freq[safe_idx[valid]]
        out_cpu[:, 1:] = (-vals).reshape_as(target_ids)
        out_cpu[:, 0] = out_cpu[:, 1:].mean(dim=1) if input_ids.shape[1] > 1 else -fallback_log_freq
        return out_cpu.to(device=device, non_blocking=True)

    @torch.no_grad()
    def score_sd(
        self,
        input_ids: torch.Tensor,
        device:    torch.device,
    ) -> torch.Tensor:
        if self.sd_scorer == "inv_freq":
            return self.score_inv_freq(input_ids, device)
        if self.sd_scorer == "bigram_freq":
            return self.score_bigram_freq(input_ids, device)
        if self.sd_scorer == "pos_inv_freq":
            return self.score_pos_inv_freq(input_ids, device)
        if self.sd_scorer == "random_noise":
            return self.score_random_noise(input_ids, device)
        raise ValueError(
            f"Unknown sd_scorer={self.sd_scorer!r}; "
            "expected inv_freq, pos_inv_freq, or random_noise"
        )

    # ── Combined ─────────────────────────────────────────────────────
    @torch.no_grad()
    def score_all(
        self,
        logits:     torch.Tensor,               # [B, T, V] float32
        input_ids:  torch.Tensor,               # [B, T]
        weights:    torch.Tensor,               # [K] simplex
        ref_losses: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                          # [B, T]
        device = logits.device

        cache: dict[str, torch.Tensor] = {}
        selected = []
        for name in self.scorer_names:
            if name not in cache:
                if name == "S_E":
                    cache[name] = self.score_excess_loss(logits, input_ids, ref_losses)
                elif name == "S_U":
                    cache[name] = self.score_entropy(logits)
                elif name == "S_L":
                    cache[name] = self.score_logit_spread(logits)
                elif name == "S_D":
                    cache[name] = self.score_sd(input_ids, device)
                else:
                    raise ValueError(f"Unknown scorer name {name!r}")
            selected.append(cache[name])

        # Stack → [B, T, K]
        raw  = torch.stack(selected, dim=-1)

        # Z-score normalise each scorer over the batch tokens (paper §3).
        # Column 0 has no loss term (padding), so it is excluded from the
        # statistics; it is never selected by MaskCache either.
        inner = raw[:, 1:]
        mean = inner.mean(dim=(0, 1), keepdim=True)       # [1, 1, K]
        std  = inner.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        norm = (raw - mean) / std

        # Weighted sum → [B, T]
        return (norm * weights.to(device).view(1, 1, -1)).sum(dim=-1)
