"""Token scorers aligned to target positions; column zero is not supervised."""
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
        bigram_table: Optional[dict] = None,
        topk_spread: int = 10,
        sd_scorer:   str = "inv_freq",
        scorer_names: Optional[list[str]] = None,
    ):
        self.freq_table  = freq_table
        self.bigram_table = bigram_table
        self.topk_spread = topk_spread
        self.sd_scorer   = sd_scorer
        self.scorer_names = scorer_names or ["S_E", "S_U", "S_L", "S_D"]
        self._noise_gen: Optional[torch.Generator] = None

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
        # Avoid in-place aliasing: log_p must remain log probabilities.
        log_p = F.log_softmax(logits[:, :-1], dim=-1)
        entropy = -(log_p.exp() * log_p).sum(dim=-1)
        return F.pad(entropy, (1, 0), value=0.0)

    # ── S_L: Inverted Logit Spread ───────────────────────────────────
    @torch.no_grad()
    def score_logit_spread(
        self,
        logits: torch.Tensor,   # [B, T, V] float32
    ) -> torch.Tensor:          # [B, T]
        # topk(10) only loads 10/50257 values — no V-dim allocation
        topk, _ = logits[:, :-1].topk(min(self.topk_spread, logits.size(-1)), dim=-1)   # [B, T, k]
        spread   = topk[..., 0] - topk[..., -1]
        return F.pad(1.0 / (spread + 1e-6), (1, 0), value=0.0)

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

        # Z-score normalise each scorer across (batch, time)
        # Prevents high-variance scorers from dominating weighted sum
        mean = raw[:, 1:].mean(dim=(0, 1), keepdim=True)         # [1, 1, 4]
        std  = raw[:, 1:].std(dim=(0, 1), keepdim=True, unbiased=False).clamp(min=1e-6)
        norm = (raw - mean) / std

        # Weighted sum → [B, T]
        return (norm * weights.to(device).view(1, 1, -1)).sum(dim=-1)
