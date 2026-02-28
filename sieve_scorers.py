from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def score_excess_loss(logits: Tensor, targets: Tensor,
                      ref_logits: Optional[Tensor] = None) -> Tensor:
    model_loss = F.cross_entropy(logits, targets, reduction="none")
    if ref_logits is not None:
        ref_loss = F.cross_entropy(ref_logits, targets, reduction="none")
        return model_loss - ref_loss
    return model_loss


@torch.no_grad()
def score_uncertainty(logits: Tensor) -> Tensor:
    p = F.softmax(logits.float(), dim=-1)
    return -(p * torch.log(p + 1e-10)).sum(dim=-1)


@torch.no_grad()
def score_attn_entropy(logits: Tensor) -> Tensor:
    std = logits.float().std(dim=-1)
    topk = torch.topk(logits.float(), k=min(10, logits.size(-1)), dim=-1)
    spread = topk.values[:, 0] - topk.values[:, -1]
    return std / (spread + 1e-6)


@torch.no_grad()
def score_diversity(targets: Tensor, vocab_size: int) -> Tensor:
    counts = torch.zeros(vocab_size, device=targets.device)
    counts.scatter_add_(0, targets, torch.ones_like(targets, dtype=counts.dtype))
    return 1.0 / (counts[targets] + 1.0)


@torch.no_grad()
def compute_all_scores(logits: Tensor, targets: Tensor,
                       ref_logits: Optional[Tensor] = None) -> dict[str, Tensor]:
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    if ref_logits is not None and ref_logits.dim() == 3:
        ref_logits = ref_logits.squeeze(0)

    V = logits.size(-1)
    return {
        "excess_loss":  score_excess_loss(logits, targets, ref_logits),
        "uncertainty":  score_uncertainty(logits),
        "attn_entropy": score_attn_entropy(logits),
        "diversity":    score_diversity(targets, V),
    }


def normalize_scores(scores: dict[str, Tensor]) -> dict[str, Tensor]:
    out = {}
    for k, s in scores.items():
        lo, hi = s.min(), s.max()
        out[k] = (s - lo) / (hi - lo + 1e-8)
    return out


def combine_scores(scores: dict[str, Tensor], weights: Tensor,
                   strategy_names: tuple) -> Tensor:
    combined = torch.zeros_like(next(iter(scores.values())))
    for i, name in enumerate(strategy_names):
        if name in scores:
            combined += weights[i].item() * scores[name]
    return combined
