"""Validation loss and perplexity."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from ..data import MemmapDataset


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset: MemmapDataset, device: torch.device,
             batch_size: int, max_batches: Optional[int], autocast_ctx) -> float:
    """Mean token NLL over the validation windows (all of them if max_batches is None)."""
    model.eval()
    total_nll, total_tok = 0.0, 0
    for ids in dataset.val_batches(batch_size, device, max_batches):
        with autocast_ctx:
            logits = model(input_ids=ids, use_cache=False).logits
        V = logits.size(-1)
        total_nll += F.cross_entropy(logits[:, :-1].float().reshape(-1, V),
                                     ids[:, 1:].reshape(-1), reduction="sum").item()
        total_tok += ids.shape[0] * (ids.shape[1] - 1)
    model.train()
    return total_nll / max(total_tok, 1)


def perplexity(mean_nll: float) -> float:
    return math.exp(mean_nll)
