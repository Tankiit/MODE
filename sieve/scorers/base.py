"""
Scorer interface.

A scorer maps the current batch to a [B, T] utility tensor. Alignment
convention (shared by every scorer): column t scores token t, i.e. the loss
term "predict input_ids[t] from logits[t-1]" that mask[t] selects. Column 0
has no loss term; it is padding and is never selected.

`slot` is the paper's arm name (S_E, S_U, S_L, S_D). The bigram and noise
controls replace the S_D slot, so they share its name in the logs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class ScoringInputs:
    input_ids: torch.Tensor                 # [B, T] int64
    logits: torch.Tensor                    # [B, T, V] float32, detached, trained model
    ref_nll: Optional[torch.Tensor] = None  # [B, T] reference NLL, column t = token t


class Scorer(ABC):
    name: str                 # registry key, e.g. "entropy"
    slot: str                 # paper arm: S_E | S_U | S_L | S_D
    needs_reference: bool = False

    @abstractmethod
    def score(self, x: ScoringInputs) -> torch.Tensor:
        """[B, T] float32; column t scores token t; column 0 is ignored."""


def pad_first_column(x: torch.Tensor) -> torch.Tensor:
    """[B, T-1] values for tokens 1..T-1 → [B, T] with a zero column 0."""
    return F.pad(x, (1, 0), value=0.0)
