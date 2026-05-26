"""
Patience-based early stopping for SIEVE training.

Stops training when curriculum converges, as measured by 2-of-3 signals:
  1. Val PPL plateau:      min-PPL not improving by > ppl_tol over window
  2. Bandit weight stability: L1 distance of expected weights < weight_tol
  3. Dirichlet entropy:    relative entropy change < entropy_tol

Rationale: any single signal has a failure mode. PPL alone misses curriculum
transitions that briefly tick PPL up. Weights alone can stabilise while PPL
still drops (exploit phase). Entropy alone can be trivially satisfied by a
dead bandit. Requiring 2-of-3 avoids all three failure modes.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import torch


class PatienceMonitor:
    def __init__(
        self,
        window: int = 5,
        min_steps: int = 5000,
        ppl_tol: float = 0.005,
        weight_tol: float = 0.05,
        entropy_tol: float = 0.05,
        mode: str = "2_of_3",
        verbose: bool = True,
    ):
        if mode not in ("2_of_3", "ppl_only", "weights_only", "entropy_only"):
            raise ValueError(f"Unknown patience mode: {mode}")
        self.window = window
        self.min_steps = min_steps
        self.ppl_tol = ppl_tol
        self.weight_tol = weight_tol
        self.entropy_tol = entropy_tol
        self.mode = mode
        self.verbose = verbose

        self._ppl_hist = deque(maxlen=window + 1)
        self._weight_hist = deque(maxlen=window + 1)
        self._entropy_hist = deque(maxlen=window + 1)
        self._steps = deque(maxlen=window + 1)

        self._last = {
            "ppl_rel_improve": float("nan"),
            "weight_l1": float("nan"),
            "entropy_rel": float("nan"),
            "signals_stable": 0,
        }

    def update(
        self,
        step: int,
        val_ppl: float,
        bandit_weights: Optional[torch.Tensor] = None,
        bandit_entropy: Optional[float] = None,
    ) -> None:
        """Called after each evaluate() in the training loop."""
        self._steps.append(step)
        self._ppl_hist.append(float(val_ppl))
        if bandit_weights is not None:
            self._weight_hist.append(bandit_weights.detach().cpu().clone())
        if bandit_entropy is not None and math.isfinite(bandit_entropy):
            self._entropy_hist.append(float(bandit_entropy))

        self._recompute_signals()

    def should_stop(self, step: int) -> tuple[bool, str]:
        """Return (stop, reason). `reason` is always set, useful for logging."""
        if step < self.min_steps:
            return False, f"below min_steps ({self.min_steps})"

        if len(self._ppl_hist) <= self.window:
            return False, f"warming up ({len(self._ppl_hist)}/{self.window + 1})"

        ppl_stable = self._ppl_stable()
        weight_stable = self._weight_stable()
        entropy_stable = self._entropy_stable()
        n_stable = int(ppl_stable) + int(weight_stable) + int(entropy_stable)

        self._last["signals_stable"] = n_stable

        if self.mode == "2_of_3":
            if n_stable >= 2:
                reason = (
                    f"2-of-3: ppl={ppl_stable} weights={weight_stable} "
                    f"entropy={entropy_stable}"
                )
                return True, reason
        elif self.mode == "ppl_only":
            if ppl_stable:
                return (
                    True,
                    f"ppl plateau (rel_improve={self._last['ppl_rel_improve']:.4f})",
                )
        elif self.mode == "weights_only":
            if weight_stable:
                return True, f"weight stable (L1={self._last['weight_l1']:.4f})"
        elif self.mode == "entropy_only":
            if entropy_stable:
                return True, f"entropy stable (rel={self._last['entropy_rel']:.4f})"

        return False, f"{n_stable}/3 signals stable"

    def log_dict(self) -> dict:
        """Emit signal values with a stable prefix for filtering."""
        return {f"patience/{k}": v for k, v in self._last.items()}

    def _recompute_signals(self) -> None:
        if len(self._ppl_hist) > self.window:
            hist = list(self._ppl_hist)
            old = hist[:- self.window]
            new = hist[-self.window :]
            old_min = min(old)
            new_min = min(new)
            rel_improve = (old_min - new_min) / max(old_min, 1e-9)
            self._last["ppl_rel_improve"] = float(rel_improve)

        if len(self._weight_hist) > self.window:
            w_now = self._weight_hist[-1]
            w_old = self._weight_hist[0]
            l1 = torch.norm(w_now - w_old, p=1).item()
            self._last["weight_l1"] = float(l1)

        if len(self._entropy_hist) > self.window:
            e_now = self._entropy_hist[-1]
            e_old = self._entropy_hist[0]
            rel_e = abs(e_now - e_old) / max(abs(e_old), 1e-9)
            self._last["entropy_rel"] = float(rel_e)

    def _ppl_stable(self) -> bool:
        rel = self._last.get("ppl_rel_improve", float("nan"))
        return math.isfinite(rel) and rel < self.ppl_tol

    def _weight_stable(self) -> bool:
        l1 = self._last.get("weight_l1", float("nan"))
        return math.isfinite(l1) and l1 < self.weight_tol

    def _entropy_stable(self) -> bool:
        rel = self._last.get("entropy_rel", float("nan"))
        return math.isfinite(rel) and rel < self.entropy_tol
