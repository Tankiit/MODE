"""CPU-side selection and Tinker adapter. No service calls on import.

The two-scorer mode uses current target NLL and corpus rarity. It is not
the four-scorer method and does not claim to compute excess loss or entropy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .bandit import DirichletBandit


MODES = ("clm", "random", "nll", "rarity", "sieve_nd")


def top_fraction(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    """One row per sequence, already shifted to eligible targets only."""
    if scores.ndim != 2 or scores.shape[1] == 0 or not 0 < ratio <= 1:
        raise ValueError("Expected nonempty [batch, targets] scores and 0 < ratio <= 1")
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite selection scores")
    k = min(scores.shape[1], max(1, round(ratio * scores.shape[1])))
    idx = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def standardize(scores: torch.Tensor) -> torch.Tensor:
    return (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-6)


@dataclass
class Selector:
    mode: str
    ratio: float
    seed: int

    def __post_init__(self):
        if self.mode not in MODES or not 0 < self.ratio <= 1:
            raise ValueError("Invalid mode or selection ratio")
        self.rng = np.random.default_rng(self.seed + 1000)
        self.bandit = DirichletBandit(n_strategies=2, seed=self.seed)
        self.weights = torch.tensor([0.5, 0.5])
        self.prev_val = None
        self.ema = 0.0
        self.reward_count = 0
        self.active = False

    def begin_interval(self, completed: int, max_steps: int):
        if self.active:
            raise RuntimeError("Previous interval has not been evaluated")
        if self.prev_val is None:
            raise RuntimeError("Evaluate the initial model before training")
        if self.mode == "sieve_nd":
            b = self.bandit.encode_context(completed, max_steps, self.prev_val, 0.0)
            self.weights = self.bandit.sample(b)
        self.active = True

    def observe(self, val_nll: float) -> float:
        if not math.isfinite(val_nll):
            raise ValueError("Non-finite validation loss")
        reward = 0.0
        if self.prev_val is not None:
            if not self.active:
                raise RuntimeError("No training interval to credit")
            improvement = self.prev_val - val_nll
            self.ema = 0.99 * self.ema + 0.01 * abs(improvement)
            self.reward_count += 1
            scale = self.ema / (1 - 0.99 ** self.reward_count)
            reward = float(np.clip(improvement / max(scale, 1e-6), -5, 5))
            if self.mode == "sieve_nd":
                self.bandit.update(reward)
        self.prev_val = float(val_nll)
        self.active = False
        return reward

    def static_mask(self, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "clm":
            return torch.ones_like(targets, dtype=torch.bool)
        if self.mode != "random":
            raise ValueError("This mode requires current-model NLL")
        noise = torch.from_numpy(self.rng.random(tuple(targets.shape)))
        return top_fraction(noise, self.ratio)

    def dynamic_mask(self, nll: torch.Tensor, rarity: torch.Tensor) -> torch.Tensor:
        if self.mode == "nll":
            scores = nll
        elif self.mode == "sieve_nd":
            scores = self.weights[0] * standardize(nll) + self.weights[1] * standardize(rarity)
        else:
            raise ValueError("This mode uses a static mask")
        return top_fraction(scores.detach(), self.ratio)

    def state_dict(self) -> dict:
        return {
            "mode": self.mode, "ratio": self.ratio, "seed": self.seed,
            "rng": self.rng.bit_generator.state,
            "alpha": self.bandit.alpha.tolist(),
            "bandit_rng": self.bandit.rng.bit_generator.state,
            "bin": self.bandit._current_bin,
            "sample": (None if self.bandit._current_weights is None
                       else self.bandit._current_weights.tolist()),
            "weights": self.weights.tolist(), "prev_val": self.prev_val,
            "ema": self.ema, "reward_count": self.reward_count, "active": self.active,
        }

    def load_state_dict(self, state: dict):
        for name in ("mode", "ratio", "seed"):
            if state[name] != getattr(self, name):
                raise ValueError(f"Selector resume mismatch: {name}")
        self.rng.bit_generator.state = state["rng"]
        self.bandit.alpha = np.asarray(state["alpha"], dtype=np.float64)
        self.bandit.rng.bit_generator.state = state["bandit_rng"]
        self.bandit._current_bin = state["bin"]
        self.bandit._current_weights = (None if state["sample"] is None
                                       else np.asarray(state["sample"], dtype=np.float64))
        self.weights = torch.tensor(state["weights"], dtype=torch.float32)
        self.prev_val = state["prev_val"]
        self.ema = state["ema"]
        self.reward_count = state["reward_count"]
        self.active = state["active"]


def make_data(ids: torch.Tensor, mask: torch.Tensor | None = None):
    """Tinker shifts are explicit: x[:-1] predicts x[1:].

    Its CE uses a sum reduction, so weights include 1 / all target tokens.
    No token is removed from the input context.
    """
    import tinker
    if ids.ndim != 2 or ids.shape[1] < 2:
        raise ValueError("Need [batch, sequence] with sequence >= 2")
    targets = ids[:, 1:]
    if mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    if mask.shape != targets.shape:
        raise ValueError("Mask must be aligned with ids[:, 1:]")
    weights = mask.float() / targets.numel()
    return [tinker.Datum(
        model_input=tinker.ModelInput.from_ints(row[:-1].tolist()),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(row[1:].contiguous()),
            "weights": tinker.TensorData.from_torch(w.contiguous()),
        },
    ) for row, w in zip(ids, weights, strict=True)]


def masked_nll(logprobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Same fixed denominator as sieve.loss.selective_loss."""
    return -(logprobs * mask.to(logprobs)).sum() / logprobs.numel()


def selection_loss(selector: Selector, rarity: torch.Tensor, metrics: dict):
    """Pure deterministic callback: no bandit/RNG mutation during API retries."""
    def callback(data, logprobs):
        lp = torch.stack(logprobs)
        nll = -lp.detach()
        mask = selector.dynamic_mask(nll, rarity.to(nll))
        loss = masked_nll(lp, mask)
        values = {
            "train_nll_all": float(nll.mean()),
            "train_loss": float(loss.detach()),
            "selected_fraction": float(mask.float().mean()),
        }
        metrics.update(values)
        return loss, values
    return callback


def train_step(client, ids, selector, frequency, adam_params):
    """Synchronous step, preserving callback -> gradient -> optimizer order."""
    targets = ids[:, 1:]
    if selector.mode in ("clm", "random", "rarity"):
        mask = (top_fraction(frequency[targets], selector.ratio)
                if selector.mode == "rarity" else selector.static_mask(targets))
        result = client.forward_backward(make_data(ids, mask), "cross_entropy").result()
        lp = torch.stack([x["logprobs"].to_torch() for x in result.loss_fn_outputs])
        metrics = {
            "train_nll_all": float(-lp.mean()),
            "train_loss": float(masked_nll(lp, mask)),
            "selected_fraction": float(mask.float().mean()),
            "extra_scoring_forward": 0,
        }
    else:
        metrics = {}
        callback = selection_loss(selector, frequency[targets], metrics)
        client.forward_backward_custom(make_data(ids), callback).result()
        metrics["extra_scoring_forward"] = 1
    client.optim_step(adam_params).result()
    return metrics


def evaluate(client, dataset, batch_size: int, max_batches: int) -> tuple[float, int]:
    """Deterministic validation prefix; always evaluate ALL eligible tokens."""
    total_nll, count = 0.0, 0
    for i, batch in enumerate(dataset.val_batches(batch_size, torch.device("cpu"))):
        if i >= max_batches:
            break
        result = client.forward(make_data(batch["input_ids"]), "cross_entropy").result()
        for output in result.loss_fn_outputs:
            lp = output["logprobs"].to_torch()
            total_nll -= float(lp.double().sum())
            count += lp.numel()
    if not count:
        raise ValueError("No validation targets")
    return total_nll / count, count
