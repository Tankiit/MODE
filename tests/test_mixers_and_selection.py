"""Mixture updates (paper Eq. update / vnts), utility, top-k and the selector."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sieve.mixers import DirichletMixer, MIXERS
from sieve.mixers.dirichlet import VNDirichletMixer
from sieve.mixers.fixed import FixedMixer
from sieve.scorers import ScoringInputs
from sieve.scorers.base import Scorer
from sieve.selection import TokenSelector, mixed_utility, retained_count, topk_mask


def test_dirichlet_update_is_alpha_plus_r_w():
    m = DirichletMixer(4, seed=0)
    w = m.sample().numpy().astype(np.float64)
    m.update(0.3)
    assert np.allclose(m.alpha, 1 + 0.3 * w, atol=1e-6)
    m.update(0.0)                                    # no discount, no floor
    assert np.allclose(m.alpha, 1 + 0.3 * w, atol=1e-6)
    with pytest.raises(ValueError):
        m.update(-0.1)


def test_vn_update_matches_equation():
    m = VNDirichletMixer(3, seed=1, eps=1e-3, sigma_warmup=2)
    alpha, contrib = np.ones(3), []
    for r in [0.2, 0.05, 0.1]:
        w = m.sample().numpy().astype(np.float64)
        m.update(r)
        contrib.append(r * w)
        sigma = np.ones(3) if len(contrib) < 2 else np.std(np.stack(contrib), axis=0)
        alpha = alpha + r * w / (sigma + 1e-3)
        assert np.allclose(m.alpha, alpha, rtol=1e-5)


def test_fixed_mixer():
    assert np.allclose(FixedMixer(4).expected_weights(), 0.25)
    assert torch.equal(FixedMixer(1, [1.0]).sample(), torch.tensor([1.0]))
    with pytest.raises(ValueError):
        FixedMixer(2, [1.0])
    assert set(MIXERS) == {"fixed", "dirichlet", "vn_dirichlet"}


def test_utility_zscore_excludes_column_zero_and_topk():
    torch.manual_seed(0)
    raw = torch.randn(3, 11, 2)
    raw[:, 0] = 1e6                                   # padding column must not affect stats
    u = mixed_utility(raw, torch.tensor([1.0, 0.0]))
    assert abs(u[:, 1:].mean().item()) < 1e-5 and abs(u[:, 1:].std().item() - 1) < 1e-4
    mask = topk_mask(u, 0.5)
    assert not mask[:, 0].any()
    assert (mask.sum(1) == retained_count(11, 0.5)).all() and retained_count(11, 0.5) == 5
    assert retained_count(1024, 0.7) == 717 and retained_count(1024, 0.3) == 307
    picked = u[:, 1:].topk(5).indices + 1
    assert all(mask[b, picked[b]].all() for b in range(3))


class _Const(Scorer):
    name, slot = "const", "S_X"

    def __init__(self, v):
        self.v = v

    def score(self, x):
        return torch.full(x.input_ids.shape, self.v)


def test_selector_resamples_every_interval_and_credits_active_weights():
    sel = TokenSelector([_Const(1.0), _Const(2.0)], DirichletMixer(2, seed=0), 0.5, rescore_interval=10)
    x = ScoringInputs(torch.zeros(1, 5, dtype=torch.long), torch.zeros(1, 5, 3))
    for step in range(30):
        sel.begin_step(step)
        sel.select(x)
    assert [s["step"] for s in sel.sample_log] == [0, 10, 20]
    w20 = np.array(sel.sample_log[-1]["weights"])
    sel.on_validation(3.0, step=30)                   # first event: no reward
    sel.on_validation(3.5, step=40)                   # regression: r = 0
    assert np.allclose(sel.mixer.alpha, 1.0)
    sel.on_validation(3.2, step=50)                   # improvement: r = 0.3 on active weights
    assert np.allclose(sel.mixer.alpha, 1 + 0.3 * w20, atol=1e-6)
    assert [e["reward"] for e in sel.reward_log] == pytest.approx([0.0, 0.3])
