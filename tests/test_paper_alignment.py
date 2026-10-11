"""
CPU tests for the code/paper alignment of RERUN_PLAN.md §1.
Run from the repo root:  python -m pytest -q tests
"""
from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sieve.bandit import DirichletBandit  # noqa: E402
from sieve.config import SieveConfig  # noqa: E402
from sieve.data import MemmapDataset  # noqa: E402
from sieve.device import DeviceCfg  # noqa: E402
from sieve.loop import _lr_factor, train  # noqa: E402
from sieve.loss import selective_loss  # noqa: E402
from sieve.rho1 import Rho1Baseline  # noqa: E402
from sieve.scorers import TokenScorers  # noqa: E402
from sieve.state import SieveState  # noqa: E402

V = 64


# ── Scorers ────────────────────────────────────────────────────────────

def test_model_state_scorers_score_the_selected_token():
    """Column t must describe the loss term (logits[t-1], token t) that mask[t] selects."""
    torch.manual_seed(0)
    B, T = 2, 7
    logits = torch.randn(B, T, V)
    ids = torch.randint(0, V, (B, T))
    ref = torch.rand(B, T)
    sc = TokenScorers()
    s_u = sc.score_entropy(logits.clone())
    s_l = sc.score_logit_spread(logits)
    s_e = sc.score_excess_loss(logits, ids, ref)
    for t in range(1, T):
        p = F.softmax(logits[:, t - 1], dim=-1)
        assert torch.allclose(s_u[:, t], -(p * p.log()).sum(-1), atol=1e-5)
        top2 = logits[:, t - 1].topk(2).values
        assert torch.allclose(s_l[:, t], -(top2[:, 0] - top2[:, 1]), atol=1e-6)
        nll = F.cross_entropy(logits[:, t - 1], ids[:, t], reduction="none")
        assert torch.allclose(s_e[:, t], nll - ref[:, t], atol=1e-5)   # no clamp


def test_se_requires_reference_and_sd_requires_table():
    sc = TokenScorers()
    with pytest.raises(ValueError):
        sc.score_excess_loss(torch.randn(1, 4, V), torch.zeros(1, 4, dtype=torch.long), None)
    with pytest.raises(ValueError):
        sc.score_inv_freq(torch.zeros(1, 4, dtype=torch.long), torch.device("cpu"))


def test_zscore_excludes_padding_column():
    sc = TokenScorers(freq_table=torch.arange(V, dtype=torch.float32), scorer_names=["S_D"])
    ids = torch.randint(0, V, (3, 9))
    u = sc.score_all(torch.randn(3, 9, V), ids, torch.tensor([1.0]))
    assert abs(u[:, 1:].mean().item()) < 1e-5
    assert abs(u[:, 1:].std().item() - 1.0) < 1e-4


# ── Mixture update and reward ──────────────────────────────────────────

def test_dirichlet_update_is_alpha_plus_r_w():
    b = DirichletBandit(n_strategies=4, seed=0)
    w = b.sample().numpy().astype(np.float64)
    b.update(0.3)
    assert np.allclose(b.alpha[0], 1.0 + 0.3 * w, atol=1e-6)
    b.update(0.0)                                       # no floor, no discount
    assert np.allclose(b.alpha[0], 1.0 + 0.3 * w, atol=1e-6)
    with pytest.raises(ValueError):
        b.update(-0.1)


def test_vn_mix_update():
    b = DirichletBandit(n_strategies=3, seed=1, variance_normalized=True, vn_eps=1e-3, sigma_warmup=2)
    alpha = np.ones(3)
    contrib = []
    for r in [0.2, 0.05, 0.1]:
        w = b.sample().numpy().astype(np.float64)
        b.update(r)
        contrib.append(r * w)
        sigma = np.ones(3) if len(contrib) < 2 else np.std(np.stack(contrib), axis=0)
        alpha = alpha + r * w / (sigma + 1e-3)
        assert np.allclose(b.alpha[0], alpha, rtol=1e-5)


def test_reward_is_clipped_validation_improvement():
    st = SieveState(SieveConfig(), torch.device("cpu"))
    st.step_begin(0, 100, 0.0, 0.0)
    w = st.bandit._current_weights.copy()
    st.on_eval(3.0, step=10)                  # first evaluation: no reward
    assert np.allclose(st.bandit.alpha[0], 1.0)
    st.on_eval(3.2, step=20)                  # regression: r = 0
    assert np.allclose(st.bandit.alpha[0], 1.0)
    st.on_eval(2.9, step=30)                  # improvement: r = 0.3
    assert np.allclose(st.bandit.alpha[0], 1.0 + 0.3 * w, atol=1e-6)
    assert [e["reward"] for e in st.reward_log] == pytest.approx([0.0, 0.3])


# ── Loss and schedule ─────────────────────────────────────────────────

def test_selective_loss_is_mean_over_selected_tokens():
    torch.manual_seed(0)
    logits, ids = torch.randn(2, 6, V), torch.randint(0, V, (2, 6))
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, [1, 4]] = True
    mask[1, [2, 5]] = True
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1),
                          reduction="none").reshape(2, 5)
    expected = nll[mask[:, 1:]].mean()
    assert torch.allclose(selective_loss(logits, ids, mask), expected, atol=1e-6)
    assert torch.allclose(selective_loss(logits, ids, None), nll.mean(), atol=1e-6)


def test_lr_schedule_warmup_then_cosine_to_ten_percent():
    cfg = SieveConfig(max_steps=25_000, warmup_steps=500, min_lr_ratio=0.1)
    assert _lr_factor(0, cfg) == pytest.approx(1 / 500)
    assert _lr_factor(499, cfg) == pytest.approx(1.0)
    assert _lr_factor(500, cfg) == pytest.approx(1.0)
    assert _lr_factor(25_000, cfg) == pytest.approx(0.1)
    vals = [_lr_factor(s, cfg) for s in range(500, 25_000, 500)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


# ── End-to-end training loop ──────────────────────────────────────────

class TinyLM(nn.Module):
    """Stand-in for an HF CausalLM: returns .logits (and .loss with labels)."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 32)
        self.head = nn.Linear(32, V)

    def forward(self, input_ids, labels=None, use_cache=False,
                output_hidden_states=False, return_dict=True):
        h = self.emb(input_ids)
        h = torch.cumsum(h, 1) / torch.arange(1, h.size(1) + 1).view(1, -1, 1)
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1),
                                   ignore_index=-100)
        return types.SimpleNamespace(logits=logits, loss=loss)


@pytest.fixture()
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    rng.integers(0, V, 50_000, dtype=np.uint16).tofile(tmp_path / "train.bin")
    rng.integers(0, V, 5_000, dtype=np.uint16).tofile(tmp_path / "val.bin")
    (tmp_path / "meta.json").write_text("{}")
    torch.save(torch.rand(V), tmp_path / "freq.pt")
    return tmp_path


def _run(data_dir, selector_factory, steps=300):
    cfg = SieveConfig(max_steps=steps, rescore_interval=100, eval_interval=100, eval_batches=2,
                      log_interval=50, batch_size=2, seq_len=16, warmup_steps=10,
                      data_dir=str(data_dir), freq_table_path=str(data_dir / "freq.pt"))
    dev = DeviceCfg(device=torch.device("cpu"), dtype=torch.float32, backend="cpu", use_amp=False)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        tr, va = MemmapDataset(str(data_dir), "train", cfg), MemmapDataset(str(data_dir), "val", cfg)
    torch.manual_seed(0)
    sel = selector_factory(cfg)
    updates = []
    orig = sel.mask_cache.update
    sel.mask_cache.update = lambda s, step, a=None: updates.append(step) or orig(s, step, a)
    with contextlib.redirect_stdout(io.StringIO()):
        hist = train(TinyLM(), tr, va, cfg, dev, sieve=sel, ref_model=TinyLM())
    return hist, sel, updates


def test_loop_scores_every_step_and_credits_the_active_mixture(data_dir):
    hist, sel, updates = _run(data_dir, lambda cfg: SieveState(cfg, torch.device("cpu")))
    assert updates == list(range(300))                          # mask on every step
    assert [s["step"] for s in hist["sample_log"]] == [0, 100, 200]
    assert hist["eval_steps"] == [100, 200, 300]
    assert [r["step"] for r in hist["reward_log"]] == [200, 300]
    # the reward at step 200 credits the weights sampled at step 100, etc.
    w = {s["step"]: np.array(s["weights"]) for s in hist["sample_log"]}
    r = {e["step"]: e["reward"] for e in hist["reward_log"]}
    expected = 1.0 + r[200] * w[100] + r[300] * w[200]
    assert np.allclose(hist["alpha_final"], expected, atol=1e-5)
    assert math.isfinite(hist["final_val_ppl"])
    json.dumps(hist["sample_log"]), json.dumps(hist["reward_log"])


def test_se_only_baseline_scores_every_step(data_dir):
    hist, sel, updates = _run(data_dir, lambda cfg: Rho1Baseline(selection_ratio=cfg.selection_ratio))
    assert updates == list(range(300))
    assert math.isfinite(hist["final_val_ppl"])
