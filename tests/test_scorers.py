"""Scorers follow the paper's definitions and the column-t = token-t convention."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sieve.scorers import SCORERS, ScoringInputs, build_scorer

V = 64


@pytest.fixture()
def inputs():
    torch.manual_seed(0)
    B, T = 2, 7
    return ScoringInputs(torch.randint(0, V, (B, T)), torch.randn(B, T, V), torch.rand(B, T))


def test_model_state_scorers_describe_the_selected_token(inputs):
    s_e = SCORERS["excess_loss"]().score(inputs)
    s_u = SCORERS["entropy"]().score(inputs)
    s_l = SCORERS["logit_margin"]().score(inputs)
    for t in range(1, inputs.input_ids.shape[1]):
        prev = inputs.logits[:, t - 1]                       # predicts token t
        p = F.softmax(prev, -1)
        assert torch.allclose(s_u[:, t], -(p * p.log()).sum(-1), atol=1e-5)
        top2 = prev.topk(2).values
        assert torch.allclose(s_l[:, t], -(top2[:, 0] - top2[:, 1]), atol=1e-6)
        nll = F.cross_entropy(prev, inputs.input_ids[:, t], reduction="none")
        assert torch.allclose(s_e[:, t], nll - inputs.ref_nll[:, t], atol=1e-5)   # no clamp


def test_excess_loss_requires_reference(inputs):
    with pytest.raises(ValueError):
        SCORERS["excess_loss"]().score(ScoringInputs(inputs.input_ids, inputs.logits, None))
    assert SCORERS["excess_loss"].needs_reference
    assert not any(SCORERS[n].needs_reference for n in SCORERS if n != "excess_loss")


def test_corpus_scorers(data_dir):
    cpu = torch.device("cpu")
    ids = torch.tensor([[3, 7, 7, 1]])
    x = ScoringInputs(ids, torch.zeros(1, 4, V))
    freq = torch.load(data_dir / "freq.pt")
    assert torch.equal(build_scorer("frequency", str(data_dir), 0, cpu).score(x), freq[ids])
    bigram = build_scorer("bigram", str(data_dir), 0, cpu).score(x)
    table = torch.load(data_dir / "bigram_freq.pt", weights_only=False)
    lookup = dict(zip(table["keys"].tolist(), table["log_freq"].tolist()))
    for t in range(1, 4):
        key = ids[0, t - 1].item() * 50257 + ids[0, t].item()
        expected = -lookup.get(key, table["fallback_log_freq"])
        assert bigram[0, t].item() == pytest.approx(expected, rel=1e-5)
    assert bigram[0, 0].item() == pytest.approx(bigram[0, 1:].mean().item(), rel=1e-5)


def test_noise_is_seeded_and_fresh_at_each_call(data_dir):
    cpu = torch.device("cpu")
    x = ScoringInputs(torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 5, V))
    a, b = build_scorer("noise", "", 7, cpu), build_scorer("noise", "", 7, cpu)
    first = a.score(x)
    assert torch.equal(first, b.score(x))          # same seed → same draws
    assert not torch.equal(first, a.score(x))      # new draws at every call


def test_missing_tables_are_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_scorer("frequency", str(tmp_path), 0, torch.device("cpu"))
    with pytest.raises(ValueError):
        build_scorer("unknown", str(tmp_path), 0, torch.device("cpu"))
