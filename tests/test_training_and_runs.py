"""Loss, schedule, configuration, run directories and end-to-end training (tiny model, CPU)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

import main as entry
from sieve.config import config_hash, load_config
from sieve.training import lr_factor, masked_mean_nll

V = 64


def test_loss_is_mean_over_selected_tokens():
    torch.manual_seed(0)
    logits, ids = torch.randn(2, 6, V), torch.randint(0, V, (2, 6))
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, [1, 4]] = mask[1, [2, 5]] = True
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1), reduction="none").reshape(2, 5)
    assert torch.allclose(masked_mean_nll(logits, ids, mask), nll[mask[:, 1:]].mean(), atol=1e-6)
    assert torch.allclose(masked_mean_nll(logits, ids), nll.mean(), atol=1e-6)


def test_lr_schedule():
    assert lr_factor(0, 500, 25000, 0.1) == pytest.approx(1 / 500)
    assert lr_factor(499, 500, 25000, 0.1) == pytest.approx(1.0)
    assert lr_factor(25000, 500, 25000, 0.1) == pytest.approx(0.1)
    vals = [lr_factor(s, 500, 25000, 0.1) for s in range(500, 25000, 500)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_config_merge_override_and_strictness(tiny_config):
    path = tiny_config("vn_mix")
    cfg = load_config([str(path)], ["seed=43", "selection.alpha=0.5"])
    assert (cfg.seed, cfg.selection.alpha, cfg.selection.mixer) == (43, 0.5, "vn_dirichlet")
    assert cfg.model.name == "tiny" and cfg.training.lr == 3e-4
    with pytest.raises(ValueError, match="unknown config key"):
        load_config([str(path)], ["training.lerning_rate=1"])
    with pytest.raises(ValueError):
        load_config([str(path)], ["selection.alpha=1.5"])
    # the hash identifies the experiment: output location and name do not count
    base = load_config([str(path)])
    moved = load_config([str(path)], ["output.root=elsewhere", "name=x"])
    assert config_hash(moved) == config_hash(base)
    assert config_hash(load_config([str(path)], ["seed=43"])) != config_hash(base)


def _run(path, *extra):
    return entry.main(["--config", str(path), *extra])


def test_end_to_end_mix4_scores_every_step_and_credits_last_block(tiny_config, tmp_path):
    path = tiny_config("mix4")
    assert _run(path) == 0
    (run,) = list((tmp_path / "runs").iterdir())
    for f in ["config.yaml", "git.txt", "env.txt", "train.log", "history.json", "DONE"]:
        assert (run / f).exists(), f
    h = json.loads((run / "history.json").read_text())
    assert h["n_scored_steps"] == 60
    assert [s["step"] for s in h["sample_log"]] == [0, 10, 20, 30, 40, 50]
    assert h["eval_steps"] == [20, 40, 60]
    assert [r["step"] for r in h["reward_log"]] == [40, 60]
    w = {s["step"]: np.array(s["weights"]) for s in h["sample_log"]}
    r = {e["step"]: e["reward"] for e in h["reward_log"]}
    # the evaluation after step 40 credits the weights sampled at step 30, etc.
    assert np.allclose(h["mixer_state"]["alpha"], 1 + r[40] * w[30] + r[60] * w[50], atol=1e-5)
    assert h["slots"] == ["S_E", "S_U", "S_L", "S_D"]
    assert math.isfinite(h["final_val_ppl"])
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    assert cfg["config_hash"] == json.loads((run / "DONE").read_text())["config_hash"]


def test_runs_are_never_overwritten_and_skip_if_done(tiny_config, tmp_path):
    path = tiny_config("se_only")
    _run(path)
    _run(path, "--skip-if-done")                       # same config, completed → skipped
    assert len(list((tmp_path / "runs").iterdir())) == 1
    _run(path)                                         # explicit rerun → new directory
    assert len(list((tmp_path / "runs").iterdir())) == 2
    _run(path, "--skip-if-done", "--set", "seed=43")   # different config → runs
    assert len(list((tmp_path / "runs").iterdir())) == 3


@pytest.mark.parametrize("method", ["clm", "su_only", "mix3", "vn_mix", "mix4_bigram", "mix4_noise", "uniform4"])
def test_every_method_runs(tiny_config, tmp_path, method):
    path = tiny_config(method)
    assert _run(path, "--set", "training.steps=20") == 0
    (run,) = list((tmp_path / "runs").iterdir())
    h = json.loads((run / "history.json").read_text())
    assert h["n_scored_steps"] == (0 if method == "clm" else 20)
    assert math.isfinite(h["final_val_ppl"])
