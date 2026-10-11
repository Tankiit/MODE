# SIEVE: Selective Token-Level Training for Language Models

This repository contains a compact implementation of selective token training
for causal language models, used for the paper "Do Adaptive Scorer Mixtures
Improve Token Selection? A Controlled Study in GPT-2".

The model keeps the full causal context but computes the language-modeling loss
only on selected token positions. Token selection is driven by a mixture of four
scorers whose weights are sampled from a Dirichlet distribution updated from
validation feedback. To rerun every experiment of the paper, follow
`RERUN_PLAN.md`.

## What Is Included

- `sieve/scorers.py`: the four token scorers (S_E, S_U, S_L, S_D) and the S_D controls
- `sieve/reference.py`: frozen reference model for S_E, run on each batch
- `sieve/bandit.py`: Dirichlet mixture over scorers (paper Eq. update; VN-Mix)
- `sieve/mask.py`: per-sequence top-k token mask, recomputed on the current batch at every step
- `sieve/loss.py`: cross-entropy averaged over the selected tokens
- `sieve/state.py`: state object tying scorers, mixture and masks together
- `sieve/loop.py`: framework-free training and evaluation loop
- `sieve/rho1.py`: excess-loss-only (S_E-only) baseline
- `scripts/prepare_data.py`: memmap data preparation (WikiText-103, WikiText-2, OpenWebMath)
- `train_sieve.py`: training entrypoint for the mixtures and baselines
- `scripts/make_commands.sh`, `scripts/run_array.slurm`: the paper's run grid
- `scripts/aggregate.py`, `scripts/plot_bandit.py`, `scripts/analyze_selection.py`: tables and figures
- `tests/`: CPU tests of the scorers, mixture update, reward, loss and training loop

## Installation

Use Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[liger]" scipy matplotlib pytest
python -m pytest -q tests
```

## Quick Start

Prepare WikiText-2 into local memmap files:

```bash
python scripts/prepare_data.py --dataset wikitext2 --out ./data/wikitext2 --bigram
```

Run a short smoke test:

```bash
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 300 \
    --eval_interval 100 --baseline sieve --out_dir runs/smoke
```

Compare against baselines:

```bash
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 500 --baseline clm
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 500 --baseline rho1
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 500 --baseline sieve
```

## Methods

Four token scorers; column t of every scorer describes token t (the loss
term predicted from position t−1):

- `S_E`: excess loss ℓθ − ℓref, with a frozen reference model (`--ref_model`,
  default GPT-2 Small) run on the same batch
- `S_U`: predictive entropy
- `S_L`: negative logit margin −(logit₁ − logit₂)
- `S_D`: −log corpus frequency, or a control (`--sd_scorer bigram_freq | random_noise`)

Each scorer is z-normalized over the batch tokens, and the utility is

```text
u = w_E * S_E + w_U * S_U + w_L * S_L + w_D * S_D
```

with w ~ Dir(α) sampled every `rescore_interval` (100) steps. At each validation
event (every 1000 steps) the active w receives r = max(0, L_prev − L_val) and
α ← α + r·w. Selection is per sequence (top ⌈αT⌉ positions) and is recomputed
at every step on the current batch.

## Loss Normalization

The selective loss is the mean NLL over the selected tokens (paper Eq.
train_loss). Every sequence retains the same number of tokens, so this equals
the mean of the per-sequence losses.

## Common Options

- `--alpha`: fraction of token positions selected for the loss
- `--baseline`: `sieve`, `rho1`, `clm`, `scalarization`, `sieve_no_se`,
  `su_only`, `sl_only`, or `sd_only`
- `--eval_interval`: validation interval in steps (default 1000)
- `--variance_normalized_bandit`: VN-Mix update
- `--out_dir`: write `history.json` (and `final.pt` with `--save_final`)
- `--log_interval`: steps between log lines and history points

## Data Format

`scripts/prepare_data.py` creates:

- `train.bin`, `val.bin`: flat `uint16` token arrays
- `meta.json`: dataset metadata and construction choices
- `freq.pt`: −log corpus frequency table for `S_D`
- `bigram_freq.pt` (with `--bigram`): log p(tok | prev) for the bigram control

## Repository Hygiene

The `.gitignore` excludes local data, logs, caches, virtual environments, and
model outputs. Commit only source files, documentation, and lightweight
configuration needed to reproduce the procedure.
