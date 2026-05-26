# SIEVE: Selective Token-Level Training for Language Models

This repository contains a compact, anonymous implementation of SIEVE-style
selective token training for causal language models.

SIEVE keeps the full causal context but computes the language-modeling loss only
on selected token positions. Token selection is driven by a mixture of scorers
and a contextual Dirichlet Thompson-sampling bandit that adapts the scorer
weights over training.

## What Is Included

- `sieve/scorers.py`: token-level scoring functions
- `sieve/bandit.py`: contextual Dirichlet Thompson sampling
- `sieve/mask.py`: per-sequence top-k token mask cache
- `sieve/loss.py`: gradient-invariant selective cross-entropy
- `sieve/state.py`: SIEVE state object tying scorers, bandit, and masks together
- `sieve/loop.py`: framework-free training and evaluation loop
- `sieve/rho1.py`: rho-1 style excess-loss baseline
- `prepare_sieve.py`: memmap data preparation for WikiText-2
- `train_sieve.py`: local training entrypoint for SIEVE and baselines

The repository intentionally excludes generated outputs, model artifacts,
machine-specific paths, and service-specific execution scripts.

## Installation

Use Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable package metadata:

```bash
pip install -e .
```

## Quick Start

Prepare WikiText-2 into local memmap files:

```bash
python prepare_sieve.py --dataset wikitext2 --data_dir ./data/wikitext2
```

Run a short SIEVE smoke test:

```bash
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 50 --baseline sieve
```

Compare against baselines:

```bash
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 500 --baseline clm
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 500 --baseline rho1
python train_sieve.py --dataset wikitext2 --model gpt2 --steps 500 --baseline sieve
```

## Methods

SIEVE combines four token scorers:

- `S_E`: excess loss, using precomputed reference losses when available or
  current token loss as a fallback
- `S_U`: prediction entropy
- `S_L`: inverted logit spread
- `S_D`: inverse token frequency or configured rarity variant

The combined score is:

```text
score = w_E * S_E + w_U * S_U + w_L * S_L + w_D * S_D
```

The bandit learns phase-dependent scorer weights. Selection is per sequence,
not global across the batch, so each sequence contributes approximately the
same selected-token fraction.

## Loss Normalization

The selective loss is normalized by the full causal-LM token count,
`B * (T - 1)`, rather than the number of selected tokens:

```text
loss = sum(selected token losses) / (B * (T - 1))
```

This avoids accidentally changing the effective learning rate when the
selection ratio changes.

## Common Options

```bash
python train_sieve.py \
  --dataset wikitext2 \
  --model gpt2 \
  --steps 10000 \
  --alpha 0.7 \
  --baseline sieve \
  --seed 42
```

Useful flags:

- `--alpha`: fraction of token positions selected for loss
- `--baseline`: `sieve`, `rho1`, `clm`, `scalarization`, `sieve_no_se`,
  `su_only`, `sl_only`, or `sd_only`
- `--eval_interval`: validation interval in steps
- `--data_fraction`: fraction of the training memmap to sample from
- `--variance_normalized_bandit`: enable variance-normalized bandit updates

## Data Format

`prepare_sieve.py` creates:

- `train.bin`: flat `uint16` token array
- `val.bin`: flat `uint16` token array
- `meta.json`: dataset metadata
- `freq.pt`: token rarity table for `S_D`
- `ref_losses.bin`: optional reference losses for `S_E`

To add another dataset, extend `prepare_dataset()` in `prepare_sieve.py` and
write the same files into a dataset directory.

## Repository Hygiene

The `.gitignore` excludes local data, logs, caches, virtual environments, and
model outputs. Commit only source files, documentation, and lightweight
configuration needed to reproduce the procedure.
