# Token-level data selection with scorer mixtures

Code for the paper "Do Adaptive Scorer Mixtures Improve Token Selection? A
Controlled Study in GPT-2". The model keeps the full causal context but
computes the language-modeling loss only on the selected tokens. Tokens are
ranked by a weighted mix of per-token scorers; the weights are either fixed or
sampled from a Dirichlet distribution updated from validation feedback.
To rerun every experiment of the paper, follow `RERUN_PLAN.md`.

## Layout

```
main.py                    run one experiment from YAML config files
configs/
  base.yaml                paper defaults (T_rescore = 100, T_val = 1000, AdamW, ...)
  models/                  gpt2, gpt2-large (pretrained checkpoints), tiny (tests)
  data/                    wikitext103, owm, wikitext2
  methods/                 clm, se_only, su_only, sl_only, sd_only, mix4, mix3,
                           vn_mix, mix4_bigram, mix4_noise, uniform4
  experiments/smoke.yaml   quick CPU check
  sweeps/paper.yaml        the 183-run grid of the paper; sweeps/smoke.yaml
sieve/
  config.py                dataclasses, YAML loading (include, --set), validation, config hash
  data/                    memmap token streams
  models/                  model factory, frozen reference model, autocast
  scorers/                 base.py (interface) + one file per scorer:
                           excess_loss, entropy, logit_margin, frequency, bigram, noise
  mixers/                  base.py + fixed.py, dirichlet.py (Dirichlet and VN-Mix)
  selection/               utility (z-norm, weighted sum, top-k) and TokenSelector
  training/                loop, loss, optimizer + schedule, evaluation
  runs/                    one directory per run, metadata, DONE marker
  analysis/                tables (aggregate), mixture figures (bandit_plots),
                           Jaccard / Spearman (selection_overlap)
scripts/
  prepare_data.py          corpora → train.bin, val.bin, freq.pt, bigram_freq.pt
  expand_sweep.py          sweep file → one resolved config per run + run lists
  run_array.slurm          Slurm array over a run list
tests/                     unit and end-to-end tests (CPU); regression/ compares
                           with the reference implementation (branch fix-per-step-mask)
```

Every method is a configuration, not code: a list of scorers plus a mixer.

| Method | Scorers | Mixer |
|---|---|---|
| `se_only` (excess-loss-only baseline) | excess_loss | fixed |
| `su_only`, `sl_only`, `sd_only` | entropy / logit_margin / frequency | fixed |
| `mix4` (Mix-4) | excess_loss, entropy, logit_margin, frequency | dirichlet |
| `mix3` (Mix-3) | entropy, logit_margin, frequency | dirichlet |
| `vn_mix` (VN-Mix) | as Mix-4 | vn_dirichlet |
| `mix4_bigram`, `mix4_noise` | Mix-4 with frequency → bigram / noise | dirichlet |
| `clm` | — (alpha: null, no selection) | — |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
```

## Running

```bash
python scripts/prepare_data.py --dataset wikitext2 --out ./data/wikitext2 --bigram
python main.py --config configs/experiments/smoke.yaml
# or compose files and override keys:
python main.py --config configs/base.yaml --config configs/models/gpt2.yaml \
    --config configs/data/wikitext103.yaml --config configs/methods/mix4.yaml \
    --set seed=43 --set selection.alpha=0.5
python main.py --config ... --dry-run          # print the resolved config and its hash
```

Unknown keys and missing data files are errors. Each run writes a new
directory `runs/<name>_<date-time>_<id>/` and never overwrites anything:
`config.yaml` (resolved config + `config_hash`), `git.txt` (commit and
uncommitted diff), `env.txt`, `train.log`, `history.json`, `final.pt`
(with `output.save_final: true`) and, last, `DONE`. With `--skip-if-done`,
a configuration that already has a completed run is not run again.

## Method summary (paper §3)

- Scorers return one value per token; column t describes token t (the loss
  term predicted from position t−1). `excess_loss` runs the frozen reference
  model (default GPT-2 Small) on the same batch.
- Each scorer is z-normalised over the batch tokens; the utility is the
  weighted sum. The top ⌈α(T−1)⌉ positions of each sequence are kept, at
  every step, on the current batch.
- Mixture weights are sampled every `rescore_interval` steps. After each
  validation event (every `eval.interval` steps), the active weights receive
  r = max(0, L_prev − L_val) and α ← α + r·w (VN-Mix: r·w/(σ̂+ε)).
- The loss is the mean NLL over the selected tokens. AdamW (β₂ = 0.95,
  weight decay 0.1), linear warmup then cosine decay to 10 %, fp32 weights
  with bf16 autocast on CUDA.
