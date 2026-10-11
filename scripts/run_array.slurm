#!/bin/bash
#SBATCH --job-name=sieve
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Run config number $SLURM_ARRAY_TASK_ID of a list written by
# scripts/expand_sweep.py. main.py --skip-if-done exits immediately when a
# completed run with the same configuration already exists, so the same
# array can be resubmitted after failures. Each run writes a new directory
# under runs/ (never overwritten).
#
# Usage (from the repo root, after `mkdir -p logs`):
#   sbatch -A <account> --time=<hh:mm:ss> \
#     --array=1-$(wc -l < sweeps/paper/list_gpt2.txt)%20 \
#     --export=ALL,LIST=sweeps/paper/list_gpt2.txt scripts/run_array.slurm
set -euo pipefail

# ── Cluster-specific environment (adapt) ─────────────────────────────
# module purge; module load <cuda/python modules>
source .venv/bin/activate
export HF_HUB_OFFLINE=1          # compute nodes are often offline
export HF_DATASETS_OFFLINE=1

CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
echo "[task ${SLURM_ARRAY_TASK_ID}] $CONFIG"
python main.py --config "$CONFIG" --skip-if-done
