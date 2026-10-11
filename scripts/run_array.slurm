#!/bin/bash
#SBATCH --job-name=sieve
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Run line $SLURM_ARRAY_TASK_ID of a command file produced by
# scripts/make_commands.sh. Runs whose history.json already exists are
# skipped, so the same array can be resubmitted after failures.
#
# Usage (from the repo root, after `mkdir -p logs runs`):
#   sbatch -A <account> --time=<hh:mm:ss> \
#     --array=1-$(wc -l < commands_small.txt)%20 \
#     --export=ALL,CMDS=commands_small.txt scripts/run_array.slurm
set -euo pipefail

# ── Cluster-specific environment (adapt) ─────────────────────────────
# module purge; module load <cuda/python modules>
source .venv/bin/activate
export HF_HUB_OFFLINE=1          # compute nodes are often offline
export HF_DATASETS_OFFLINE=1

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMDS")
OUTDIR=$(echo "$CMD" | sed -E 's/.*--out_dir ([^ ]+).*/\1/')
if [[ -f "$OUTDIR/history.json" ]]; then
  echo "[task ${SLURM_ARRAY_TASK_ID}] already done: $OUTDIR"
  exit 0
fi
mkdir -p "$OUTDIR"
echo "[task ${SLURM_ARRAY_TASK_ID}] $CMD"
eval "$CMD" 2>&1 | tee "$OUTDIR/train.log"
