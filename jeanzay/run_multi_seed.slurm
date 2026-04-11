#!/bin/bash
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --array=42-46
#SBATCH --output=logs/multiple_seed/%x_s%a_%A.out
#SBATCH --error=logs/multiple_seed/%x_s%a_%A.err
#
# Run 5 seeds (42..46) of one baseline as a SLURM job array.
#
# Usage:
#   sbatch -A <account> -J sieve --export=ALL,METHOD=sieve jeanzay/run_multi_seed.slurm
#   sbatch -A <account> -J rho1  --export=ALL,METHOD=rho1  jeanzay/run_multi_seed.slurm
#   sbatch -A <account> -J clm   --export=ALL,METHOD=clm   jeanzay/run_multi_seed.slurm
#
set -euo pipefail

module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0
source "$SCRATCH/venvs/sieve/bin/activate"
export HF_HUB_OFFLINE=1

cd "$WORK/SIEVE"
mkdir -p logs/multiple_seed

python train_sieve.py \
    --dataset wikitext2 \
    --model gpt2 \
    --steps 20000 \
    --baseline "$METHOD" \
    --seed "$SLURM_ARRAY_TASK_ID"
