#!/bin/bash
#SBATCH --job-name=rho1_wikitext2
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=logs/rho1_%j.out
#SBATCH --error=logs/rho1_%j.err

set -euo pipefail

module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0
source "$SCRATCH/venvs/sieve/bin/activate"
export HF_HUB_OFFLINE=1

cd "$WORK/SIEVE"
mkdir -p logs

python train_sieve.py \
    --dataset wikitext2 \
    --model gpt2 \
    --steps 20000 \
    --baseline rho1 \
    --seed 42
