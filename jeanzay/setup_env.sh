#!/bin/bash
# Setup environment on Jean Zay (run once, interactively)
#
#   bash jeanzay/setup_env.sh
#
set -euo pipefail

module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

VENV_DIR="$SCRATCH/venvs/sieve"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR ..."
    python -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --no-cache-dir transformers datasets tqdm wandb huggingface-hub

# Prepare wikitext2 data
echo "Preparing wikitext2 dataset..."
cd "$WORK/SIEVE"
python prepare_sieve.py --dataset wikitext2 --data_dir ./data/wikitext2

# Pre-download HF models (compute nodes have no internet)
# Use huggingface-cli to avoid importing torch (fails on login node)
echo "Downloading GPT-2 model and tokenizer to cache..."
python -m huggingface_hub.commands.huggingface_cli download gpt2 

echo ""
echo "Done. Venv: $VENV_DIR"
echo "Data:  $WORK/SIEVE/data/wikitext2/"

