#!/bin/bash

set -euo pipefail

rsync -avP \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='datasets/' \
  --exclude='hf_trainer_out/' \
  --exclude='results/' \
  --exclude='data/' \
  --exclude='assets/' \
  --exclude='wandb/' \
  --exclude='logs/' \
  --exclude='jeanzay/logs/' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='*.bin' \
  --exclude='uv.lock' \
  --exclude='.venv' \
  --exclude='SIEVE_Training_Colab.ipynb' \
  --exclude='docs/' \
  --exclude='.claude' \
  --exclude='.vscode' \
  --exclude='experiments.csv' \
  /media/naim/encrypted_drive/project/MODE/ jeanzay:/lustre/fswork/projects/rech/sum/ufj45ra/SIEVE/
