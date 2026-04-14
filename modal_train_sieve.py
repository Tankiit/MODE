"""
Modal: CUDA image with `liger-kernel` for faster GPT-2 backward in train_sieve.

Image build includes:
  pip install liger-kernel  (+ torch, transformers, …)

Example:
  modal run modal_train_sieve.py

Override training args:
  modal run modal_train_sieve.py --train-cli "--dataset wikitext2 --model gpt2 --epochs 3 --baseline sieve"

Bake your prepared data into the image or attach a `modal.Volume` at `./data` as needed.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent

app = modal.App("ramengpt-train-sieve")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "liger-kernel",
        "torch",
        "transformers>=4.36",
        "accelerate",
        "datasets",
        "tiktoken",
        "numpy",
        "wandb",
        "huggingface-hub",
        "einops",
        "kernels",
    )
    .add_local_dir(
        REPO,
        remote_path="/repo",
        ignore=["**/.git/**", "**/__pycache__/**", "**/*.pyc", "**/data/**"],
    )
)


@app.function(image=image, gpu="T4", timeout=86400)
def train_remote(train_cli: str = "--dataset wikitext2 --model gpt2 --steps 10 --baseline clm") -> None:
    cmd = [sys.executable, "train_sieve.py", *shlex.split(train_cli)]
    subprocess.check_call(cmd, cwd="/repo")


@app.local_entrypoint()
def main(train_cli: str = "") -> None:
    cli = train_cli.strip() or "--dataset wikitext2 --model gpt2 --steps 10 --baseline clm"
    train_remote.remote(cli)
