# modal run modal_prepare_c4.py [--tokens N] [--extra "..."]
# modal volume get sieve-c4-data /c4_out ./data
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent

app = modal.App("ramengpt-prepare-c4")

data_volume = modal.Volume.from_name("sieve-c4-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "torch",
        "transformers>=4.36",
        "datasets>=2.19",
        "accelerate",
    )
    .copy_local_dir(str(REPO), "/workspace")
)


@app.function(
    image=image,
    timeout=24 * 3600,
    memory=65536,
    cpu=8.0,
    volumes={"/c4_out": data_volume},
)
def prepare_c4_remote(cli: str = "") -> None:
    base = [
        sys.executable,
        "/workspace/scripts/prepare_c4.py",
        "--output",
        "/c4_out/c4_1b",
    ]
    if cli.strip():
        base.extend(shlex.split(cli))
    subprocess.check_call(base, cwd="/workspace")
    data_volume.commit()


@app.local_entrypoint()
def main(tokens: int = 0, extra: str = "") -> None:
    parts: list[str] = []
    if tokens > 0:
        parts.append(f"--tokens {int(tokens)}")
    if extra.strip():
        parts.append(extra.strip())
    cli = " ".join(parts)
    prepare_c4_remote.remote(cli)
