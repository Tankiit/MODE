"""
Modal: CUDA image with `liger-kernel` for faster GPT-2 backward in train_sieve.

Image build includes:
  pip install liger-kernel  (+ torch, transformers, …)

Example:
  modal run modal_train_sieve.py

Override training args:
  modal run modal_train_sieve.py --train-cli "--dataset wikitext2 --model gpt2 --steps 100 --baseline sieve"

W&B (optional): attach a Modal secret that provides `WANDB_API_KEY` (the wandb client reads
  that env var — not a key named `wand_secret`).
  1) Create a secret named e.g. `wandb_secret`:  modal secret create wandb_secret WANDB_API_KEY=...
  2) Before `modal run`:  export MODAL_WANDB_SECRET_NAME=wandb_secret
  To run without any secret (no W&B auth on the worker):  export MODAL_WANDB_SECRET_NAME=
  Then omit `--wandb ...` from `--train-cli` or wandb may fail to log in.

Results + checkpoints live on volume `sieve-results-v1` under `/results/<run_name>/`.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent

DEFAULT_CLI = (
    "--dataset wikitext2 --model gpt2 --steps 5000 "
    "--baseline sieve --seed 42 --alpha 0.70"
)

app = modal.App("ramengpt-train-sieve")

volume_results = modal.Volume.from_name("sieve-results", create_if_missing=True)
volume_data    = modal.Volume.from_name("sieve-data", create_if_missing=True)
# Optional: mount a Modal secret that defines WANDB_API_KEY, e.g. name `wandb_secret`:
#   modal secret create wandb_secret WANDB_API_KEY=...
#   export MODAL_WANDB_SECRET_NAME=wandb_secret
# Default (unset): no secret. To force no secret: export MODAL_WANDB_SECRET_NAME=
_wandb_raw = os.environ.get("MODAL_WANDB_SECRET_NAME")
if _wandb_raw is None:
    _wandb_secret_name = ""
elif _wandb_raw.strip() == "":
    _wandb_secret_name = ""
else:
    _wandb_secret_name = _wandb_raw.strip()

_function_secrets: list = []
if _wandb_secret_name:
    _function_secrets = [modal.Secret.from_name(_wandb_secret_name)]

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
        ignore=["**/.git/**", "**/__pycache__/**", "**/*.pyc"],
    )
)


def _parse_cli_flags(train_cli: str) -> dict[str, str]:
    """Split `train_cli` into `--key` / `--key=val` flags (first occurrence wins)."""
    out: dict[str, str] = {}
    parts = shlex.split(train_cli)
    i, n = 0, len(parts)
    while i < n:
        tok = parts[i]
        if not tok.startswith("--"):
            i += 1
            continue
        body = tok[2:]
        if "=" in body:
            k, v = body.split("=", 1)
            out.setdefault(k, v)
            i += 1
            continue
        k = body
        if i + 1 < n and not parts[i + 1].startswith("--"):
            out.setdefault(k, parts[i + 1])
            i += 2
        else:
            out.setdefault(k, "1")
            i += 1
    return out


def _derive_run_name(train_cli: str) -> str:
    """Match train_sieve default run_name: {baseline}_{dataset}_{model}_a{..}_s{seed}."""
    d = _parse_cli_flags(train_cli)
    baseline = d.get("baseline", "sieve")
    dataset = d.get("dataset", "wikitext2")
    model = d.get("model", "gpt2")
    seed = d.get("seed", "42")
    alpha = float(d.get("alpha", "0.7"))
    return f"{baseline}_{dataset}_{model}_a{int(alpha * 100)}_s{seed}"


def _parse_train_log(log_path: Path) -> dict:
    """Lightweight scrape of train.log for final metrics."""
    m: dict = {}
    if not log_path.is_file():
        return m
    for line in log_path.read_text(errors="replace").splitlines():
        if "[final] val_ppl=" in line:
            try:
                rest = line.split("val_ppl=", 1)[1].strip()
                m["final_val_ppl"] = float(rest.split()[0].rstrip(","))
            except (ValueError, IndexError):
                pass
        if "[ckpt] final at step" in line:
            m["checkpoint_summary"] = line.strip()
    return m


@app.function(
    image=image,
    gpu="A10G",
    timeout=3 * 3600,
    volumes={"/results": volume_results, "/data": volume_data},
    secrets=_function_secrets,
    retries=modal.Retries(max_retries=1, initial_delay=30.0),
)
def train_remote(train_cli: str = DEFAULT_CLI) -> dict:
    import os

    run_name = _derive_run_name(train_cli)
    run_dir = Path(f"/results/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    save_flags: list[str] = []
    if "--save_dir" not in train_cli:
        save_flags.append(f"--save_dir={run_dir / 'checkpoints'}")
    if "--save_interval" not in train_cli:
        save_flags.append("--save_interval=1000")
    if "--max_checkpoints" not in train_cli:
        save_flags.append("--max_checkpoints=5")
    
    # ── INSERT THIS BLOCK ─────────────────────────────────────────────
    # Mount volume datasets at the path train_sieve.py expects.
    # train_sieve.py opens 'data/<ds>/train.bin' (relative to /repo).
    # The sieve-data volume is mounted at /data, so symlink
    # /repo/data/<ds> → /data/<ds>.
    import shutil
    Path("/repo/data").mkdir(exist_ok=True)
    for ds in ("wikitext2", "wikitext103", "openwebmath_1b"):
        src = Path(f"/data/{ds}")
        dst = Path(f"/repo/data/{ds}")
        if not src.exists():
            continue
        if dst.is_symlink():
            if dst.resolve() == src.resolve():
                continue
            dst.unlink()
        elif dst.exists():
            shutil.rmtree(dst)
        dst.symlink_to(src)
    print(f"[modal] data symlinks: "
          f"{sorted(p.name for p in Path('/repo/data').iterdir())}")
    # ── END INSERT ────────────────────────────────────────────────────

    env = os.environ.copy()
    env.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "42",
            "WANDB_DIR": str(run_dir),
            "WANDB_NAME": run_name,
        }
    )

    cmd = [sys.executable, "train_sieve.py", *shlex.split(train_cli), *save_flags]
    print(f"[modal] cmd: {' '.join(cmd)}")

    log_path = run_dir / "train.log"
    t0 = time.time()
    with open(log_path, "w") as log:
        result = subprocess.run(
            cmd,
            cwd="/repo",
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall = time.time() - t0

    metrics: dict = {
        "run_name": run_name,
        "cli": train_cli,
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "wall_seconds": wall,
        "checkpoint_dir": str(run_dir / "checkpoints"),
    }
    metrics.update(_parse_train_log(log_path))

    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(p.name for p in ckpt_dir.glob("ckpt-step*"))
        metrics["checkpoints"] = ckpts
        best = ckpt_dir / "ckpt-best"
        if best.is_symlink():
            metrics["best_checkpoint"] = Path(best.readlink()).name

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    volume_results.commit()

    print(
        f"[modal] {metrics['status']}  returncode={result.returncode}  "
        f"wall={wall:.1f}s  run_dir={run_dir}"
    )
    if metrics.get("final_val_ppl") is not None:
        print(f"[modal] final_val_ppl={metrics['final_val_ppl']:.2f}")
    if result.returncode != 0:
        print(f"[modal] see log: {log_path}")

    return metrics


@app.local_entrypoint()
def main(train_cli: str = "") -> None:
    cli = train_cli.strip() or DEFAULT_CLI
    metrics = train_remote.remote(cli)
    print(json.dumps(metrics, indent=2))
    if metrics.get("status") != "ok":
        raise SystemExit(
            f"train_sieve exited with returncode={metrics.get('returncode')}"
        )
