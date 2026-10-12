"""
One directory per run, never overwritten.

runs/<name>_<YYYYmmdd-HHMMSS>_<6 hex>/
  config.yaml      resolved configuration (+ config_hash)
  git.txt          commit, branch and uncommitted diff of the code
  env.txt          Python / torch / transformers versions, device
  train.log        log of the run
  history.json     training record (written at the end, atomically)
  final.pt         model state_dict (output.save_final)
  DONE             written last: final metrics; marks the run as complete
"""
from __future__ import annotations

import json
import logging
import os
import platform
import secrets
import subprocess
import sys
import time
from pathlib import Path

import yaml

from ..config import RunConfig, config_hash, to_dict

REPO_ROOT = Path(__file__).resolve().parents[2]


def create_run_dir(root: str, name: str) -> Path:
    path = Path(root) / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}_{secrets.token_hex(3)}"
    path.mkdir(parents=True, exist_ok=False)          # never reuse a directory
    return path


def find_completed(root: str, cfg_hash: str) -> list[Path]:
    """Run directories with the same config hash and a DONE marker."""
    out = []
    for cfg_file in Path(root).glob("*/config.yaml"):
        d = cfg_file.parent
        if (d / "DONE").exists():
            meta = yaml.safe_load(cfg_file.read_text()) or {}
            if meta.get("config_hash") == cfg_hash:
                out.append(d)
    return sorted(out)


def _git(*args: str) -> tuple[bool, str]:
    """(ok, output); ok is False if git is missing or the command fails."""
    try:
        r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    except Exception as e:                              # git missing, timeout...
        return False, str(e)
    return r.returncode == 0, (r.stdout if r.returncode == 0 else r.stderr).strip()


def write_metadata(run_dir: Path, cfg: RunConfig) -> None:
    d = to_dict(cfg)
    d["config_hash"] = config_hash(cfg)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    ok_c, commit = _git("rev-parse", "HEAD")
    ok_b, branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    ok_d, diff = _git("diff", "HEAD")
    if ok_c and ok_b and ok_d:
        text = (f"commit: {commit}\nbranch: {branch}\n"
                f"uncommitted changes: {'yes' if diff else 'no'}\n\n{diff}\n")
    else:
        text = f"git information unavailable: {commit if not ok_c else branch if not ok_b else diff}\n"
    (run_dir / "git.txt").write_text(text)
    import torch
    import transformers
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    (run_dir / "env.txt").write_text(
        f"python: {sys.version.split()[0]}\nplatform: {platform.platform()}\n"
        f"torch: {torch.__version__}\ntransformers: {transformers.__version__}\ndevice: {dev}\n"
        f"command: {' '.join(sys.argv)}\n")


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def mark_done(run_dir: Path, summary: dict) -> None:
    write_json_atomic(run_dir / "DONE", summary)


def setup_logging(run_dir: Path) -> logging.Logger:
    log = logging.getLogger("sieve")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.propagate = False
    return log
