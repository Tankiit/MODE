from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


app = modal.App("sieve-checkpoint-figure-extract")
volume_results = modal.Volume.from_name("sieve-results", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")


def _load_runs_local() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []

    multi = json.loads(Path("multi_seed_wikitext103_a50_a70_results.json").read_text())
    for r in multi["runs"]:
        if r.get("status") != "ok":
            continue
        runs.append(
            {
                "run_name": r["run_name"],
                "method": r["baseline"],
                "alpha": float(r["alpha"]),
                "seed": int(r["seed"]),
                "checkpoint_count": int(r["checkpoint_count"]),
            }
        )

    vnts = json.loads(Path("vnts_wikitext103_checkpoint_summary.json").read_text())
    for r in vnts["runs"]:
        if r.get("status") != "ok":
            continue
        runs.append(
            {
                "run_name": r["run_name"],
                "method": "vnts",
                "alpha": float(r["alpha"]),
                "seed": int(r["seed"]),
                "checkpoint_count": int(r["checkpoint_count"]),
            }
        )

    return runs


def _expected_weights(alpha):
    out = []
    for row in alpha:
        denom = float(sum(row))
        out.append([float(x) / denom if denom else 0.0 for x in row])
    return out


@app.function(
    image=image,
    timeout=20 * 60,
    volumes={"/results": volume_results},
)
def extract_checkpoint_info(runs: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    trajectories: dict[str, list[dict[str, Any]]] = {}
    bandit_states: dict[str, dict[str, Any]] = {}
    final_runs: list[dict[str, Any]] = []

    for run in runs:
        run_name = run["run_name"]
        ckpt_root = Path("/results") / run_name / "checkpoints"
        points = []
        for i in range(1, int(run["checkpoint_count"]) + 1):
            step = i * 1000
            meta_path = ckpt_root / f"ckpt-step{step:06d}" / "metadata.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            points.append(
                {
                    "step": int(meta.get("step", step)),
                    "val_ppl": float(meta["val_ppl"]),
                }
            )
        trajectories[run_name] = points

        if points:
            final = points[-1]
            best = min(points, key=lambda p: p["val_ppl"])
            final_runs.append(
                {
                    **run,
                    "final_val_ppl": final["val_ppl"],
                    "best_val_ppl": best["val_ppl"],
                    "best_step": best["step"],
                    "checkpoint_count_found": len(points),
                }
            )

        state_path = ckpt_root / "ckpt-step025000" / "bandit_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            alpha = state.get("alpha")
            if hasattr(alpha, "tolist"):
                alpha = alpha.tolist()
            history = state.get("history", [])
            bandit_states[run_name] = {
                "alpha": alpha,
                "history": history,
                "expected_weights_by_bin": _expected_weights(alpha),
            }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "wikitext103",
        "model": "gpt2l",
        "runs_requested": runs,
        "runs": final_runs,
        "trajectories": trajectories,
        "bandit_states": bandit_states,
    }


@app.local_entrypoint()
def main(out: str = "plots/modal_checkpoint_extract_raw.json"):
    runs = _load_runs_local()
    print(f"[extract] runs={len(runs)}")
    data = extract_checkpoint_info.remote(runs)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"[extract] wrote {out_path.resolve()}")
