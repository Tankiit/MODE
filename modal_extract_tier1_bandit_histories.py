from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal


app = modal.App("sieve-tier1-bandit-history-extract")
volume_results = modal.Volume.from_name("sieve-results", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch")


def expected_runs(dataset: str, alpha_tag: int, seeds: list[int]) -> list[dict]:
    rows = []
    for method, prefix in [
        ("ts_sieve", f"sieve_{dataset}_gpt2l_a{alpha_tag}"),
        ("vnts_sieve", f"sieve_VNTS_{dataset}_gpt2l_a{alpha_tag}"),
    ]:
        for seed in seeds:
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "alpha": alpha_tag / 100.0,
                    "seed": seed,
                    "run_name": f"{prefix}_s{seed}_st25000",
                }
            )
    return rows


def active_bin(step: int, max_steps: int = 25000) -> int:
    p = step / max(max_steps, 1)
    c0 = int(p < 0.10)
    c1 = int(p < 0.30)
    c2 = int(0.30 <= p < 0.70)
    c3 = int(p >= 0.70)
    return (c0 << 3) | (c1 << 2) | c3 | (c2 << 1)


def expected_weights(alpha, bin_idx: int) -> list[float] | None:
    if alpha is None:
        return None
    row = alpha[bin_idx]
    denom = float(sum(row))
    if denom <= 0:
        return None
    return [float(x) / denom for x in row]


@app.function(image=image, timeout=10 * 60, volumes={"/results": volume_results})
def extract_histories(dataset: str, alpha_tag: int, seeds: list[int]) -> dict:
    import torch

    out = []
    for row in expected_runs(dataset, alpha_tag, seeds):
        ckpt_root = Path("/results") / row["run_name"] / "checkpoints"
        final_ckpt = ckpt_root / "ckpt-step025000"
        state_path = final_ckpt / "bandit_state.pt"
        meta_path = final_ckpt / "metadata.json"
        ckpts = sorted(ckpt_root.glob("ckpt-step*")) if ckpt_root.exists() else []
        record = {
            **row,
            "exists": ckpt_root.exists(),
            "checkpoint_count": len(ckpts),
            "has_final_state": state_path.exists(),
            "has_final_metadata": meta_path.exists(),
            "history": [],
            "state_keys": [],
            "alpha_matrix": None,
            "checkpoint_weights": [],
            "final_val_ppl": None,
        }
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                record["final_val_ppl"] = meta.get("val_ppl")
            except Exception as exc:
                record["metadata_error"] = str(exc)
        if state_path.exists():
            try:
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                record["state_keys"] = sorted(list(state.keys())) if isinstance(state, dict) else []
                history = state.get("history", [])
                alpha = state.get("alpha")
                if hasattr(alpha, "tolist"):
                    alpha = alpha.tolist()
                record["history"] = history
                record["alpha_matrix"] = alpha
            except Exception as exc:
                record["state_error"] = str(exc)

        for ckpt in ckpts:
            try:
                step = int(ckpt.name.split("step")[-1])
            except ValueError:
                continue
            state_file = ckpt / "bandit_state.pt"
            if not state_file.exists():
                continue
            try:
                state = torch.load(state_file, map_location="cpu", weights_only=False)
                alpha = state.get("alpha")
                if hasattr(alpha, "tolist"):
                    alpha = alpha.tolist()
                b = active_bin(step)
                weights = expected_weights(alpha, b)
                if weights is not None:
                    record["checkpoint_weights"].append(
                        {"step": step, "bin": b, "weights": weights}
                    )
            except Exception as exc:
                record.setdefault("checkpoint_errors", []).append({"step": step, "error": str(exc)})
        out.append(record)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "alpha": alpha_tag / 100.0,
        "seeds": seeds,
        "runs": out,
    }


@app.local_entrypoint()
def main(
    dataset: str = "wikitext103",
    alpha: float = 0.7,
    seeds: str = "42,43,44",
    out: str = "plots/tier1_bandit_histories_wikitext103_a70.json",
):
    seed_values = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    alpha_tag = int(round(alpha * 100))
    data = extract_histories.remote(dataset, alpha_tag, seed_values)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path.resolve()}")
    for row in data["runs"]:
        print(
            f"{row['method']:10s} seed={row['seed']} "
            f"ckpts={row['checkpoint_count']:2d} final_state={row['has_final_state']} "
            f"hist={len(row['history']):3d} points={len(row['checkpoint_weights']):2d} "
            f"ppl={row['final_val_ppl']} {row['run_name']}"
        )
