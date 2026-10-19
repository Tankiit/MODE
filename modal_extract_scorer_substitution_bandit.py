from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal


app = modal.App("sieve-scorer-sub-bandit-extract")
volume_results = modal.Volume.from_name("sieve-results", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")


def expected_runs() -> list[dict]:
    rows = []
    for variant in ("bigram_freq", "random_noise"):
        for seed in (42, 43, 44):
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "run_name": f"sieve_{variant}_wikitext103_gpt2l_a70_s{seed}_st25000",
                }
            )
    return rows


@app.function(image=image, timeout=10 * 60, volumes={"/results": volume_results})
def extract() -> dict:
    import torch

    rows = []
    for spec in expected_runs():
        ckpt_root = Path("/results") / spec["run_name"] / "checkpoints"
        final_state = ckpt_root / "ckpt-step025000" / "bandit_state.pt"
        final_meta = ckpt_root / "ckpt-step025000" / "metadata.json"
        record = {
            **spec,
            "history": [],
            "alpha": None,
            "final_expected_weights": None,
            "final_val_ppl": None,
        }
        if final_meta.exists():
            record["final_val_ppl"] = json.loads(final_meta.read_text()).get("val_ppl")
        if final_state.exists():
            state = torch.load(final_state, map_location="cpu", weights_only=False)
            history = state.get("history", [])
            alpha = state.get("alpha")
            final_expected = state.get("final_expected_weights")
            if hasattr(alpha, "tolist"):
                alpha = alpha.tolist()
            if hasattr(final_expected, "tolist"):
                final_expected = final_expected.tolist()
            record["history"] = history
            record["alpha"] = alpha
            record["final_expected_weights"] = final_expected
        rows.append(record)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "wikitext103",
        "alpha": 0.7,
        "scorers": ["S_E", "S_U", "S_L", "S_D"],
        "runs": rows,
    }


@app.local_entrypoint()
def main(out: str = "plots/scorer_substitution_bandit_histories.json"):
    data = extract.remote()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path.resolve()}")
    for row in data["runs"]:
        print(
            f"{row['variant']:12s} seed={row['seed']} "
            f"hist={len(row['history']):3d} ppl={row['final_val_ppl']} "
            f"final={row['final_expected_weights']}"
        )
