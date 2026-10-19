from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal


app = modal.App("sieve-owm-sweep-status")
volume_results = modal.Volume.from_name("sieve-results", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11")


def expected_runs() -> list[str]:
    names: list[str] = []
    for seed in (42, 43, 44):
        for alpha in (30, 50, 70, 90):
            names.append(f"sieve_VNTS_owm_gpt2l_a{alpha}_s{seed}_st25000")
    for seed in (42, 43, 44):
        for alpha in (50, 70):
            names.append(f"sieve_owm_gpt2l_a{alpha}_s{seed}_st25000")
            names.append(f"rho1_owm_gpt2l_a{alpha}_s{seed}_st25000")
    return names


@app.function(image=image, timeout=5 * 60, volumes={"/results": volume_results})
def check_status() -> dict:
    rows = []
    for name in expected_runs():
        run_dir = Path("/results") / name
        ckpt_root = run_dir / "checkpoints"
        ckpts = sorted(ckpt_root.glob("ckpt-step*")) if ckpt_root.exists() else []
        final_meta = ckpt_root / "ckpt-step025000" / "metadata.json"
        last_step = None
        final_val_ppl = None
        if ckpts:
            try:
                last_step = max(int(p.name.split("step")[-1]) for p in ckpts)
            except ValueError:
                last_step = None
        if final_meta.exists():
            try:
                meta = json.loads(final_meta.read_text())
                final_val_ppl = meta.get("val_ppl")
            except Exception:
                final_val_ppl = None
        rows.append(
            {
                "run_name": name,
                "exists": run_dir.exists(),
                "checkpoint_count": len(ckpts),
                "last_step": last_step,
                "has_final": final_meta.exists(),
                "final_val_ppl": final_val_ppl,
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_runs": len(rows),
        "complete_runs": sum(1 for r in rows if r["checkpoint_count"] >= 25 and r["has_final"]),
        "present_runs": sum(1 for r in rows if r["exists"]),
        "rows": rows,
    }


@app.local_entrypoint()
def main(out: str = "owm_sweep_status.json"):
    data = check_status.remote()
    Path(out).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: data[k] for k in ("generated_at_utc", "expected_runs", "present_runs", "complete_runs")}, indent=2))
    for row in data["rows"]:
        status = "done" if row["checkpoint_count"] >= 25 and row["has_final"] else "pending"
        print(
            f"{status:7s} {row['checkpoint_count']:2d} ckpts "
            f"last={str(row['last_step']):>5s} final={row['final_val_ppl']}  {row['run_name']}"
        )
    print(f"wrote {Path(out).resolve()}")
