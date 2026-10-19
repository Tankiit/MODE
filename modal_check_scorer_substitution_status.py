from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal


app = modal.App("sieve-scorer-substitution-status")
volume_results = modal.Volume.from_name("sieve-results", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11")


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


@app.function(image=image, timeout=5 * 60, volumes={"/results": volume_results})
def check_status() -> dict:
    rows = []
    for spec in expected_runs():
        run_dir = Path("/results") / spec["run_name"]
        ckpt_root = run_dir / "checkpoints"
        ckpts = sorted(ckpt_root.glob("ckpt-step*")) if ckpt_root.exists() else []
        last_step = None
        if ckpts:
            try:
                last_step = max(int(p.name.split("step")[-1]) for p in ckpts)
            except ValueError:
                pass

        metrics_path = run_dir / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
            except Exception as exc:
                metrics = {"metrics_error": str(exc)}

        final_meta = ckpt_root / "ckpt-step025000" / "metadata.json"
        final_meta_data = {}
        if final_meta.exists():
            try:
                final_meta_data = json.loads(final_meta.read_text())
            except Exception as exc:
                final_meta_data = {"metadata_error": str(exc)}

        rows.append(
            {
                **spec,
                "exists": run_dir.exists(),
                "checkpoint_count": len(ckpts),
                "last_step": last_step,
                "has_metrics": metrics_path.exists(),
                "status": metrics.get("status"),
                "final_val_ppl": metrics.get("final_val_ppl", final_meta_data.get("val_ppl")),
                "best_val_ppl": metrics.get("best_val_ppl"),
                "has_final_metadata": final_meta.exists(),
                "returncode": metrics.get("returncode"),
            }
        )

    complete = [
        r for r in rows
        if r["has_metrics"] and r["status"] == "ok" and r["final_val_ppl"] is not None
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_runs": len(rows),
        "present_runs": sum(1 for r in rows if r["exists"]),
        "finished_ok_runs": len(complete),
        "rows": rows,
    }


@app.local_entrypoint()
def main(out: str = "scorer_substitution_status.json"):
    data = check_status.remote()
    Path(out).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: data[k] for k in ("generated_at_utc", "expected_runs", "present_runs", "finished_ok_runs")}, indent=2))
    for row in data["rows"]:
        done = row["status"] == "ok" and row["final_val_ppl"] is not None
        status = "done" if done else "pending"
        print(
            f"{status:7s} {row['variant']:12s} seed={row['seed']} "
            f"ckpts={row['checkpoint_count']:2d} last={str(row['last_step']):>5s} "
            f"ppl={row['final_val_ppl']} status={row['status']} {row['run_name']}"
        )
    print(f"wrote {Path(out).resolve()}")
