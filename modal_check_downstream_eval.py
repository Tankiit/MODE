from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("sieve-downstream-eval-status")
volume_results = modal.Volume.from_name("sieve-results", create_if_missing=True)

RUNS = [
    "sieve_no_se_wikitext103_gpt2l_a70_s42_st25000",
    "su_only_wikitext103_gpt2l_a70_s42_st25000",
    "sl_only_wikitext103_gpt2l_a70_s42_st25000",
    "sd_only_wikitext103_gpt2l_a70_s42_st25000",
]


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/results": volume_results},
    timeout=5 * 60,
)
def check_status(step: int = 25000) -> dict:
    volume_results.reload()
    root = Path("/results/downstream_eval")
    out = {"step": step, "runs": []}
    for run_name in RUNS:
        run_dir = root / run_name / f"step{step:06d}"
        summary_path = run_dir / "summary.json"
        row: dict = {
            "run_name": run_name,
            "exists": run_dir.exists(),
            "summary_exists": summary_path.exists(),
        }
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            row["status"] = summary.get("status")
            if summary.get("status_note"):
                row["status_note"] = summary.get("status_note")
            row["tasks"] = {}
            for task, task_summary in summary.get("tasks", {}).items():
                task_row = {
                    "returncode": task_summary.get("returncode"),
                    "wall_seconds": task_summary.get("wall_seconds"),
                }
                for key, value in task_summary.items():
                    if "/" in key and isinstance(value, (int, float)):
                        task_row[key] = value
                if "error_tail" in task_summary:
                    task_row["error_tail"] = task_summary["error_tail"][-600:]
                row["tasks"][task] = task_row
        else:
            row["task_dirs"] = [
                p.name for p in sorted(run_dir.iterdir())
            ] if run_dir.exists() else []
        out["runs"].append(row)
    return out


@app.local_entrypoint()
def main(step: int = 25000) -> None:
    status = check_status.remote(step=step)
    Path("downstream_eval_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    print("wrote downstream_eval_status.json")
