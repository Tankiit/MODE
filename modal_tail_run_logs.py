from __future__ import annotations

from pathlib import Path

import modal


app = modal.App("sieve-tail-run-logs")
volume_results = modal.Volume.from_name("sieve-results", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, timeout=2 * 60, volumes={"/results": volume_results})
def tail_logs(run_names: list[str], n_chars: int = 8000) -> dict[str, str]:
    out = {}
    for name in run_names:
        p = Path("/results") / name / "train.log"
        if not p.exists():
            out[name] = "<missing train.log>"
            continue
        text = p.read_text(errors="replace")
        out[name] = text[-n_chars:]
    return out


@app.local_entrypoint()
def main(runs: str, n_chars: int = 8000):
    run_names = [r.strip() for r in runs.split(",") if r.strip()]
    logs = tail_logs.remote(run_names, n_chars)
    for name, text in logs.items():
        print(f"\n==== {name} ====\n{text}")
