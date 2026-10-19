from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from extract_checkpoint_figures import (
    PLOTS,
    SCORERS,
    Run,
    aggregate_final,
    grouped_trajectory,
    load_variance_analysis,
    write_bin_heatmap_svg,
    write_final_ppl_svg,
    write_trajectory_svg,
    write_variance_svg,
    write_weight_history_svg,
)


ROOT = Path(__file__).resolve().parents[1]


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    raw_path = PLOTS / "modal_checkpoint_extract_raw.json"
    raw = json.loads(raw_path.read_text())
    runs = [
        Run(
            run_name=r["run_name"],
            method=r["method"],
            alpha=float(r["alpha"]),
            seed=int(r["seed"]),
            checkpoint_count=int(r["checkpoint_count"]),
        )
        for r in raw["runs_requested"]
    ]
    trajectories = raw["trajectories"]
    states = raw["bandit_states"]
    finals = raw["runs"]
    agg = aggregate_final(finals)
    grouped = grouped_trajectory(runs, trajectories)
    variance = load_variance_analysis()

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(raw_path),
        "dataset": raw["dataset"],
        "model": raw["model"],
        "scorers": SCORERS,
        "runs": finals,
        "aggregate_final": agg,
        "trajectory_groups": grouped,
        "bandit_states": states,
        "variance_analysis": variance,
        "notes": [
            "Checkpoint trajectories are from metadata.json at every 1k saved checkpoint.",
            "Scorer-function plots use final checkpoint bandit_state.pt histories.",
            "Rho-1 has fixed scorer weights and no bandit posterior.",
        ],
    }

    PLOTS.mkdir(exist_ok=True)
    (PLOTS / "sieve_checkpoint_figure_data.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    write_csv(finals, PLOTS / "sieve_checkpoint_final_runs.csv")
    write_csv(agg, PLOTS / "sieve_checkpoint_final_aggregate.csv")
    write_trajectory_svg(grouped, PLOTS / "fig_val_ppl_trajectories.svg")
    write_final_ppl_svg(agg, PLOTS / "fig_final_ppl_by_alpha.svg")
    write_weight_history_svg(states, runs, PLOTS / "fig_scorer_weight_history.svg")
    write_bin_heatmap_svg(states, runs, PLOTS / "fig_vnts_bin_weights_heatmap.svg")
    if variance:
        write_variance_svg(variance, PLOTS / "fig_scorer_variance_diagnostics.svg")
    print(f"[done] wrote figures/data under {PLOTS.resolve()}")


if __name__ == "__main__":
    main()
