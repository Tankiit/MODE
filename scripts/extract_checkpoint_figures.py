from __future__ import annotations

import csv
import json
import math
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "sieve_checkpoint_figures"
PLOTS = ROOT / "plots"
VOLUME = "sieve-results"
SCORERS = ["S_E", "S_U", "S_L", "S_D"]
COLORS = {
    "rho1": "#4c78a8",
    "sieve": "#f58518",
    "vnts": "#54a24b",
    "S_E": "#4c78a8",
    "S_U": "#f58518",
    "S_L": "#54a24b",
    "S_D": "#b279a2",
}


@dataclass(frozen=True)
class Run:
    run_name: str
    method: str
    alpha: float
    seed: int
    checkpoint_count: int = 25


def step_name(step: int) -> str:
    return f"ckpt-step{step:06d}"


def modal_get(remote: str, local: Path, retries: int = 2) -> bool:
    if local.exists() and local.stat().st_size > 0:
        return True
    local.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["modal", "volume", "get", VOLUME, remote, str(local)]
    for attempt in range(retries + 1):
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if proc.returncode == 0 and local.exists() and local.stat().st_size > 0:
            return True
        if "No such file" in (proc.stderr + proc.stdout):
            return False
        if attempt == retries:
            print(f"[warn] failed modal get: {remote}\n{proc.stderr or proc.stdout}", file=sys.stderr)
    return False


def load_multiseed_runs() -> list[Run]:
    path = ROOT / "multi_seed_wikitext103_a50_a70_results.json"
    data = json.loads(path.read_text())
    runs: list[Run] = []
    for r in data["runs"]:
        if r.get("status") != "ok":
            continue
        method = r["baseline"]
        runs.append(
            Run(
                run_name=r["run_name"],
                method=method,
                alpha=float(r["alpha"]),
                seed=int(r["seed"]),
                checkpoint_count=int(r["checkpoint_count"]),
            )
        )
    return runs


def load_vnts_runs() -> list[Run]:
    path = ROOT / "vnts_wikitext103_checkpoint_summary.json"
    data = json.loads(path.read_text())
    runs: list[Run] = []
    for r in data["runs"]:
        if r.get("status") != "ok":
            continue
        runs.append(
            Run(
                run_name=r["run_name"],
                method="vnts",
                alpha=float(r["alpha"]),
                seed=int(r["seed"]),
                checkpoint_count=int(r["checkpoint_count"]),
            )
        )
    return runs


def fetch_metadata(runs: list[Run]) -> dict[str, list[dict[str, Any]]]:
    tasks: list[tuple[Run, int, str, Path]] = []
    for run in runs:
        for i in range(1, run.checkpoint_count + 1):
            step = i * 1000
            remote = f"{run.run_name}/checkpoints/{step_name(step)}/metadata.json"
            local = CACHE / "metadata" / run.run_name / f"{step_name(step)}.json"
            tasks.append((run, step, remote, local))

    print(f"[fetch] metadata files: {len(tasks)}")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(modal_get, remote, local): (run, step, local) for run, step, remote, local in tasks}
        for fut in as_completed(futs):
            fut.result()

    out: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        points = []
        for i in range(1, run.checkpoint_count + 1):
            step = i * 1000
            local = CACHE / "metadata" / run.run_name / f"{step_name(step)}.json"
            if not local.exists():
                continue
            meta = json.loads(local.read_text())
            points.append({"step": int(meta.get("step", step)), "val_ppl": float(meta["val_ppl"])})
        out[run.run_name] = points
    return out


def fetch_bandit_states(runs: list[Run]) -> dict[str, dict[str, Any]]:
    bandit_runs = [r for r in runs if r.method in {"sieve", "vnts"}]
    tasks = []
    for run in bandit_runs:
        remote = f"{run.run_name}/checkpoints/ckpt-step025000/bandit_state.pt"
        local = CACHE / "bandit_state" / run.run_name / "bandit_state.pt"
        tasks.append((run, remote, local))
    print(f"[fetch] bandit_state files: {len(tasks)}")
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(modal_get, remote, local): (run, local) for run, remote, local in tasks}
        for fut in as_completed(futs):
            fut.result()

    import torch

    out: dict[str, dict[str, Any]] = {}
    for run, _, local in tasks:
        if not local.exists():
            continue
        obj = torch.load(local, map_location="cpu", weights_only=False)
        alpha = obj.get("alpha")
        hist = obj.get("history", [])
        if hasattr(alpha, "tolist"):
            alpha = alpha.tolist()
        out[run.run_name] = {
            "alpha": alpha,
            "history": hist,
            "expected_weights_by_bin": expected_weights(alpha),
        }
    return out


def expected_weights(alpha: list[list[float]]) -> list[list[float]]:
    rows = []
    for row in alpha:
        s = sum(row)
        rows.append([float(x) / s if s else 0.0 for x in row])
    return rows


def load_variance_analysis() -> dict[str, Any]:
    path = ROOT / "variance_analysis_wikitext103_gpt2l_a70_s42.json"
    return json.loads(path.read_text()) if path.exists() else {}


def grouped_trajectory(
    runs: list[Run], trajectories: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, float], list[list[dict[str, Any]]]] = {}
    for run in runs:
        pts = trajectories.get(run.run_name, [])
        if len(pts) < 2:
            continue
        grouped.setdefault((run.method, run.alpha), []).append(pts)

    out: dict[str, dict[str, Any]] = {}
    for (method, alpha), series in sorted(grouped.items()):
        steps = sorted({p["step"] for s in series for p in s})
        rows = []
        for step in steps:
            vals = [p["val_ppl"] for s in series for p in s if p["step"] == step]
            if vals:
                rows.append(
                    {
                        "step": step,
                        "mean": sum(vals) / len(vals),
                        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                        "n": len(vals),
                    }
                )
        out[f"{method}_a{int(alpha * 100)}"] = {
            "method": method,
            "alpha": alpha,
            "points": rows,
        }
    return out


def final_summary(runs: list[Run], trajectories: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        pts = trajectories.get(run.run_name, [])
        if not pts:
            continue
        final = pts[-1]["val_ppl"]
        best = min(pts, key=lambda p: p["val_ppl"])
        rows.append(
            {
                "run_name": run.run_name,
                "method": run.method,
                "alpha": run.alpha,
                "seed": run.seed,
                "final_val_ppl": final,
                "best_val_ppl": best["val_ppl"],
                "best_step": best["step"],
                "checkpoint_count": len(pts),
            }
        )
    return rows


def aggregate_final(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["alpha"]), []).append(row)
    out = []
    for (method, alpha), rs in sorted(grouped.items()):
        vals = [r["final_val_ppl"] for r in rs]
        out.append(
            {
                "method": method,
                "alpha": alpha,
                "n": len(vals),
                "mean_final_val_ppl": sum(vals) / len(vals),
                "std_final_val_ppl": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min_final_val_ppl": min(vals),
                "max_final_val_ppl": max(vals),
                "seeds": sorted(r["seed"] for r in rs),
            }
        )
    return out


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#202020}",
        ".title{font-size:22px;font-weight:700}.subtitle{font-size:13px;fill:#666}",
        ".axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.label{font-size:12px;fill:#555}",
        ".legend{font-size:12px}.tick{font-size:11px;fill:#555}",
        "</style>",
    ]


def scale(v: float, lo: float, hi: float, a: float, b: float) -> float:
    if hi == lo:
        return (a + b) / 2
    return a + (v - lo) * (b - a) / (hi - lo)


def polyline(points: list[tuple[float, float]], color: str, width: float = 2.2) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{pts}"/>'


def draw_axes(
    parts: list[str],
    x0: int,
    y0: int,
    w: int,
    h: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    x_label: str,
    y_label: str,
) -> None:
    parts.append(f'<line class="axis" x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}"/>')
    parts.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}"/>')
    for i in range(6):
        x = x0 + i * w / 5
        xv = xmin + i * (xmax - xmin) / 5
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+h}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{y0+h+18}" text-anchor="middle">{xv/1000:.0f}k</text>')
    for i in range(6):
        y = y0 + i * h / 5
        yv = ymax - i * (ymax - ymin) / 5
        parts.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{x0-8}" y="{y+4:.1f}" text-anchor="end">{yv:.2f}</text>')
    parts.append(f'<text class="label" x="{x0+w/2}" y="{y0+h+42}" text-anchor="middle">{x_label}</text>')
    parts.append(
        f'<text class="label" transform="translate({x0-48},{y0+h/2}) rotate(-90)" text-anchor="middle">{y_label}</text>'
    )


def write_trajectory_svg(grouped: dict[str, dict[str, Any]], path: Path) -> None:
    selected = [k for k in ["rho1_a50", "sieve_a50", "vnts_a50", "rho1_a70", "sieve_a70", "vnts_a70"] if k in grouped]
    all_pts = [p for k in selected for p in grouped[k]["points"]]
    xmin, xmax = 1000, 25000
    ymin = min(p["mean"] for p in all_pts) - 0.15
    ymax = max(p["mean"] for p in all_pts) + 0.15
    width, height = 980, 620
    x0, y0, w, h = 88, 78, 820, 430
    parts = svg_header(width, height)
    parts.append('<text class="title" x="42" y="36">Validation PPL Across Saved Checkpoints</text>')
    parts.append('<text class="subtitle" x="42" y="58">Mean over available seeds; checkpoint metadata every 1k steps</text>')
    draw_axes(parts, x0, y0, w, h, xmin, xmax, ymin, ymax, "training step", "validation PPL")
    legend_x, legend_y = 705, 86
    labels = []
    for key in selected:
        g = grouped[key]
        color = COLORS[g["method"]]
        dash = " stroke-dasharray=\"6 4\"" if g["alpha"] == 0.5 else ""
        pts = [
            (
                scale(p["step"], xmin, xmax, x0, x0 + w),
                scale(p["mean"], ymin, ymax, y0 + h, y0),
            )
            for p in g["points"]
        ]
        line = polyline(pts, color)
        if dash:
            line = line.replace("/>", f"{dash}/>")
        parts.append(line)
        labels.append((f'{g["method"].upper()} α={g["alpha"]:.1f}', color, dash))
    for i, (label, color, dash) in enumerate(labels):
        y = legend_y + i * 22
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+34}" y2="{y}" stroke="{color}" stroke-width="3"{dash}/>')
        parts.append(f'<text class="legend" x="{legend_x+42}" y="{y+4}">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_final_ppl_svg(agg: list[dict[str, Any]], path: Path) -> None:
    rows = [r for r in agg if r["method"] in {"rho1", "sieve", "vnts"}]
    alphas = sorted({r["alpha"] for r in rows})
    methods = ["rho1", "sieve", "vnts"]
    width, height = 980, 560
    x0, y0, w, h = 86, 74, 820, 360
    vals = [r["mean_final_val_ppl"] for r in rows]
    ymin, ymax = min(vals) - 0.2, max(vals) + 0.35
    parts = svg_header(width, height)
    parts.append('<text class="title" x="42" y="36">Final Validation PPL by α</text>')
    parts.append('<text class="subtitle" x="42" y="58">Points show mean across seeds; whiskers are sample std</text>')
    parts.append(f'<line class="axis" x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}"/>')
    parts.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}"/>')
    for i in range(6):
        y = y0 + i * h / 5
        yv = ymax - i * (ymax - ymin) / 5
        parts.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{x0-8}" y="{y+4:.1f}" text-anchor="end">{yv:.2f}</text>')
    alpha_x = {a: x0 + (i + 0.5) * w / len(alphas) for i, a in enumerate(alphas)}
    offset = {"rho1": -32, "sieve": 0, "vnts": 32}
    lookup = {(r["method"], r["alpha"]): r for r in rows}
    for a in alphas:
        parts.append(f'<text class="tick" x="{alpha_x[a]:.1f}" y="{y0+h+24}" text-anchor="middle">α={a:.1f}</text>')
        for m in methods:
            r = lookup.get((m, a))
            if not r:
                continue
            x = alpha_x[a] + offset[m]
            y = scale(r["mean_final_val_ppl"], ymin, ymax, y0 + h, y0)
            err = r["std_final_val_ppl"]
            y_hi = scale(r["mean_final_val_ppl"] + err, ymin, ymax, y0 + h, y0)
            y_lo = scale(r["mean_final_val_ppl"] - err, ymin, ymax, y0 + h, y0)
            parts.append(f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" stroke="{COLORS[m]}" stroke-width="2"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{COLORS[m]}"/>')
    for i, m in enumerate(methods):
        x = 720
        y = 92 + i * 24
        parts.append(f'<circle cx="{x}" cy="{y}" r="5.5" fill="{COLORS[m]}"/>')
        parts.append(f'<text class="legend" x="{x+14}" y="{y+4}">{m.upper()}</text>')
    parts.append(f'<text class="label" transform="translate({x0-48},{y0+h/2}) rotate(-90)" text-anchor="middle">final validation PPL</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def mean_history(states: dict[str, dict[str, Any]], runs: list[Run], method: str, alpha: float) -> list[list[float]]:
    histories = []
    for r in runs:
        if r.method == method and abs(r.alpha - alpha) < 1e-9 and r.run_name in states:
            h = states[r.run_name]["history"]
            if h:
                histories.append(h)
    if not histories:
        return []
    n = min(len(h) for h in histories)
    out = []
    for i in range(n):
        row = []
        for k in range(4):
            row.append(sum(float(h[i][k]) for h in histories) / len(histories))
        out.append(row)
    return out


def write_weight_history_svg(states: dict[str, dict[str, Any]], runs: list[Run], path: Path) -> None:
    panels = [("sieve", 0.7, "SIEVE α=0.7"), ("vnts", 0.7, "VN-TS α=0.7")]
    width, height = 980, 610
    parts = svg_header(width, height)
    parts.append('<text class="title" x="42" y="36">Scorer Weight Trajectories</text>')
    parts.append('<text class="subtitle" x="42" y="58">Mean sampled scorer weights from final bandit histories; all four scoring functions shown</text>')
    for pi, (method, alpha, title) in enumerate(panels):
        x0 = 78 + pi * 465
        y0, w, h = 92, 380, 360
        hist = mean_history(states, runs, method, alpha)
        if not hist:
            continue
        xmin, xmax, ymin, ymax = 0, len(hist) - 1, 0, 1
        parts.append(f'<text class="label" x="{x0+w/2}" y="{y0-18}" text-anchor="middle">{title}</text>')
        draw_axes(parts, x0, y0, w, h, xmin, xmax, ymin, ymax, "rescore update", "weight")
        for k, scorer in enumerate(SCORERS):
            pts = [
                (scale(i, xmin, xmax, x0, x0 + w), scale(row[k], ymin, ymax, y0 + h, y0))
                for i, row in enumerate(hist)
            ]
            parts.append(polyline(pts, COLORS[scorer], 2.0))
    for i, scorer in enumerate(SCORERS):
        x, y = 382 + i * 92, 552
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x+28}" y2="{y}" stroke="{COLORS[scorer]}" stroke-width="3"/>')
        parts.append(f'<text class="legend" x="{x+34}" y="{y+4}">{scorer}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_variance_svg(variance: dict[str, Any], path: Path) -> None:
    width, height = 980, 560
    parts = svg_header(width, height)
    parts.append('<text class="title" x="42" y="36">Scorer Variance and Redundancy Diagnostics</text>')
    parts.append('<text class="subtitle" x="42" y="58">wikitext103, GPT-2L, α=0.7 seed 42; lower S_D variance is the stability-bias signal</text>')
    intra = variance.get("intra_batch_variance", {})
    across = variance.get("across_step_variance", {})
    metrics = [
        ("Late intra-batch variance", [intra[s]["variance"] for s in SCORERS]),
        ("Across-step Δ variance", [across[s]["variance_of_delta"] for s in SCORERS]),
    ]
    for pi, (title, vals) in enumerate(metrics):
        x0 = 80 + pi * 440
        y0, w, h = 100, 300, 320
        vmax = max(vals) * 1.12 if max(vals) > 0 else 1
        parts.append(f'<text class="label" x="{x0+w/2}" y="{y0-18}" text-anchor="middle">{title}</text>')
        parts.append(f'<line class="axis" x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}"/>')
        parts.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}"/>')
        for i, (scorer, val) in enumerate(zip(SCORERS, vals)):
            bw = 48
            x = x0 + 26 + i * 68
            y = scale(val, 0, vmax, y0 + h, y0)
            parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{y0+h-y:.1f}" fill="{COLORS[scorer]}"/>')
            parts.append(f'<text class="tick" x="{x+bw/2}" y="{y0+h+18}" text-anchor="middle">{scorer}</text>')
            label = f"{val:.2g}"
            parts.append(f'<text class="tick" x="{x+bw/2}" y="{y-6:.1f}" text-anchor="middle">{label}</text>')
    pairwise = variance.get("pairwise_spearman_late", {})
    x0, y0 = 692, 120
    parts.append(f'<text class="label" x="{x0+100}" y="{y0-22}" text-anchor="middle">Late Spearman ρ</text>')
    for i, a in enumerate(SCORERS):
        parts.append(f'<text class="tick" x="{x0-10}" y="{y0+i*42+26}" text-anchor="end">{a}</text>')
        parts.append(f'<text class="tick" x="{x0+i*42+21}" y="{y0-8}" text-anchor="middle">{a}</text>')
        for j, b in enumerate(SCORERS):
            if i == j:
                rho = 1.0
            else:
                key = f"{min(a,b)}_vs_{max(a,b)}"
                # Pair keys were written in a fixed non-lexical order.
                rho = pairwise.get(f"{a}_vs_{b}", pairwise.get(f"{b}_vs_{a}", 0.0))
            t = (rho + 1) / 2
            red = int(220 * t + 35 * (1 - t))
            blue = int(220 * (1 - t) + 35 * t)
            color = f"#{red:02x}f0{blue:02x}"
            parts.append(f'<rect x="{x0+j*42}" y="{y0+i*42}" width="40" height="40" fill="{color}" stroke="#fff"/>')
            parts.append(f'<text class="tick" x="{x0+j*42+20}" y="{y0+i*42+25}" text-anchor="middle">{rho:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_bin_heatmap_svg(states: dict[str, dict[str, Any]], runs: list[Run], path: Path) -> None:
    selected = [r for r in runs if r.method == "vnts" and abs(r.alpha - 0.7) < 1e-9 and r.run_name in states]
    if not selected:
        return
    weights = []
    for b in range(16):
        row = []
        for k in range(4):
            row.append(sum(states[r.run_name]["expected_weights_by_bin"][b][k] for r in selected) / len(selected))
        weights.append(row)
    width, height = 760, 620
    parts = svg_header(width, height)
    parts.append('<text class="title" x="42" y="36">VN-TS Final Expected Weights by Phase Bin</text>')
    parts.append('<text class="subtitle" x="42" y="58">α=0.7, mean over seeds 42/43/44; rows are 16 contextual bandit bins</text>')
    x0, y0, cell = 165, 92, 54
    for k, scorer in enumerate(SCORERS):
        parts.append(f'<text class="label" x="{x0+k*cell+cell/2}" y="{y0-12}" text-anchor="middle">{scorer}</text>')
    for b in range(16):
        parts.append(f'<text class="tick" x="{x0-12}" y="{y0+b*28+19}" text-anchor="end">bin {b:02d}</text>')
        for k in range(4):
            v = weights[b][k]
            intensity = int(245 - 165 * v)
            color = f"rgb({intensity},{intensity},{245})"
            parts.append(f'<rect x="{x0+k*cell}" y="{y0+b*28}" width="{cell-3}" height="25" fill="{color}" stroke="#fff"/>')
            parts.append(f'<text class="tick" x="{x0+k*cell+(cell-3)/2}" y="{y0+b*28+17}" text-anchor="middle">{v:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    runs = load_multiseed_runs() + load_vnts_runs()
    trajectories = fetch_metadata(runs)
    states = fetch_bandit_states(runs)
    variance = load_variance_analysis()
    grouped = grouped_trajectory(runs, trajectories)
    finals = final_summary(runs, trajectories)
    agg = aggregate_final(finals)

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "wikitext103",
        "model": "gpt2l",
        "scorers": SCORERS,
        "runs": finals,
        "aggregate_final": agg,
        "trajectory_groups": grouped,
        "bandit_states": states,
        "variance_analysis": variance,
        "notes": [
            "Periodic checkpoints contain metadata.json only.",
            "Final checkpoints contain bandit_state.pt with alpha[16,4] and sampled scorer-weight history.",
            "Rho-1 has fixed scorer weights and no bandit posterior.",
        ],
    }
    (PLOTS / "sieve_checkpoint_figure_data.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    write_csv(finals, PLOTS / "sieve_checkpoint_final_runs.csv")
    write_csv(agg, PLOTS / "sieve_checkpoint_final_aggregate.csv")
    write_trajectory_svg(grouped, PLOTS / "fig_val_ppl_trajectories.svg")
    write_final_ppl_svg(agg, PLOTS / "fig_final_ppl_by_alpha.svg")
    write_weight_history_svg(states, runs, PLOTS / "fig_scorer_weight_history.svg")
    if variance:
        write_variance_svg(variance, PLOTS / "fig_scorer_variance_diagnostics.svg")
    write_bin_heatmap_svg(states, runs, PLOTS / "fig_vnts_bin_weights_heatmap.svg")
    print(f"[done] wrote figures/data under {PLOTS}")


if __name__ == "__main__":
    main()
