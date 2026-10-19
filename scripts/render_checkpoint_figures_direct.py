from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
DATA = PLOTS / "sieve_checkpoint_figure_data.json"
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"

COL = {
    "rho1": "#2563eb",
    "sieve": "#f97316",
    "vnts": "#16a34a",
    "S_E": "#2563eb",
    "S_U": "#f97316",
    "S_L": "#16a34a",
    "S_D": "#9333ea",
}
SCORERS = ["S_E", "S_U", "S_L", "S_D"]


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def scale(v: float, lo: float, hi: float, a: float, b: float) -> float:
    return (a + b) / 2 if hi == lo else a + (v - lo) * (b - a) / (hi - lo)


class Canvas:
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.ops: list[str] = []

    def line(self, x1, y1, x2, y2, color="#111111", width=1):
        self.ops.append(f"stroke {color} stroke-width {width} line {x1:.1f},{y1:.1f} {x2:.1f},{y2:.1f}")

    def rect(self, x, y, w, h, fill="#ffffff", stroke="none", width=1):
        self.ops.append(
            f"fill {fill} stroke {stroke} stroke-width {width} rectangle {x:.1f},{y:.1f} {x+w:.1f},{y+h:.1f}"
        )

    def circle(self, x, y, r, fill="#111111", stroke="none", width=1):
        self.ops.append(
            f"fill {fill} stroke {stroke} stroke-width {width} circle {x:.1f},{y:.1f} {x+r:.1f},{y:.1f}"
        )

    def text(self, x, y, s, size=18, fill="#111111", anchor="left", weight="normal"):
        # Avoid ImageMagick gravity because it changes the coordinate frame.
        # Approximate text width well enough for labels and tick alignment.
        width_est = len(s) * size * 0.55
        if anchor == "center":
            x -= width_est / 2
        elif anchor == "right":
            x -= width_est
        style = "font-weight 700" if weight == "bold" else "font-weight 400"
        self.ops.append(
            f"fill {fill} stroke none {style} font-size {size} text {x:.1f},{y:.1f} '{esc(s)}'"
        )

    def polyline(self, pts, color="#111111", width=2, every=1):
        pts = pts[::every]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            self.line(x1, y1, x2, y2, color, width)

    def save(self, stem: str):
        png = PLOTS / f"{stem}.png"
        pdf = PLOTS / f"{stem}.pdf"
        draw = "\n".join(self.ops)
        for out in [png, pdf]:
            subprocess.run(
                [
                    "magick",
                    "-size",
                    f"{self.w}x{self.h}",
                    "xc:white",
                    "-font",
                    FONT,
                    "-draw",
                    draw,
                    str(out),
                ],
                check=True,
            )
        print(png)
        print(pdf)


def axes(c: Canvas, x0, y0, w, h, xmin, xmax, ymin, ymax, xlabel, ylabel, xfmt=None):
    c.line(x0, y0 + h, x0 + w, y0 + h, "#111", 1)
    c.line(x0, y0, x0, y0 + h, "#111", 1)
    for i in range(6):
        x = x0 + i * w / 5
        xv = xmin + i * (xmax - xmin) / 5
        c.line(x, y0, x, y0 + h, "#e5e7eb", 1)
        lab = xfmt(xv) if xfmt else f"{xv:.0f}"
        c.text(x, y0 + h + 24, lab, 18, "#444", "center")
    for i in range(6):
        y = y0 + i * h / 5
        yv = ymax - i * (ymax - ymin) / 5
        c.line(x0, y, x0 + w, y, "#e5e7eb", 1)
        c.text(x0 - 12, y + 6, f"{yv:.2f}", 18, "#444", "right")
    c.text(x0 + w / 2, y0 + h + 60, xlabel, 20, "#333", "center")
    c.text(x0, y0 - 24, ylabel, 20, "#333")


def final_ppl(data):
    agg = data["aggregate_final"]
    alphas = sorted({r["alpha"] for r in agg})
    methods = ["rho1", "sieve", "vnts"]
    vals = [r["mean_final_val_ppl"] for r in agg]
    ymin, ymax = min(vals) - 0.25, max(vals) + 0.35
    c = Canvas(2200, 1350)
    c.text(70, 70, "Final Validation PPL by alpha", 52, "#000", weight="bold")
    c.text(70, 125, "Mean +/- sample std; numeric labels show mean final PPL", 30, "#555")
    x0, y0, w, h = 190, 205, 1780, 850
    axes(c, x0, y0, w, h, 0, len(alphas) - 1, ymin, ymax, "", "final validation PPL", lambda _: "")
    alpha_x = {a: x0 + (i + 0.5) * w / len(alphas) for i, a in enumerate(alphas)}
    offsets = {"rho1": -58, "sieve": 0, "vnts": 58}
    lookup = {(r["method"], r["alpha"]): r for r in agg}
    for a in alphas:
        c.text(alpha_x[a], y0 + h + 48, f"alpha={a:.1f}", 24, "#333", "center")
        for m in methods:
            r = lookup.get((m, a))
            if not r:
                continue
            mean = r["mean_final_val_ppl"]
            sd = r["std_final_val_ppl"]
            x = alpha_x[a] + offsets[m]
            y = scale(mean, ymin, ymax, y0 + h, y0)
            yhi = scale(mean + sd, ymin, ymax, y0 + h, y0)
            ylo = scale(mean - sd, ymin, ymax, y0 + h, y0)
            c.line(x, yhi, x, ylo, "#888", 2)
            c.circle(x, y, 14, COL[m])
            c.text(x, y - 32, f"{mean:.3f}", 24, "#111", "center", "bold")
    for i, m in enumerate(methods):
        y = 230 + i * 56
        c.circle(1740, y, 14, COL[m])
        c.text(1780, y + 7, m.upper(), 28, "#222")
    c.save("fig_final_ppl_by_alpha_direct")


def trajectories(data):
    groups = data["trajectory_groups"]
    keys = [k for k in ["rho1_a50", "sieve_a50", "vnts_a50", "rho1_a70", "sieve_a70", "vnts_a70"] if k in groups]
    all_pts = [p for k in keys for p in groups[k]["points"]]
    ymin = min(p["mean"] for p in all_pts) - 0.2
    ymax = max(p["mean"] for p in all_pts) + 0.25
    c = Canvas(2400, 1450)
    c.text(70, 70, "Validation PPL Across Saved Checkpoints", 52, "#000", weight="bold")
    c.text(70, 125, "Mean across available seeds; final values printed at line endpoints", 30, "#555")
    x0, y0, w, h = 190, 205, 1720, 900
    axes(c, x0, y0, w, h, 1000, 25000, ymin, ymax, "training step", "validation PPL", lambda x: f"{x/1000:.0f}k")
    endpoint_counts = {}
    for key in keys:
        g = groups[key]
        m, a = g["method"], g["alpha"]
        pts = [
            (
                scale(p["step"], 1000, 25000, x0, x0 + w),
                scale(p["mean"], ymin, ymax, y0 + h, y0),
            )
            for p in g["points"]
        ]
        c.polyline(pts, COL[m], 5)
        lx, ly = pts[-1]
        final = g["points"][-1]["mean"]
        bucket = round(ly / 28)
        bump = endpoint_counts.get(bucket, 0)
        endpoint_counts[bucket] = bump + 1
        c.circle(lx, ly, 9, COL[m])
        c.text(lx + 24, ly - 20 + bump * 30, f"{m.upper()} a={a:.1f}: {final:.3f}", 22, "#111")
    for i, key in enumerate(keys):
        g = groups[key]
        y = 230 + i * 48
        c.line(1680, y, 1740, y, COL[g["method"]], 5)
        c.text(1765, y + 8, f'{g["method"].upper()} alpha={g["alpha"]:.1f}', 25, "#222")
    c.save("fig_val_ppl_trajectories_direct")


def scorer_weights(data):
    states = data["bandit_states"]
    runs = data["runs"]

    def mean_history(method, alpha):
        hs = []
        for r in runs:
            if r["method"] == method and abs(r["alpha"] - alpha) < 1e-9 and r["run_name"] in states:
                h = states[r["run_name"]]["history"]
                if h:
                    hs.append(h)
        n = min((len(h) for h in hs), default=0)
        out = []
        for i in range(n):
            out.append([sum(float(h[i][k]) for h in hs) / len(hs) for k in range(4)])
        return out

    panels = [("sieve", 0.7, "SIEVE alpha=0.7"), ("vnts", 0.7, "VNTS alpha=0.7")]
    c = Canvas(2400, 1350)
    c.text(70, 70, "Scorer Weight Trajectories", 52, "#000", weight="bold")
    c.text(70, 125, "Mean sampled scorer weights from final bandit histories; final weights printed on each panel", 30, "#555")
    for pi, (method, alpha, title) in enumerate(panels):
        hist = mean_history(method, alpha)
        if not hist:
            continue
        x0, y0, w, h = 150 + pi * 1080, 230, 860, 720
        c.text(x0 + w / 2, y0 - 36, title, 28, "#111", "center", "bold")
        axes(c, x0, y0, w, h, 0, len(hist) - 1, 0, 1, "rescore update", "weight", lambda x: f"{x:.0f}")
        every = max(1, len(hist) // 160)
        for k, scorer in enumerate(SCORERS):
            pts = [
                (scale(i, 0, len(hist) - 1, x0, x0 + w), scale(row[k], 0, 1, y0 + h, y0))
                for i, row in enumerate(hist)
            ]
            c.polyline(pts, COL[scorer], 4, every=every)
            c.text(x0 + w - 8, scale(hist[-1][k], 0, 1, y0 + h, y0) + 7, f"{scorer} {hist[-1][k]:.2f}", 20, COL[scorer], "right", "bold")
    for i, scorer in enumerate(SCORERS):
        x, y = 760 + i * 190, 1250
        c.line(x, y, x + 55, y, COL[scorer], 5)
        c.text(x + 68, y + 8, scorer, 24, "#222")
    c.save("fig_scorer_weight_history_direct")


def variance_diag(data):
    v = data["variance_analysis"]
    intra = v["intra_batch_variance"]
    across = v["across_step_variance"]
    c = Canvas(2200, 1250)
    c.text(70, 70, "Scorer Variance Diagnostics", 52, "#000", weight="bold")
    c.text(70, 125, "Numeric labels show variance; wikitext103 alpha=0.7 seed=42", 30, "#555")
    panels = [
        ("Late intra-batch variance", [intra[s]["variance"] for s in SCORERS]),
        ("Across-step delta variance", [across[s]["variance_of_delta"] for s in SCORERS]),
    ]
    for pi, (title, vals) in enumerate(panels):
        x0, y0, w, h = 160 + pi * 980, 235, 720, 650
        ymax = max(vals) * 1.15 if max(vals) else 1
        c.text(x0 + w / 2, y0 - 40, title, 30, "#111", "center", "bold")
        c.line(x0, y0 + h, x0 + w, y0 + h, "#111", 1)
        c.line(x0, y0, x0, y0 + h, "#111", 1)
        for i, s in enumerate(SCORERS):
            val = vals[i]
            bw = 110
            x = x0 + 70 + i * 155
            y = scale(val, 0, ymax, y0 + h, y0)
            c.rect(x, y, bw, y0 + h - y, COL[s])
            c.text(x + bw / 2, y0 + h + 34, s, 24, "#333", "center")
            c.text(x + bw / 2, y - 15, f"{val:.3g}", 22, "#111", "center", "bold")
    c.save("fig_scorer_variance_diagnostics_direct")


def heatmap(data):
    states = data["bandit_states"]
    runs = [r for r in data["runs"] if r["method"] == "vnts" and abs(r["alpha"] - 0.7) < 1e-9 and r["run_name"] in states]
    weights = []
    for b in range(16):
        weights.append([sum(states[r["run_name"]]["expected_weights_by_bin"][b][k] for r in runs) / len(runs) for k in range(4)])
    c = Canvas(1500, 1300)
    c.text(70, 70, "VNTS Final Expected Weights by Phase Bin", 44, "#000", weight="bold")
    c.text(70, 125, "alpha=0.7, mean over seeds 42/43/44; cell labels are expected weights", 26, "#555")
    x0, y0, cw, ch = 330, 205, 190, 55
    for k, s in enumerate(SCORERS):
        c.text(x0 + k * cw + cw / 2, y0 - 28, s, 24, "#111", "center", "bold")
    for b, row in enumerate(weights):
        y = y0 + b * ch
        c.text(x0 - 24, y + 35, f"bin {b:02d}", 20, "#333", "right")
        for k, val in enumerate(row):
            blue = int(245 - 155 * val)
            color = f"#{blue:02x}{blue:02x}ff"
            x = x0 + k * cw
            c.rect(x, y, cw - 8, ch - 5, color, "#ffffff", 1)
            c.text(x + (cw - 8) / 2, y + 34, f"{val:.2f}", 20, "#111", "center", "bold")
    c.save("fig_vnts_bin_weights_heatmap_direct")


def main():
    data = json.loads(DATA.read_text())
    final_ppl(data)
    trajectories(data)
    scorer_weights(data)
    variance_diag(data)
    heatmap(data)


if __name__ == "__main__":
    main()
