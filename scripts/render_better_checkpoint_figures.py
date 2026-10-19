from __future__ import annotations

import json
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
    "delta": "#111827",
    "zero": "#9ca3af",
}
SCORERS = ["S_E", "S_U", "S_L", "S_D"]


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def scale(v, lo, hi, a, b):
    return (a + b) / 2 if hi == lo else a + (v - lo) * (b - a) / (hi - lo)


class Canvas:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.ops: list[str] = []

    def line(self, x1, y1, x2, y2, color="#111", width=1):
        self.ops.append(f"stroke {color} stroke-width {width} line {x1:.1f},{y1:.1f} {x2:.1f},{y2:.1f}")

    def rect(self, x, y, w, h, fill="#fff", stroke="none", width=1):
        self.ops.append(f"fill {fill} stroke {stroke} stroke-width {width} rectangle {x:.1f},{y:.1f} {x+w:.1f},{y+h:.1f}")

    def circle(self, x, y, r, fill="#111", stroke="white", width=2):
        self.ops.append(f"fill {fill} stroke {stroke} stroke-width {width} circle {x:.1f},{y:.1f} {x+r:.1f},{y:.1f}")

    def text(self, x, y, s, size=18, fill="#111", anchor="left", bold=False):
        width_est = len(s) * size * 0.52
        if anchor == "center":
            x -= width_est / 2
        elif anchor == "right":
            x -= width_est
        weight = "font-weight 700" if bold else "font-weight 400"
        self.ops.append(f"fill {fill} stroke none {weight} font-size {size} text {x:.1f},{y:.1f} '{esc(s)}'")

    def polyline(self, pts, color="#111", width=4):
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            self.line(x1, y1, x2, y2, color, width)

    def save(self, stem: str):
        png = PLOTS / f"{stem}.png"
        pdf = PLOTS / f"{stem}.pdf"
        draw = "\n".join(self.ops)
        for out in [png, pdf]:
            subprocess.run(
                ["magick", "-size", f"{self.w}x{self.h}", "xc:white", "-font", FONT, "-draw", draw, str(out)],
                check=True,
            )
        print(png)
        print(pdf)


def grouped_runs(data):
    out = {}
    for r in data["runs"]:
        out.setdefault((r["method"], float(r["alpha"])), []).append(r)
    return out


def alpha_seed_curve(data):
    runs = grouped_runs(data)
    rows = data["aggregate_final"]
    methods = ["rho1", "sieve", "vnts"]
    alphas = sorted({float(r["alpha"]) for r in rows})
    vals = [r["final_val_ppl"] for r in data["runs"]]
    ymin, ymax = min(vals) - 0.15, max(vals) + 0.22
    c = Canvas(2200, 1250)
    c.text(70, 70, "Alpha Sensitivity with Seed Scatter", 50, bold=True)
    c.text(70, 125, "Small dots are individual seeds; thick lines connect method means", 28, "#555")
    x0, y0, w, h = 190, 210, 1680, 780
    for i in range(6):
        y = y0 + i * h / 5
        yv = ymax - i * (ymax - ymin) / 5
        c.line(x0, y, x0 + w, y, "#e5e7eb", 1)
        c.text(x0 - 12, y + 6, f"{yv:.2f}", 18, "#444", "right")
    for i, a in enumerate(alphas):
        x = scale(i, 0, len(alphas) - 1, x0 + 90, x0 + w - 90)
        c.line(x, y0, x, y0 + h, "#f3f4f6", 1)
        c.text(x, y0 + h + 40, f"alpha={a:.1f}", 24, "#333", "center")
    c.line(x0, y0 + h, x0 + w, y0 + h, "#111", 1)
    c.line(x0, y0, x0, y0 + h, "#111", 1)
    agg_lookup = {(r["method"], float(r["alpha"])): r for r in rows}
    offsets = {"rho1": -20, "sieve": 0, "vnts": 20}
    for m in methods:
        mean_pts = []
        for i, a in enumerate(alphas):
            if (m, a) not in agg_lookup:
                continue
            x = scale(i, 0, len(alphas) - 1, x0 + 90, x0 + w - 90)
            mean = agg_lookup[(m, a)]["mean_final_val_ppl"]
            y = scale(mean, ymin, ymax, y0 + h, y0)
            mean_pts.append((x, y))
            for rr in runs.get((m, a), []):
                sx = x + offsets[m] + (rr["seed"] - 43) * 7
                sy = scale(rr["final_val_ppl"], ymin, ymax, y0 + h, y0)
                c.circle(sx, sy, 6, COL[m], "white", 2)
            c.circle(x, y, 12, COL[m], "white", 3)
            c.text(x, y - 24, f"{mean:.3f}", 18, COL[m], "center", True)
        c.polyline(mean_pts, COL[m], 5)
    for i, m in enumerate(methods):
        y = 220 + i * 52
        c.line(1660, y, 1720, y, COL[m], 6)
        c.text(1745, y + 7, m.upper(), 28, "#222")
    c.text(70, 1035, "final validation PPL", 22, "#333")
    c.save("fig_alpha_seed_scatter_direct")


def delta_plot(data):
    runs = grouped_runs(data)
    comparisons = [
        ("SIEVE - RHO1", "sieve", "rho1", [0.5, 0.7]),
        ("VNTS - SIEVE", "vnts", "sieve", [0.5, 0.7]),
        ("VNTS - RHO1", "vnts", "rho1", [0.5, 0.7]),
    ]
    rows = []
    for label, a_method, b_method, alphas in comparisons:
        for alpha in alphas:
            a_by_seed = {r["seed"]: r["final_val_ppl"] for r in runs.get((a_method, alpha), [])}
            b_by_seed = {r["seed"]: r["final_val_ppl"] for r in runs.get((b_method, alpha), [])}
            common = sorted(set(a_by_seed) & set(b_by_seed))
            if not common:
                continue
            vals = [a_by_seed[s] - b_by_seed[s] for s in common]
            rows.append({"label": label, "alpha": alpha, "mean": sum(vals) / len(vals), "vals": vals, "seeds": common})
    ymin = min(min(r["vals"]) for r in rows) - 0.04
    ymax = max(max(r["vals"]) for r in rows) + 0.04
    c = Canvas(1900, 1080)
    c.text(70, 70, "Paired Final-PPL Differences", 48, bold=True)
    c.text(70, 125, "Delta < 0 means the first method has lower PPL than the second", 28, "#555")
    x0, y0, w, h = 170, 210, 1450, 650
    zero = scale(0, ymin, ymax, y0 + h, y0)
    for i in range(6):
        y = y0 + i * h / 5
        yv = ymax - i * (ymax - ymin) / 5
        c.line(x0, y, x0 + w, y, "#e5e7eb", 1)
        c.text(x0 - 12, y + 6, f"{yv:+.2f}", 18, "#444", "right")
    c.line(x0, zero, x0 + w, zero, COL["zero"], 4)
    c.line(x0, y0, x0, y0 + h, "#111", 1)
    c.line(x0, y0 + h, x0 + w, y0 + h, "#111", 1)
    slot_w = w / len(rows)
    for i, r in enumerate(rows):
        x = x0 + slot_w * (i + 0.5)
        y = scale(r["mean"], ymin, ymax, y0 + h, y0)
        c.line(x, zero, x, y, COL["delta"], 5)
        c.circle(x, y, 12, COL["delta"])
        c.text(x, y - 26 if r["mean"] < 0 else y + 34, f"{r['mean']:+.3f}", 20, "#111", "center", True)
        for j, v in enumerate(r["vals"]):
            sy = scale(v, ymin, ymax, y0 + h, y0)
            c.circle(x - 20 + j * 20, sy, 5, "#6b7280")
        c.text(x, y0 + h + 38, f"{r['label']}\na={r['alpha']:.1f}", 18, "#333", "center")
    c.text(70, 905, "delta final PPL", 22, "#333")
    c.save("fig_paired_delta_direct")


def scorer_final_bars(data):
    states = data["bandit_states"]
    runs = data["runs"]
    rows = []
    for method, alpha in [("sieve", 0.7), ("vnts", 0.7), ("vnts", 0.9)]:
        selected = [r for r in runs if r["method"] == method and abs(r["alpha"] - alpha) < 1e-9 and r["run_name"] in states]
        if not selected:
            continue
        histories = [states[r["run_name"]]["history"] for r in selected]
        vals = [sum(float(h[-1][k]) for h in histories if h) / len(histories) for k in range(4)]
        rows.append({"label": f"{method.upper()} alpha={alpha:.1f}", "vals": vals})
    c = Canvas(1700, 980)
    c.text(70, 70, "Final Sampled Scorer Mix", 48, bold=True)
    c.text(70, 125, "Stacked bars show final mean sampled weights across seeds", 28, "#555")
    x0, y0, bar_w, bar_h = 300, 240, 980, 82
    for i, row in enumerate(rows):
        y = y0 + i * 150
        c.text(x0 - 30, y + 52, row["label"], 24, "#111", "right", True)
        x = x0
        for scorer, val in zip(SCORERS, row["vals"]):
            seg = bar_w * val
            c.rect(x, y, seg, bar_h, COL[scorer], "white", 2)
            if seg > 80:
                c.text(x + seg / 2, y + 52, f"{scorer} {val:.2f}", 22, "white", "center", True)
            else:
                c.text(x + seg + 8, y + 52, f"{scorer} {val:.2f}", 18, COL[scorer], "left", True)
            x += seg
        c.rect(x0, y, bar_w, bar_h, "none", "#111", 1)
    for i, scorer in enumerate(SCORERS):
        x = 480 + i * 210
        y = 800
        c.rect(x, y, 42, 24, COL[scorer])
        c.text(x + 56, y + 21, scorer, 24, "#111")
    c.save("fig_final_scorer_mix_direct")


def main():
    data = json.loads(DATA.read_text())
    alpha_seed_curve(data)
    delta_plot(data)
    scorer_final_bars(data)


if __name__ == "__main__":
    main()
