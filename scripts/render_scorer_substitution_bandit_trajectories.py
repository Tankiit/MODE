from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
DATA = PLOTS / "scorer_substitution_bandit_histories.json"
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"

SCORERS = ["S_E", "S_U", "S_L", "S_D"]
COL = {
    "S_E": "#2563eb",
    "S_U": "#f97316",
    "S_L": "#16a34a",
    "S_D": "#9333ea",
}


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def scale(v: float, lo: float, hi: float, a: float, b: float) -> float:
    return (a + b) / 2 if hi == lo else a + (v - lo) * (b - a) / (hi - lo)


def smooth(rows: list[list[float]], window: int = 15) -> list[list[float]]:
    out = []
    half = window // 2
    for i in range(len(rows)):
        lo = max(0, i - half)
        hi = min(len(rows), i + half + 1)
        n = hi - lo
        out.append([sum(rows[j][k] for j in range(lo, hi)) / n for k in range(4)])
    return out


class Canvas:
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self.ops: list[str] = []

    def line(self, x1, y1, x2, y2, color="#111111", width=1):
        self.ops.append(f"stroke {color} stroke-width {width} line {x1:.1f},{y1:.1f} {x2:.1f},{y2:.1f}")

    def rect(self, x, y, w, h, fill="#ffffff", stroke="none", width=1):
        self.ops.append(f"fill {fill} stroke {stroke} stroke-width {width} rectangle {x:.1f},{y:.1f} {x+w:.1f},{y+h:.1f}")

    def circle(self, x, y, r, fill="#111111", stroke="white", width=2):
        self.ops.append(f"fill {fill} stroke {stroke} stroke-width {width} circle {x:.1f},{y:.1f} {x+r:.1f},{y:.1f}")

    def text(self, x, y, s, size=18, fill="#111111", anchor="left", bold=False):
        width_est = len(s) * size * 0.53
        if anchor == "center":
            x -= width_est / 2
        elif anchor == "right":
            x -= width_est
        weight = "font-weight 700" if bold else "font-weight 400"
        self.ops.append(f"fill {fill} stroke none {weight} font-size {size} text {x:.1f},{y:.1f} '{esc(s)}'")

    def polyline(self, pts, color="#111111", width=4, every=1):
        pts = pts[::every]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            self.line(x1, y1, x2, y2, color, width)

    def save(self, stem: str):
        draw = "\n".join(self.ops)
        for suffix in ("png", "pdf"):
            out = PLOTS / f"{stem}.{suffix}"
            subprocess.run(
                ["magick", "-size", f"{self.w}x{self.h}", "xc:white", "-font", FONT, "-draw", draw, str(out)],
                check=True,
            )
            print(out)


def mean_history(data: dict, variant: str) -> tuple[list[list[float]], int, float]:
    histories = [r["history"] for r in data["runs"] if r["variant"] == variant and r["history"]]
    n = min(len(h) for h in histories)
    rows = []
    for i in range(n):
        rows.append([sum(float(h[i][k]) for h in histories) / len(histories) for k in range(4)])
    ppl = sum(float(r["final_val_ppl"]) for r in data["runs"] if r["variant"] == variant) / len(histories)
    return smooth(rows), len(histories), ppl


def draw_panel(c: Canvas, x0: int, y0: int, w: int, h: int, title: str, subtitle: str, hist: list[list[float]], nseed: int, ppl: float):
    ymin, ymax = 0.0, 0.75
    c.text(x0 + w / 2, y0 - 54, title, 34, "#111827", "center", True)
    c.text(x0 + w / 2, y0 - 20, f"{subtitle}; n={nseed}; mean PPL={ppl:.3f}", 20, "#4b5563", "center")
    for i in range(6):
        y = y0 + i * h / 5
        val = ymax - i * (ymax - ymin) / 5
        c.line(x0, y, x0 + w, y, "#e5e7eb", 1)
        c.text(x0 - 12, y + 6, f"{val:.2f}", 18, "#374151", "right")
    for frac, label in [(0.1, "10%"), (0.3, "30%"), (0.7, "70%")]:
        x = x0 + frac * w
        c.line(x, y0, x, y0 + h, "#d1d5db", 2)
        c.text(x, y0 + h + 30, label, 18, "#6b7280", "center")
    c.line(x0, y0 + h, x0 + w, y0 + h, "#111827", 1)
    c.line(x0, y0, x0, y0 + h, "#111827", 1)
    uy = scale(0.25, ymin, ymax, y0 + h, y0)
    c.line(x0, uy, x0 + w, uy, "#9ca3af", 2)
    c.text(x0 + 8, uy - 9, "uniform", 16, "#6b7280")

    every = max(1, len(hist) // 240)
    for k, scorer in enumerate(SCORERS):
        pts = [
            (
                scale(i, 0, len(hist) - 1, x0, x0 + w),
                scale(row[k], ymin, ymax, y0 + h, y0),
            )
            for i, row in enumerate(hist)
        ]
        c.polyline(pts, COL[scorer], 5, every=every)
        lx, ly = pts[-1]
        c.circle(lx, ly, 8, COL[scorer])
        c.text(lx - 8, ly - 12, f"{scorer} {hist[-1][k]:.2f}", 18, COL[scorer], "right", True)

    late = hist[int(0.8 * len(hist)):]
    late_sd = sum(row[3] for row in late) / len(late)
    c.text(x0 + 18, y0 + 36, f"late S_D={late_sd:.2f}", 23, COL["S_D"], bold=True)


def main():
    data = json.loads(DATA.read_text())
    bigram, n_bigram, ppl_bigram = mean_history(data, "bigram_freq")
    noise, n_noise, ppl_noise = mean_history(data, "random_noise")

    c = Canvas(2450, 1280)
    c.text(70, 76, "Scorer-Substitution Bandit Weight Trajectories", 50, "#111827", bold=True)
    c.text(70, 130, "Slot D is replaced with either exact bigram rarity or fresh Gaussian noise; vanilla TS in both panels.", 27, "#4b5563")
    c.text(160, 220, "scorer weight", 22, "#374151")
    draw_panel(c, 160, 245, 930, 690, "A1: Bigram Corpus Statistic", "S_D = -log f(prev, token), alpha=0.7", bigram, n_bigram, ppl_bigram)
    draw_panel(c, 1350, 245, 930, 690, "B1: Noisy Negative Control", "S_D = Gaussian noise per rescore, alpha=0.7", noise, n_noise, ppl_noise)
    c.text(1215, 1012, "training progress", 24, "#374151", "center")
    for i, scorer in enumerate(SCORERS):
        x = 770 + i * 235
        y = 1160
        c.line(x, y, x + 58, y, COL[scorer], 6)
        c.text(x + 75, y + 8, scorer, 26, "#111827", bold=True)
    c.text(70, 1230, "Source: final checkpoint bandit histories, 3 seeds each. Lines are seed means smoothed only for display.", 20, "#6b7280")
    c.save("fig_scorer_substitution_bandit_trajectories_direct")


if __name__ == "__main__":
    main()
