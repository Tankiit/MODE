from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
DATA = PLOTS / "sieve_checkpoint_figure_data.json"
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
    if not rows:
        return []
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
        self.ops.append(
            f"stroke {color} stroke-width {width} line {x1:.1f},{y1:.1f} {x2:.1f},{y2:.1f}"
        )

    def rect(self, x, y, w, h, fill="#ffffff", stroke="none", width=1):
        self.ops.append(
            f"fill {fill} stroke {stroke} stroke-width {width} rectangle {x:.1f},{y:.1f} {x+w:.1f},{y+h:.1f}"
        )

    def text(self, x, y, s, size=18, fill="#111111", anchor="left", bold=False):
        width_est = len(s) * size * 0.53
        if anchor == "center":
            x -= width_est / 2
        elif anchor == "right":
            x -= width_est
        weight = "font-weight 700" if bold else "font-weight 400"
        self.ops.append(
            f"fill {fill} stroke none {weight} font-size {size} text {x:.1f},{y:.1f} '{esc(s)}'"
        )

    def circle(self, x, y, r, fill="#111111", stroke="white", width=2):
        self.ops.append(
            f"fill {fill} stroke {stroke} stroke-width {width} circle {x:.1f},{y:.1f} {x+r:.1f},{y:.1f}"
        )

    def polyline(self, pts: list[tuple[float, float]], color="#111111", width=4, every=1):
        pts = pts[::every]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            self.line(x1, y1, x2, y2, color, width)

    def save(self, stem: str):
        PLOTS.mkdir(exist_ok=True)
        draw = "\n".join(self.ops)
        for suffix in ("png", "pdf"):
            out = PLOTS / f"{stem}.{suffix}"
            subprocess.run(
                ["magick", "-size", f"{self.w}x{self.h}", "xc:white", "-font", FONT, "-draw", draw, str(out)],
                check=True,
            )
            print(out)


def mean_history(data: dict, method: str, alpha: float) -> tuple[list[list[float]], int]:
    states = data["bandit_states"]
    histories = []
    for run in data["runs"]:
        if run["method"] != method or abs(float(run["alpha"]) - alpha) > 1e-9:
            continue
        state = states.get(run["run_name"])
        if not state:
            continue
        hist = state.get("history") or []
        if hist:
            histories.append(hist)
    if not histories:
        return [], 0
    n = min(len(h) for h in histories)
    rows = []
    for i in range(n):
        rows.append([sum(float(h[i][k]) for h in histories) / len(histories) for k in range(4)])
    return smooth(rows), len(histories)


def draw_panel(c: Canvas, x0: int, y0: int, w: int, h: int, title: str, subtitle: str, hist: list[list[float]], nseed: int):
    ymin, ymax = 0.0, 0.72
    c.text(x0 + w / 2, y0 - 56, title, 34, "#111827", "center", True)
    c.text(x0 + w / 2, y0 - 22, f"{subtitle}; n={nseed} seed histories", 21, "#4b5563", "center")

    # Phase bands matching the four active bins.
    bands = [(0.0, 0.1, "#f9fafb"), (0.1, 0.3, "#ffffff"), (0.3, 0.7, "#f9fafb"), (0.7, 1.0, "#ffffff")]
    for lo, hi, color in bands:
        c.rect(x0 + lo * w, y0, (hi - lo) * w, h, color, "none")

    for i in range(5):
        y = y0 + i * h / 4
        val = ymax - i * (ymax - ymin) / 4
        c.line(x0, y, x0 + w, y, "#e5e7eb", 1)
        c.text(x0 - 14, y + 6, f"{val:.2f}", 18, "#374151", "right")
    for frac, label in [(0.1, "10%"), (0.3, "30%"), (0.7, "70%")]:
        x = x0 + frac * w
        c.line(x, y0, x, y0 + h, "#d1d5db", 2)
        c.text(x, y0 + h + 31, label, 18, "#6b7280", "center")
    c.line(x0, y0 + h, x0 + w, y0 + h, "#111827", 1)
    c.line(x0, y0, x0, y0 + h, "#111827", 1)

    # Uniform baseline.
    uy = scale(0.25, ymin, ymax, y0 + h, y0)
    c.line(x0, uy, x0 + w, uy, "#9ca3af", 2)
    c.text(x0 + 8, uy - 9, "uniform", 16, "#6b7280")

    if not hist:
        c.text(x0 + w / 2, y0 + h / 2, "no history available", 28, "#991b1b", "center", True)
        return

    every = max(1, len(hist) // 220)
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
        c.text(lx - 8, ly - 14, f"{scorer} {hist[-1][k]:.2f}", 18, COL[scorer], "right", True)

    late_sd = sum(row[3] for row in hist[int(len(hist) * 0.8):]) / max(1, len(hist) - int(len(hist) * 0.8))
    c.text(x0 + 18, y0 + 36, f"late S_D mean={late_sd:.2f}", 22, COL["S_D"], bold=True)


def main():
    data = json.loads(DATA.read_text())
    ts_hist, ts_n = mean_history(data, "sieve", 0.7)
    vnts_hist, vnts_n = mean_history(data, "vnts", 0.7)

    c = Canvas(2400, 1280)
    c.text(70, 78, "Dirichlet Weight Evolution", 52, "#111827", bold=True)
    c.text(
        70,
        132,
        "TS-SIEVE concentrates more mass on the deterministic frequency arm; VN-TS attenuates that collapse.",
        27,
        "#4b5563",
    )
    draw_panel(c, 160, 245, 900, 690, "TS-SIEVE", "vanilla Dirichlet TS, alpha=0.7", ts_hist, ts_n)
    draw_panel(c, 1320, 245, 900, 690, "VN-TS-SIEVE", "variance-normalized update, alpha=0.7", vnts_hist, vnts_n)

    c.text(1140, 1010, "training progress", 24, "#374151", "center")
    c.text(160, 220, "scorer weight", 22, "#374151")
    for i, scorer in enumerate(SCORERS):
        x = 760 + i * 230
        y = 1160
        c.line(x, y, x + 58, y, COL[scorer], 6)
        c.text(x + 75, y + 8, scorer, 26, "#111827", bold=True)
    c.text(
        70,
        1230,
        "Source: saved bandit histories from WikiText-103, alpha=0.7. Lines are seed means smoothed only for display.",
        20,
        "#6b7280",
    )
    c.save("fig_tier1_weight_evolution_direct")


if __name__ == "__main__":
    main()
