from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plots" / "scorer_substitution_bandit_histories.json"
OUT = ROOT / "plots"
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"

SEEDS = [42, 43, 44]
VARIANTS = [("bigram_freq", "Bigram-sub"), ("random_noise", "Noise-sub")]
SCORER_COLOR = {
    0: "#d62728",
    1: "#1f77b4",
    2: "#2ca02c",
    3: "#ff7f0e",
}
SCORER_LABEL = {
    0: "S_E",
    1: "S_U",
    2: "S_L",
    3: "S_D substituted",
}


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def scale(v: float, lo: float, hi: float, a: float, b: float) -> float:
    return (a + b) / 2 if hi == lo else a + (v - lo) * (b - a) / (hi - lo)


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
        width_est = len(s) * size * 0.52
        if anchor == "center":
            x -= width_est / 2
        elif anchor == "right":
            x -= width_est
        weight = "font-weight 700" if bold else "font-weight 400"
        self.ops.append(
            f"fill {fill} stroke none {weight} font-size {size} text {x:.1f},{y:.1f} '{esc(s)}'"
        )

    def polyline(self, pts, color="#111111", width=2, every=1):
        pts = pts[::every]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            self.line(x1, y1, x2, y2, color, width)

    def save(self, stem: str):
        draw = "\n".join(self.ops)
        for suffix in ("png", "pdf"):
            out = OUT / f"{stem}.{suffix}"
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
            print(out)


def load_runs() -> dict[tuple[str, int], list[list[float]]]:
    data = json.loads(DATA.read_text())
    return {
        (r["variant"], int(r["seed"])): [[float(x) for x in row] for row in r["history"]]
        for r in data["runs"]
    }


def draw_panel(c: Canvas, x0, y0, w, h, hist: list[list[float]], show_y: bool, show_x: bool):
    xmin, xmax = 0, 25000
    ymin, ymax = 0.0, 1.0

    c.line(x0, y0 + h, x0 + w, y0 + h, "#111827", 1)
    c.line(x0, y0, x0, y0 + h, "#111827", 1)

    for step in [0, 5000, 10000, 15000, 20000, 25000]:
        x = scale(step, xmin, xmax, x0, x0 + w)
        c.line(x, y0, x, y0 + h, "#e5e7eb", 1)
        if show_x:
            label = "0" if step == 0 else f"{step // 1000}k"
            c.text(x, y0 + h + 20, label, 13, "#374151", "center")

    for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = scale(val, ymin, ymax, y0 + h, y0)
        c.line(x0, y, x0 + w, y, "#e5e7eb", 1)
        if show_y:
            c.text(x0 - 9, y + 5, f"{val:.2f}", 13, "#374151", "right")

    y_uniform = scale(0.25, ymin, ymax, y0 + h, y0)
    c.line(x0, y_uniform, x0 + w, y_uniform, "#9ca3af", 1)

    steps = [i * 50 for i in range(len(hist))]
    for k in range(4):
        pts = [
            (
                scale(steps[i], xmin, xmax, x0, x0 + w),
                scale(hist[i][k], ymin, ymax, y0 + h, y0),
            )
            for i in range(len(hist))
        ]
        c.polyline(pts, SCORER_COLOR[k], 3, every=3)


def main():
    runs = load_runs()
    c = Canvas(1720, 1260)

    left, top = 150, 115
    panel_w, panel_h = 665, 285
    col_gap, row_gap = 105, 68

    for col, (_, label) in enumerate(VARIANTS):
        x = left + col * (panel_w + col_gap) + panel_w / 2
        c.text(x, 62, label, 24, "#111827", "center", True)

    for row, seed in enumerate(SEEDS):
        y = top + row * (panel_h + row_gap) + panel_h / 2
        c.text(50, y + 8, f"seed {seed}", 20, "#111827", "center", True)

    for row, seed in enumerate(SEEDS):
        for col, (variant, _) in enumerate(VARIANTS):
            x0 = left + col * (panel_w + col_gap)
            y0 = top + row * (panel_h + row_gap)
            draw_panel(
                c,
                x0,
                y0,
                panel_w,
                panel_h,
                runs[(variant, seed)],
                show_y=(col == 0),
                show_x=(row == len(SEEDS) - 1),
            )

    c.text(left + panel_w + col_gap / 2, 1148, "Training step", 18, "#374151", "center")
    c.text(18, 54, "Expected posterior weight", 18, "#374151")

    legend_x, legend_y = 455, 1202
    for k in range(4):
        x = legend_x + k * 220
        c.line(x, legend_y, x + 48, legend_y, SCORER_COLOR[k], 4)
        c.text(x + 62, legend_y + 6, SCORER_LABEL[k], 17, "#111827")

    c.save("fig_arm_trajectories_grid")


if __name__ == "__main__":
    main()
