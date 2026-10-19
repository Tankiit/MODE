from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plots" / "scorer_substitution_bandit_histories.json"
OUT = ROOT / "plots"
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"

SEED_COLOR = {42: "#2563eb", 43: "#f97316", 44: "#16a34a"}
VARIANT_COLOR = {"bigram_freq": "#7c3aed", "random_noise": "#8b5e4f"}
VARIANT_LABEL = {"bigram_freq": "Bigram-sub", "random_noise": "Noise-sub"}


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

    def circle(self, x, y, r, fill="#111111", stroke="white", width=2):
        self.ops.append(
            f"fill {fill} stroke {stroke} stroke-width {width} circle {x:.1f},{y:.1f} {x+r:.1f},{y:.1f}"
        )

    def polygon(self, pts, fill="#111111", stroke="none", width=1):
        if len(pts) < 3:
            return
        body = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.ops.append(f"fill {fill} stroke {stroke} stroke-width {width} polygon {body}")

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


def l1(a: list[list[float]], b: list[list[float]]) -> list[float]:
    n = min(len(a), len(b))
    return [sum(abs(a[i][k] - b[i][k]) for k in range(4)) for i in range(n)]


def mean_series(series: list[list[float]]) -> list[float]:
    n = min(len(s) for s in series)
    return [statistics.mean(s[i] for s in series) for i in range(n)]


def band(series: list[list[float]]) -> tuple[list[float], list[float], list[float]]:
    n = min(len(s) for s in series)
    means, lows, highs = [], [], []
    for i in range(n):
        vals = [s[i] for s in series]
        means.append(statistics.mean(vals))
        lows.append(min(vals))
        highs.append(max(vals))
    return means, lows, highs


def axes(c: Canvas, x0, y0, w, h, xmin, xmax, ymin, ymax, ylabel, xlabel=True):
    c.line(x0, y0 + h, x0 + w, y0 + h, "#111827", 1)
    c.line(x0, y0, x0, y0 + h, "#111827", 1)
    for i in range(5):
        x = x0 + i * w / 4
        val = xmin + i * (xmax - xmin) / 4
        c.line(x, y0, x, y0 + h, "#e5e7eb", 1)
        if xlabel:
            c.text(x, y0 + h + 24, "0" if val == 0 else f"{val/1000:.0f}k", 15, "#374151", "center")
    for i in range(5):
        y = y0 + i * h / 4
        val = ymax - i * (ymax - ymin) / 4
        c.line(x0, y, x0 + w, y, "#e5e7eb", 1)
        c.text(x0 - 10, y + 6, f"{val:.2f}", 15, "#374151", "right")
    c.text(x0, y0 - 18, ylabel, 18, "#374151")


def draw_distance_panel(c: Canvas, runs: dict[tuple[str, int], list[list[float]]]):
    x0, y0, w, h = 110, 145, 870, 430
    xmin, xmax = 8000, 25000
    ymin, ymax = 0.0, 1.0
    axes(c, x0, y0, w, h, xmin, xmax, ymin, ymax, "L1 distance")

    paired = [l1(runs[("bigram_freq", s)], runs[("random_noise", s)]) for s in [42, 43, 44]]
    paired_mean, paired_lo, paired_hi = band(paired)

    cross = []
    for variant in ["bigram_freq", "random_noise"]:
        for a, b in [(42, 43), (42, 44), (43, 44)]:
            cross.append(l1(runs[(variant, a)], runs[(variant, b)]))
    cross_mean, cross_lo, cross_hi = band(cross)

    start_idx = 8000 // 50
    paired_post = [v for s in paired for v in s[start_idx:]]
    cross_post = [v for s in cross for v in s[start_idx:]]
    ratio = statistics.mean(cross_post) / statistics.mean(paired_post)
    steps = [i * 50 for i in range(start_idx, len(paired_mean))]
    paired_mean = paired_mean[start_idx:]
    paired_lo = paired_lo[start_idx:]
    paired_hi = paired_hi[start_idx:]
    cross_mean = cross_mean[start_idx:]
    cross_lo = cross_lo[start_idx:]
    cross_hi = cross_hi[start_idx:]

    def pts(vals):
        return [
            (
                scale(steps[i], xmin, xmax, x0, x0 + w),
                scale(max(ymin, min(ymax, vals[i])), ymin, ymax, y0 + h, y0),
            )
            for i in range(len(vals))
        ]

    every = 3
    # Show range bands lightly, then mean lines.
    c.polygon(pts(cross_hi)[::every] + list(reversed(pts(cross_lo)[::every])), "#eadcf8")
    c.polygon(pts(paired_hi)[::every] + list(reversed(pts(paired_lo)[::every])), "#dbeafe")
    c.polyline(pts(cross_mean), "#7c3aed", 5, every=every)
    c.polyline(pts(paired_mean), "#2563eb", 5, every=every)

    c.rect(x0 + 28, y0 + 26, 348, 80, "#ffffff", "#d1d5db", 1)
    c.line(x0 + 50, y0 + 55, x0 + 104, y0 + 55, "#2563eb", 5)
    c.text(x0 + 120, y0 + 61, "same seed: bigram vs noise", 17, "#111827")
    c.line(x0 + 50, y0 + 86, x0 + 104, y0 + 86, "#7c3aed", 5)
    c.text(x0 + 120, y0 + 92, "different seeds", 17, "#111827")


def draw_slotd_panel(c: Canvas, runs: dict[tuple[str, int], list[list[float]]]):
    x0, y0, w, h = 1160, 145, 760, 430
    xmin, xmax = 8000, 25000
    ymin, ymax = 0.0, 0.7
    axes(c, x0, y0, w, h, xmin, xmax, ymin, ymax, "E[w_D]")
    y_uniform = scale(0.25, ymin, ymax, y0 + h, y0)
    c.line(x0, y_uniform, x0 + w, y_uniform, "#9ca3af", 2)

    for variant in ["bigram_freq", "random_noise"]:
        color = VARIANT_COLOR[variant]
        label_y_offset = -12 if variant == "bigram_freq" else 18
        for seed in [42, 43, 44]:
            hist = runs[(variant, seed)]
            start_idx = xmin // 50
            vals = [row[3] for row in hist][start_idx:]
            steps = [i * 50 for i in range(start_idx, start_idx + len(vals))]
            pts = [
                (scale(steps[i], xmin, xmax, x0, x0 + w), scale(vals[i], ymin, ymax, y0 + h, y0))
                for i in range(len(vals))
            ]
            # Use variant color, seed labels at terminal points.
            c.polyline(pts, color, 3 if variant == "bigram_freq" else 2, every=3)
            lx, ly = pts[-1]
            c.circle(lx, ly, 6, SEED_COLOR[seed])
            c.text(lx - 10, ly + label_y_offset, f"s{seed}", 14, color, "right", True)

    c.rect(x0 + 28, y0 + 28, 250, 88, "#ffffff", "#d1d5db", 1)
    c.line(x0 + 50, y0 + 58, x0 + 106, y0 + 58, VARIANT_COLOR["bigram_freq"], 5)
    c.text(x0 + 123, y0 + 64, "bigram-sub", 18, "#111827")
    c.line(x0 + 50, y0 + 92, x0 + 106, y0 + 92, VARIANT_COLOR["random_noise"], 4)
    c.text(x0 + 123, y0 + 98, "noise-sub", 18, "#111827")


def main():
    runs = load_runs()
    c = Canvas(2050, 710)
    c.text(58, 54, "Scorer Substitution Does Not Determine Bandit Dynamics", 29, "#111827", bold=True)
    c.text(
        58,
        84,
        "Scorer-substitution bandit histories on WikiText-103, alpha=0.7; same-seed substitutions stay closer than cross-seed runs.",
        18,
        "#4b5563",
    )
    draw_distance_panel(c, runs)
    draw_slotd_panel(c, runs)
    c.save("fig_seed_dominance_bandit_dynamics")


if __name__ == "__main__":
    main()
