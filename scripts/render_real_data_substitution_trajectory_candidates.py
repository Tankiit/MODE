from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plots" / "scorer_substitution_bandit_histories.json"
OUT = ROOT / "plots" / "real_data_candidates"
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"

SCORERS = ["S_E", "S_U", "S_L", "S_D"]
SCORER_COLOR = {
    "S_E": "#d62728",
    "S_U": "#1f77b4",
    "S_L": "#2ca02c",
    "S_D": "#ff7f0e",
}
SEED_COLOR = {42: "#1f77b4", 43: "#ff7f0e", 44: "#2ca02c"}
VARIANT_COLOR = {"bigram_freq": "#9467bd", "random_noise": "#8c564b"}
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
        OUT.mkdir(parents=True, exist_ok=True)
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


def load_data() -> dict[tuple[str, int], dict]:
    data = json.loads(DATA.read_text())
    runs = {}
    for r in data["runs"]:
        key = (r["variant"], int(r["seed"]))
        hist = [[float(x) for x in row] for row in r["history"]]
        steps = [i * 50 for i in range(len(hist))]
        runs[key] = {
            "steps": steps,
            "history": hist,
            "ppl": float(r["final_val_ppl"]),
            "run_name": r["run_name"],
        }
    return runs


def axes(c: Canvas, x0, y0, w, h, xmin=0, xmax=25000, ymin=0.0, ymax=0.75, ylabel=False, xlabel=False):
    c.line(x0, y0 + h, x0 + w, y0 + h, "#111111", 1)
    c.line(x0, y0, x0, y0 + h, "#111111", 1)
    for i in range(4):
        x = x0 + i * w / 3
        val = xmin + i * (xmax - xmin) / 3
        c.line(x, y0, x, y0 + h, "#e5e7eb", 1)
        if xlabel:
            label = "0" if val == 0 else f"{val/1000:.0f}k"
            c.text(x, y0 + h + 24, label, 14, "#374151", "center")
    for i in range(4):
        y = y0 + i * h / 3
        val = ymax - i * (ymax - ymin) / 3
        c.line(x0, y, x0 + w, y, "#e5e7eb", 1)
        if ylabel:
            c.text(x0 - 10, y + 5, f"{val:.2f}", 14, "#374151", "right")


def line_trajectory(c: Canvas, x0, y0, w, h, steps, values, color, width=3, ymin=0.0, ymax=0.75):
    pts = [
        (scale(s, 0, 25000, x0, x0 + w), scale(v, ymin, ymax, y0 + h, y0))
        for s, v in zip(steps, values)
    ]
    c.polyline(pts, color, width, every=max(1, len(pts) // 180))


def candidate_r1_seed_paired(runs: dict[tuple[str, int], dict]):
    c = Canvas(2100, 1500)
    c.text(60, 68, "R1: Seed-paired bandit trajectories", 38, "#111827", bold=True)
    c.text(60, 112, "Rows are seeds; columns replace slot D with bigram rarity or Gaussian noise.", 22, "#4b5563")
    x0, y0, w, h = 190, 210, 760, 290
    gap_x, gap_y = 150, 95
    variants = ["bigram_freq", "random_noise"]
    seeds = [42, 43, 44]
    for row, seed in enumerate(seeds):
        for col, variant in enumerate(variants):
            x = x0 + col * (w + gap_x)
            y = y0 + row * (h + gap_y)
            axes(c, x, y, w, h, ylabel=(col == 0), xlabel=(row == 2))
            uniform_y = scale(0.25, 0, 0.75, y + h, y)
            c.line(x, uniform_y, x + w, uniform_y, "#888888", 2)
            r = runs[(variant, seed)]
            for k, scorer in enumerate(SCORERS):
                vals = [hist_row[k] for hist_row in r["history"]]
                line_trajectory(c, x, y, w, h, r["steps"], vals, SCORER_COLOR[scorer], 3)
            if row == 0:
                c.text(x + w / 2, y - 25, VARIANT_LABEL[variant], 24, "#111827", "center", True)
            if col == 0:
                c.text(x - 72, y + h / 2, f"Seed {seed}", 23, "#111827", "right", True)
    for i, scorer in enumerate(SCORERS):
        lx = 610 + i * 230
        ly = 1420
        c.line(lx, ly, lx + 50, ly, SCORER_COLOR[scorer], 5)
        c.text(lx + 65, ly + 7, scorer, 23, "#111827", bold=True)
    c.save("R1_seed_paired_grid_real")


def candidate_r2_cross_seed_dispersion(runs: dict[tuple[str, int], dict]):
    c = Canvas(2100, 1120)
    c.text(60, 68, "R2: Cross-seed dispersion by arm", 38, "#111827", bold=True)
    c.text(60, 112, "Within each variant, seed identity explains much of the trajectory variation.", 22, "#4b5563")
    x0, y0, w, h = 145, 205, 430, 300
    gap_x, gap_y = 55, 120
    for row, variant in enumerate(["bigram_freq", "random_noise"]):
        for col, scorer in enumerate(SCORERS):
            x = x0 + col * (w + gap_x)
            y = y0 + row * (h + gap_y)
            axes(c, x, y, w, h, ylabel=(col == 0), xlabel=(row == 1))
            uniform_y = scale(0.25, 0, 0.75, y + h, y)
            c.line(x, uniform_y, x + w, uniform_y, "#888888", 2)
            k = SCORERS.index(scorer)
            for seed in [42, 43, 44]:
                r = runs[(variant, seed)]
                vals = [hist_row[k] for hist_row in r["history"]]
                line_trajectory(c, x, y, w, h, r["steps"], vals, SEED_COLOR[seed], 3)
            if row == 0:
                c.text(x + w / 2, y - 25, scorer, 24, "#111827", "center", True)
            if col == 0:
                c.text(x - 42, y + h / 2, VARIANT_LABEL[variant], 22, "#111827", "right", True)
    for i, seed in enumerate([42, 43, 44]):
        lx = 740 + i * 220
        ly = 1045
        c.line(lx, ly, lx + 50, ly, SEED_COLOR[seed], 5)
        c.text(lx + 65, ly + 7, f"Seed {seed}", 23, "#111827", bold=True)
    c.save("R2_cross_seed_dispersion_real")


def l1(a: list[list[float]], b: list[list[float]]) -> list[float]:
    n = min(len(a), len(b))
    return [sum(abs(a[i][k] - b[i][k]) for k in range(4)) for i in range(n)]


def candidate_r3_seed_pairing_distance(runs: dict[tuple[str, int], dict]):
    c = Canvas(2100, 780)
    c.text(60, 68, "R3: Paired substitution distance vs cross-seed distance", 38, "#111827", bold=True)
    c.text(60, 112, "Same-seed bigram/noise runs stay closer than different seeds for much of training.", 22, "#4b5563")
    panels = [
        (150, 185, 820, 420, "Bigram-sub vs Noise-sub at same seed"),
        (1120, 185, 820, 420, "Cross-seed distance within each variant"),
    ]
    for x, y, w, h, title in panels:
        axes(c, x, y, w, h, ymin=0, ymax=1.2, ylabel=(x < 500), xlabel=True)
        c.text(x + w / 2, y - 25, title, 24, "#111827", "center", True)

    left_x, left_y, left_w, left_h, _ = panels[0]
    for seed in [42, 43, 44]:
        big = runs[("bigram_freq", seed)]
        noise = runs[("random_noise", seed)]
        vals = l1(big["history"], noise["history"])
        line_trajectory(c, left_x, left_y, left_w, left_h, big["steps"][: len(vals)], vals, SEED_COLOR[seed], 4, ymin=0, ymax=1.2)

    right_x, right_y, right_w, right_h, _ = panels[1]
    pairs = [(42, 43), (42, 44), (43, 44)]
    for variant in ["bigram_freq", "random_noise"]:
        for pair_i, (a, b) in enumerate(pairs):
            ra, rb = runs[(variant, a)], runs[(variant, b)]
            vals = l1(ra["history"], rb["history"])
            line_trajectory(
                c,
                right_x,
                right_y,
                right_w,
                right_h,
                ra["steps"][: len(vals)],
                vals,
                VARIANT_COLOR[variant],
                3 if pair_i == 0 else 2,
                ymin=0,
                ymax=1.2,
            )

    c.text(120, 168, "L1 distance", 18, "#374151")
    for i, seed in enumerate([42, 43, 44]):
        lx = 310 + i * 190
        ly = 705
        c.line(lx, ly, lx + 45, ly, SEED_COLOR[seed], 5)
        c.text(lx + 58, ly + 7, f"Seed {seed}", 21, "#111827", bold=True)
    for i, variant in enumerate(["bigram_freq", "random_noise"]):
        lx = 1180 + i * 280
        ly = 705
        c.line(lx, ly, lx + 45, ly, VARIANT_COLOR[variant], 5)
        c.text(lx + 58, ly + 7, VARIANT_LABEL[variant], 21, "#111827", bold=True)
    c.save("R3_pairing_distance_real")


def main():
    runs = load_data()
    candidate_r1_seed_paired(runs)
    candidate_r2_cross_seed_dispersion(runs)
    candidate_r3_seed_pairing_distance(runs)


if __name__ == "__main__":
    main()
