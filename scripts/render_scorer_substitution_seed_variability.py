from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
DATA = ROOT / "scorer_substitution_bandit_history_summary.json"
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"

SCORERS = ["S_E", "S_U", "S_L", "S_D"]
COL = {
    "S_E": "#2563eb",
    "S_U": "#f97316",
    "S_L": "#16a34a",
    "S_D": "#9333ea",
}
SEED_COL = {
    42: "#111827",
    43: "#6b7280",
    44: "#9ca3af",
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

    def circle(self, x, y, r, fill="#111111", stroke="white", width=3):
        self.ops.append(
            f"fill {fill} stroke {stroke} stroke-width {width} circle {x:.1f},{y:.1f} {x+r:.1f},{y:.1f}"
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

    def save(self, stem: str):
        draw = "\n".join(self.ops)
        for suffix in ("png", "pdf"):
            out = PLOTS / f"{stem}.{suffix}"
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


def draw_axes(c: Canvas, x0: int, y0: int, w: int, h: int, ymax: float = 0.7):
    c.line(x0, y0 + h, x0 + w, y0 + h, "#111827", 2)
    c.line(x0, y0, x0, y0 + h, "#111827", 2)
    for i in range(8):
        val = i * 0.1
        y = scale(val, 0, ymax, y0 + h, y0)
        c.line(x0, y, x0 + w, y, "#e5e7eb", 1)
        c.text(x0 - 16, y + 7, f"{val:.1f}", 20, "#374151", "right")
    for val, label, color in [(0.25, "uniform 0.25", "#9ca3af"), (0.40, "collapse threshold 0.40", "#ef4444")]:
        y = scale(val, 0, ymax, y0 + h, y0)
        c.line(x0, y, x0 + w, y, color, 3)
        c.text(x0 + w - 8, y - 8, label, 18, color, "right", bold=True)


def draw_slot_d_panel(c: Canvas, rows: list[dict]):
    x0, y0, w, h = 150, 250, 760, 630
    ymax = 0.7
    c.text(x0, y0 - 82, "A. Slot-D mass varies by seed", 34, "#111827", bold=True)
    c.text(x0, y0 - 42, "Late mean over final 50 bandit updates", 22, "#4b5563")
    draw_axes(c, x0, y0, w, h, ymax=ymax)
    xs = {"bigram_freq": x0 + w * 0.32, "random_noise": x0 + w * 0.70}
    labels = {"bigram_freq": "bigram", "random_noise": "noise"}
    by_seed: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[row["variant"]] = row
    for seed, by_variant in sorted(by_seed.items()):
        if "bigram_freq" not in by_variant or "random_noise" not in by_variant:
            continue
        pts = []
        for variant in ["bigram_freq", "random_noise"]:
            v = float(by_variant[variant]["late_mean_S_D"])
            pts.append((xs[variant], scale(v, 0, ymax, y0 + h, y0), v))
        c.line(pts[0][0], pts[0][1], pts[1][0], pts[1][1], SEED_COL[seed], 4)
        for x, y, v in pts:
            c.circle(x, y, 13, SEED_COL[seed], "white", 4)
            c.text(x + 21, y + 7, f"s{seed}: {v:.2f}", 18, SEED_COL[seed], bold=True)
    for variant, x in xs.items():
        c.text(x, y0 + h + 44, labels[variant], 25, "#111827", "center", True)
    c.text(x0 + w / 2, y0 + h + 86, "slot-D replacement", 22, "#374151", "center")
    c.text(x0 - 80, y0 - 18, "weight", 22, "#374151")


def draw_stack_panel(c: Canvas, rows: list[dict]):
    x0, y0, w, h = 1090, 250, 1180, 630
    c.text(x0, y0 - 82, "B. Final TS-4 scorer mixtures", 34, "#111827", bold=True)
    c.text(x0, y0 - 42, "Each bar is one seed; slot D is purple", 22, "#4b5563")
    c.line(x0, y0 + h, x0 + w, y0 + h, "#111827", 2)
    c.line(x0, y0, x0, y0 + h, "#111827", 2)
    for i in range(6):
        val = i * 0.2
        y = scale(val, 0, 1, y0 + h, y0)
        c.line(x0, y, x0 + w, y, "#e5e7eb", 1)
        c.text(x0 - 16, y + 7, f"{val:.1f}", 20, "#374151", "right")
    ordered = sorted(rows, key=lambda r: (r["variant"], int(r["seed"])))
    bar_w = 82
    gap = 42
    group_gap = 120
    x = x0 + 100
    last_variant = None
    group_centers = {}
    group_xs = {}
    for row in ordered:
        variant = row["variant"]
        if last_variant is not None and variant != last_variant:
            x += group_gap
        group_xs.setdefault(variant, []).append(x + bar_w / 2)
        y_top = y0 + h
        for scorer in SCORERS:
            v = float(row[f"final_{scorer}"])
            bh = v * h
            c.rect(x, y_top - bh, bar_w, bh, COL[scorer], "white", 2)
            y_top -= bh
        c.text(x + bar_w / 2, y0 + h + 34, f"s{row['seed']}", 20, "#374151", "center", True)
        c.text(x + bar_w / 2, y0 + h + 63, f"{float(row['final_S_D']):.2f}", 18, COL["S_D"], "center", True)
        x += bar_w + gap
        last_variant = variant
    for variant, xs in group_xs.items():
        group_centers[variant] = sum(xs) / len(xs)
    c.text(group_centers["bigram_freq"], y0 + h + 111, "bigram substitute", 24, "#111827", "center", True)
    c.text(group_centers["random_noise"], y0 + h + 111, "Gaussian noise substitute", 24, "#111827", "center", True)
    c.text(x0 + w - 70, y0 + 38, "final S_D labels", 18, COL["S_D"], "right", True)

    lx = x0 + 775
    ly = y0 - 5
    for i, scorer in enumerate(SCORERS):
        xx = lx + i * 105
        c.rect(xx, ly, 28, 18, COL[scorer])
        c.text(xx + 38, ly + 18, scorer, 20, "#111827", bold=True)


def main():
    data = json.loads(DATA.read_text())
    rows = data["runs"]
    c = Canvas(2450, 1120)
    c.text(72, 80, "Scorer-Substitution TS-4 Is Not Seed-Stable", 48, "#111827", bold=True)
    c.text(
        72,
        132,
        "Replacing slot D with bigram rarity or Gaussian noise yields similar average D mass, but large seed-to-seed variation.",
        25,
        "#4b5563",
    )
    draw_slot_d_panel(c, rows)
    draw_stack_panel(c, rows)
    agg = data["aggregate"]
    b = agg["bigram_freq"]["late_mean_weights_mean"]["S_D"]
    n = agg["random_noise"]["late_mean_weights_mean"]["S_D"]
    bs = agg["bigram_freq"]["late_mean_weights_std"]["S_D"]
    ns = agg["random_noise"]["late_mean_weights_std"]["S_D"]
    c.text(
        72,
        1050,
        f"Late slot-D mass: bigram {b:.2f}+/-{bs:.2f}, noise {n:.2f}+/-{ns:.2f}. Both are below a clean collapse story once seeds are pooled.",
        24,
        "#374151",
    )
    c.save("fig_scorer_substitution_seed_variability")


if __name__ == "__main__":
    main()
