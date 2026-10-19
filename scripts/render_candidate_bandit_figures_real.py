from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
OUT = PLOTS / "fig_candidates_real"
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"

SIEVE_DATA = PLOTS / "sieve_checkpoint_figure_data.json"
SUB_DATA = PLOTS / "scorer_substitution_bandit_histories.json"

SCORERS = ["S_E", "S_U", "S_L", "S_D"]
SCORER_COLOR = {
    "S_E": "#d62728",
    "S_U": "#1f77b4",
    "S_L": "#2ca02c",
    "S_D": "#ff7f0e",
}
SCORER_LABEL = {
    "S_E": "S_E",
    "S_U": "S_U",
    "S_L": "S_L",
    "S_D": "S_D",
}


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def scale(v: float, lo: float, hi: float, a: float, b: float) -> float:
    return (a + b) / 2 if hi == lo else a + (v - lo) * (b - a) / (hi - lo)


def smooth(rows: list[list[float]], window: int = 9) -> list[list[float]]:
    if window <= 1:
        return rows
    half = window // 2
    out = []
    for i in range(len(rows)):
        lo = max(0, i - half)
        hi = min(len(rows), i + half + 1)
        n = hi - lo
        out.append([sum(rows[j][k] for j in range(lo, hi)) / n for k in range(4)])
    return out


def mean_history(histories: list[list[list[float]]]) -> list[list[float]]:
    n = min(len(h) for h in histories if h)
    out = []
    for i in range(n):
        out.append([statistics.mean(float(h[i][k]) for h in histories) for k in range(4)])
    return smooth(out)


def load_real_histories() -> dict[str, dict]:
    sieve = json.loads(SIEVE_DATA.read_text())
    sub = json.loads(SUB_DATA.read_text())

    ts4_runs = [
        v["history"]
        for k, v in sieve["bandit_states"].items()
        if k.startswith("sieve_wikitext103_gpt2l_a70_") and v.get("history")
    ]
    bigram_runs = [
        r["history"] for r in sub["runs"] if r["variant"] == "bigram_freq" and r.get("history")
    ]
    noise_runs = [
        r["history"] for r in sub["runs"] if r["variant"] == "random_noise" and r.get("history")
    ]

    return {
        "TS-4": {"history": mean_history(ts4_runs), "n": len(ts4_runs)},
        "Bigram-sub": {"history": mean_history(bigram_runs), "n": len(bigram_runs)},
        "Noise-sub": {"history": mean_history(noise_runs), "n": len(noise_runs)},
    }


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


def steps_for(hist: list[list[float]]) -> list[float]:
    n = len(hist)
    return [i * 25000 / max(n - 1, 1) for i in range(n)]


def axes(c: Canvas, x0, y0, w, h, ymin, ymax, ylabel=False):
    c.line(x0, y0 + h, x0 + w, y0 + h, "#111111", 1)
    c.line(x0, y0, x0, y0 + h, "#111111", 1)
    for i in range(6):
        y = y0 + i * h / 5
        val = ymax - i * (ymax - ymin) / 5
        c.line(x0, y, x0 + w, y, "#e5e7eb", 1)
        if ylabel:
            c.text(x0 - 10, y + 5, f"{val:.1f}", 15, "#374151", "right")
    for frac, label in [(0.0, "0"), (0.5, "12.5k"), (1.0, "25k")]:
        x = x0 + frac * w
        c.line(x, y0 + h, x, y0 + h + 7, "#111111", 1)
        c.text(x, y0 + h + 25, label, 14, "#374151", "center")


def draw_line_panel(c: Canvas, x0, y0, w, h, name: str, hist: list[list[float]], ylabel=False):
    ymin, ymax = 0.0, 0.75
    axes(c, x0, y0, w, h, ymin, ymax, ylabel=ylabel)
    c.text(x0 + w / 2, y0 - 20, name, 22, "#111111", "center", True)
    y_uniform = scale(0.25, ymin, ymax, y0 + h, y0)
    c.line(x0, y_uniform, x0 + w, y_uniform, "#7f7f7f", 2)
    steps = steps_for(hist)
    every = max(1, len(hist) // 180)
    for k, scorer in enumerate(SCORERS):
        pts = [
            (
                scale(steps[i], 0, 25000, x0, x0 + w),
                scale(row[k], ymin, ymax, y0 + h, y0),
            )
            for i, row in enumerate(hist)
        ]
        c.polyline(pts, SCORER_COLOR[scorer], 3, every=every)
    c.text(x0 + 8, y_uniform - 6, "uniform", 13, "#6b7280")


def candidate_1(data: dict):
    c = Canvas(2100, 640)
    c.text(55, 62, "Candidate 1: Side-by-side posterior trajectories", 32, "#111111", bold=True)
    c.text(55, 98, "Real TS-4 / bigram-sub / noise-sub histories; lines are seed means.", 20, "#4b5563")
    x0, y0, w, h, gap = 120, 160, 560, 330, 70
    for i, name in enumerate(["TS-4", "Bigram-sub", "Noise-sub"]):
        draw_line_panel(c, x0 + i * (w + gap), y0, w, h, name, data[name]["history"], ylabel=(i == 0))
        c.text(x0 + i * (w + gap) + w / 2, y0 + h + 58, f"n={data[name]['n']} seeds", 16, "#4b5563", "center")
    c.text(870, 590, "training step", 18, "#374151", "center")
    for i, scorer in enumerate(SCORERS):
        x = 560 + i * 220
        y = 545
        c.line(x, y, x + 45, y, SCORER_COLOR[scorer], 4)
        c.text(x + 58, y + 6, SCORER_LABEL[scorer], 18, "#111111", bold=True)
    c.save("cand1_side_by_side_real")


def draw_stack_panel(c: Canvas, x0, y0, w, h, name: str, hist: list[list[float]], ylabel=False):
    axes(c, x0, y0, w, h, 0, 1, ylabel=ylabel)
    c.text(x0 + w / 2, y0 - 20, name, 22, "#111111", "center", True)
    steps = steps_for(hist)
    every = max(1, len(hist) // 170)
    idxs = list(range(0, len(hist), every))
    if idxs[-1] != len(hist) - 1:
        idxs.append(len(hist) - 1)
    lower = [0.0 for _ in idxs]
    for k, scorer in enumerate(SCORERS):
        upper = [lower[j] + hist[i][k] for j, i in enumerate(idxs)]
        top = [
            (scale(steps[i], 0, 25000, x0, x0 + w), scale(upper[j], 0, 1, y0 + h, y0))
            for j, i in enumerate(idxs)
        ]
        bottom = [
            (scale(steps[i], 0, 25000, x0, x0 + w), scale(lower[j], 0, 1, y0 + h, y0))
            for j, i in reversed(list(enumerate(idxs)))
        ]
        c.polygon(top + bottom, SCORER_COLOR[scorer], "white", 1)
        lower = upper


def candidate_2(data: dict):
    c = Canvas(2100, 700)
    c.text(55, 62, "Candidate 2: Stacked simplex trajectories", 32, "#111111", bold=True)
    c.text(55, 98, "Stacked areas emphasize that the bandit weights sum to one.", 20, "#4b5563")
    x0, y0, w, h, gap = 120, 160, 560, 370, 70
    for i, name in enumerate(["TS-4", "Bigram-sub", "Noise-sub"]):
        draw_stack_panel(c, x0 + i * (w + gap), y0, w, h, name, data[name]["history"], ylabel=(i == 0))
    for i, scorer in enumerate(SCORERS):
        x = 560 + i * 220
        y = 620
        c.rect(x, y - 14, 40, 18, SCORER_COLOR[scorer])
        c.text(x + 54, y + 4, SCORER_LABEL[scorer], 18, "#111111", bold=True)
    c.save("cand2_stacked_area_real")


def candidate_3(data: dict):
    c = Canvas(1100, 760)
    c.text(55, 62, "Candidate 3: Slot-D trajectory overlay", 32, "#111111", bold=True)
    c.text(55, 98, "Focused view: bigram and noise do not produce a seed-mean D-collapse.", 20, "#4b5563")
    x0, y0, w, h = 130, 150, 820, 430
    ymin, ymax = 0.0, 0.65
    axes(c, x0, y0, w, h, ymin, ymax, ylabel=True)
    y_uniform = scale(0.25, ymin, ymax, y0 + h, y0)
    c.line(x0, y_uniform, x0 + w, y_uniform, "#7f7f7f", 2)
    c.text(x0 + w - 8, y_uniform - 6, "uniform", 16, "#6b7280", "right")
    styles = {
        "TS-4": ("#1f77b4", 5),
        "Bigram-sub": ("#ff7f0e", 5),
        "Noise-sub": ("#d62728", 5),
    }
    for name in ["TS-4", "Bigram-sub", "Noise-sub"]:
        hist = data[name]["history"]
        steps = steps_for(hist)
        pts = [
            (
                scale(steps[i], 0, 25000, x0, x0 + w),
                scale(row[3], ymin, ymax, y0 + h, y0),
            )
            for i, row in enumerate(hist)
        ]
        color, width = styles[name]
        c.polyline(pts, color, width, every=max(1, len(pts) // 260))
        lx, ly = pts[-1]
        c.circle(lx, ly, 8, color)
        c.text(lx - 12, ly - 12, f"{name} {hist[-1][3]:.2f}", 18, color, "right", True)
    c.text(x0 + w / 2, y0 + h + 70, "training step", 18, "#374151", "center")
    c.text(x0 - 30, y0 - 24, "E[w_D]", 18, "#374151")
    c.save("cand3_overlay_slot_d_real")


def main():
    data = load_real_histories()
    candidate_1(data)
    candidate_2(data)
    candidate_3(data)


if __name__ == "__main__":
    main()
