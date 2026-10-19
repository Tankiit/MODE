from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
SCORERS = ["S_E", "S_U", "S_L", "S_D"]


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def hex_color(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def mix(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> str:
    t = clamp(t, 0.0, 1.0)
    return hex_color(
        round(c1[0] + (c2[0] - c1[0]) * t),
        round(c1[1] + (c2[1] - c1[1]) * t),
        round(c1[2] + (c2[2] - c1[2]) * t),
    )


def corr_color(v: float) -> str:
    # Blue-white-red diverging map, centered at zero.
    blue = (37, 99, 235)
    white = (248, 250, 252)
    red = (220, 38, 38)
    if v < 0:
        return mix(blue, white, (v + 1.0) / 1.0)
    return mix(white, red, v)


class Canvas:
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self.ops: list[str] = []

    def line(self, x1, y1, x2, y2, color="#111", width=1):
        self.ops.append(
            f"stroke {color} stroke-width {width} line {x1:.1f},{y1:.1f} {x2:.1f},{y2:.1f}"
        )

    def rect(self, x, y, w, h, fill="#fff", stroke="none", width=1):
        self.ops.append(
            f"fill {fill} stroke {stroke} stroke-width {width} rectangle {x:.1f},{y:.1f} {x+w:.1f},{y+h:.1f}"
        )

    def text(self, x, y, s, size=18, fill="#111", anchor="left", bold=False):
        width_est = len(s) * size * 0.52
        if anchor == "center":
            x -= width_est / 2
        elif anchor == "right":
            x -= width_est
        weight = "font-weight 700" if bold else "font-weight 400"
        self.ops.append(
            f"fill {fill} stroke none {weight} font-size {size} text {x:.1f},{y:.1f} '{esc(s)}'"
        )

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


def load_matrix(path: Path) -> tuple[str, float, list[list[float]]]:
    data = json.loads(path.read_text())
    pairs = data["pairwise_spearman_late"]
    matrix = [[1.0 if i == j else 0.0 for j in range(len(SCORERS))] for i in range(len(SCORERS))]
    for i, a in enumerate(SCORERS):
        for j, b in enumerate(SCORERS):
            if i == j:
                continue
            key1 = f"{a}_vs_{b}"
            key2 = f"{b}_vs_{a}"
            matrix[i][j] = float(pairs.get(key1, pairs.get(key2)))
    return str(data["dataset"]), float(data["alpha"]), matrix


def draw_panel(c: Canvas, x0: int, y0: int, title: str, matrix: list[list[float]], cell: int = 116):
    c.text(x0, y0 - 56, title, 32, "#111827", bold=True)
    c.text(x0, y0 - 18, "Spearman rho at late checkpoint", 21, "#4b5563")

    for i, s in enumerate(SCORERS):
        c.text(x0 + 132 + i * cell + cell / 2, y0 + 33, s, 24, "#111827", "center", True)
        c.text(x0 + 90, y0 + 92 + i * cell + cell / 2, s, 24, "#111827", "right", True)

    for i in range(len(SCORERS)):
        for j in range(len(SCORERS)):
            v = matrix[i][j]
            x = x0 + 132 + j * cell
            y = y0 + 72 + i * cell
            c.rect(x, y, cell, cell, corr_color(v), "white", 3)
            color = "white" if abs(v) > 0.62 else "#111827"
            c.text(x + cell / 2, y + cell / 2 + 9, f"{v:+.2f}", 25, color, "center", True)

    c.rect(x0 + 132, y0 + 72, cell * 4, cell * 4, "none", "#111827", 2)


def draw_colorbar(c: Canvas, x: int, y: int, w: int, h: int):
    steps = 80
    for i in range(steps):
        v0 = -1.0 + 2.0 * i / (steps - 1)
        c.rect(x + i * w / steps, y, w / steps + 1, h, corr_color(v0), "none")
    c.rect(x, y, w, h, "none", "#111827", 1)
    for v, label in [(-1, "-1"), (0, "0"), (1, "+1")]:
        tx = x + (v + 1) / 2 * w
        c.line(tx, y + h, tx, y + h + 10, "#111827", 2)
        c.text(tx, y + h + 36, label, 20, "#111827", "center")
    c.text(x + w / 2, y - 18, "Spearman rho", 20, "#374151", "center")


def main():
    wt = load_matrix(ROOT / "variance_analysis_wikitext103_gpt2l_a70_s42.json")
    owm = load_matrix(ROOT / "variance_analysis_owm_gpt2l_a70_s42.json")

    c = Canvas(1780, 980)
    c.text(70, 78, "Scorer Rank-Correlation Heatmap", 48, "#111827", bold=True)
    c.text(
        70,
        132,
        "Low correlations with S_D argue against the scorer-redundancy explanation; S_U and S_L are highly redundant.",
        26,
        "#4b5563",
    )

    draw_panel(c, 100, 250, f"{wt[0]} alpha={wt[1]:.1f}", wt[2])
    draw_panel(c, 940, 250, f"{owm[0]} alpha={owm[1]:.1f}", owm[2])
    draw_colorbar(c, 650, 850, 480, 28)

    c.text(70, 930, "Source: same validation batch, late checkpoint, flattened token scores.", 20, "#6b7280")
    c.save("fig_spearman_correlation_heatmap_direct")


if __name__ == "__main__":
    main()
