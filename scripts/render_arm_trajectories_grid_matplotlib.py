from __future__ import annotations

  import json
  from pathlib import Path

  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib as mpl


  ROOT = Path(__file__).resolve().parents[1]
  DATA = ROOT / "plots" / "scorer_substitution_bandit_histories.json"
  OUT_DIR = ROOT / "plots"


  mpl.rcParams.update({
      "font.family": "serif",
      "font.serif": ["Times New Roman", "DejaVu Serif"],
      "font.size": 8,
      "axes.labelsize": 8,
      "axes.titlesize": 9,
      "xtick.labelsize": 7,
      "ytick.labelsize": 7,
      "legend.fontsize": 7,
      "axes.linewidth": 0.6,
      "lines.linewidth": 1.0,
      "xtick.major.width": 0.6,
      "ytick.major.width": 0.6,
      "xtick.major.size": 3,
      "ytick.major.size": 3,
      "pdf.fonttype": 42,
      "ps.fonttype": 42,
      "savefig.dpi": 300,
      "savefig.bbox": "tight",
      "savefig.pad_inches": 0.02,
  })


  SCORER_COLOR = {
      0: "#d62728",  # S_E
      1: "#1f77b4",  # S_U
      2: "#2ca02c",  # S_L
      3: "#ff7f0e",  # S_D / substituted
  }
  SCORER_LABEL = [
      r"$S_E$",
      r"$S_U$",
      r"$S_L$",
      r"$S_D$ substituted",
  ]


  def load_runs():
      with DATA.open() as f:
          data = json.load(f)

      runs = {}
      for r in data["runs"]:
          history = np.asarray(r["history"], dtype=float)
          steps = np.arange(len(history)) * 50
          runs[(r["variant"], int(r["seed"]))] = {
              "steps": steps,
              "history": history,
          }
      return runs


  def main():
      runs = load_runs()

      seeds = [42, 43, 44]
      variants = [
          ("bigram_freq", "Bigram-sub"),
          ("random_noise", "Noise-sub"),
      ]

      fig, axes = plt.subplots(
          3, 2,
          figsize=(7.0, 5.0),
          sharex=True,
          sharey=True,
      )

      for row, seed in enumerate(seeds):
          for col, (variant_key, variant_title) in enumerate(variants):
              ax = axes[row, col]
              r = runs[(variant_key, seed)]

              ax.axhline(
                  y=0.25,
                  color="0.55",
                  linestyle="-",
                  linewidth=0.45,
                  alpha=0.8,
                  zorder=0,
              )

              for k in range(4):
                  ax.plot(
                      r["steps"],
                      r["history"][:, k],
                      color=SCORER_COLOR[k],
                      linewidth=1.0,
                      label=SCORER_LABEL[k] if (row == 0 and col == 0) else None,
                  )

              if row == 0:
                  ax.set_title(variant_title, pad=4)

              if col == 0:
                  ax.set_ylabel(f"seed {seed}", rotation=0, labelpad=30, va="center")

              ax.set_xlim(0, 25000)
              ax.set_ylim(0.0, 1.0)
              ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
              ax.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
              ax.set_xticklabels(["0", "5k", "10k", "15k", "20k", "25k"])
              ax.grid(True, linewidth=0.25, alpha=0.35)

      fig.supxlabel("Training step", y=0.06, fontsize=8)
      fig.supylabel("Expected posterior weight", x=0.01, fontsize=8)

      handles, labels = axes[0, 0].get_legend_handles_labels()
      fig.legend(
          handles,
          labels,
          loc="lower center",
          ncol=4,
          frameon=False,
          bbox_to_anchor=(0.5, -0.005),
      )

      fig.tight_layout(rect=[0.035, 0.08, 1.0, 1.0])

      OUT_DIR.mkdir(exist_ok=True)
      pdf_path = OUT_DIR / "fig_arm_trajectories_grid.pdf"
      png_path = OUT_DIR / "fig_arm_trajectories_grid.png"

      fig.savefig(pdf_path)
      fig.savefig(png_path)
      plt.close(fig)

      print(f"Saved: {pdf_path}")
      print(f"Saved: {png_path}")


  if __name__ == "__main__":
      main()