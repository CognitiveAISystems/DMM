"""Render the corridor refinement-depth and teacher-forcing figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from corridor_stats import canonical, load, p_valid_mean_ci

INDIGO = "#6768B7"
CORAL = "#BB534F"
CHARCOAL = "#2F2F2F"
CHANCE_GRAY = "#8A8A8A"
Y_RANGE = (0.45, 1.01)
K_TEST = (2, 4, 8, 12)
FLOORS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def chance_line(axis) -> None:
    axis.axhline(0.5, color=CHANCE_GRAY, linestyle=":", linewidth=2.2, alpha=0.85, zorder=1)


def points(axis, x, values, *, color, marker, size, connected, **kwargs):
    means, halves = zip(*values)
    style = dict(linewidth=3.4) if connected else dict(linestyle="none")
    axis.errorbar(x, means, yerr=halves, color=color, marker=marker, markersize=size,
                  markeredgecolor="white", markeredgewidth=1.3, capsize=8, capthick=2.4,
                  zorder=3, **style, **kwargs)


def panel_a(axis, data_dir: Path) -> None:
    results = load(data_dir / "round_generalization_matrix.json")["results"]
    values = [p_valid_mean_ci(results[f"train=4/test={k}"]) for k in K_TEST]
    chance_line(axis)
    axis.axvline(4, color=CHARCOAL, linestyle="--", linewidth=2.6, alpha=0.6, zorder=1)
    points(axis, K_TEST, values, color=INDIGO, marker="o", size=13, connected=True)
    axis.set_xlim(0.5, 13.5)
    axis.set_xticks(K_TEST)
    axis.set_xlabel(r"$K_{\mathrm{test}}$")
    axis.set_title("(a) Refinement depth", fontsize=18, fontweight="bold", pad=10)


def panel_b(axis, data_dir: Path) -> None:
    annealed = load(data_dir / "ablation_1_mechanism_isolation.json")["results"]
    disabled = load(data_dir / "ablation_3_onoff.json")["results"]
    standard = p_valid_mean_ci(canonical(data_dir))
    categories = ["None", r"$\beta_0$-only", r"$\beta_r$-only", "Both"]
    annealed_values = [p_valid_mean_ci(annealed[c])
                       for c in ("neither (tf=0)", "dirichlet-only", "round-only")] + [standard]
    disabled_values = [p_valid_mean_ci(disabled[c])
                       for c in ("both off", "dirichlet on", "round on")] + [standard]

    positions = np.arange(len(categories))
    chance_line(axis)
    points(axis, positions - 0.13, annealed_values, color=INDIGO, marker="o", size=13,
           connected=False, label="zero-floor")
    points(axis, positions + 0.13, disabled_values, color=CORAL, marker="s", size=12,
           connected=False, label="off from start")
    axis.set_xlim(-0.5, len(categories) - 0.5)
    axis.set_xticks(positions, labels=categories)
    axis.set_title("(b) TF mechanism", fontsize=18, fontweight="bold", pad=10)
    axis.legend(loc="lower right", frameon=True, fancybox=True, fontsize=13.5,
                handlelength=1.3, borderpad=0.5)


def panel_c(axis, data_dir: Path) -> None:
    results = load(data_dir / "ablation_2_floor_sweep.json")["results"]
    standard = p_valid_mean_ci(canonical(data_dir))
    values = [standard if floor == 0.8 else p_valid_mean_ci(results[f"{floor}"])
              for floor in FLOORS]
    chance_line(axis)
    axis.axvline(0.8, color=CHARCOAL, linestyle="--", linewidth=2.6, alpha=0.6, zorder=1)
    points(axis, FLOORS, values, color=INDIGO, marker="o", size=13, connected=True)
    axis.set_xlim(-0.05, 1.05)
    axis.set_xticks(FLOORS)
    axis.set_xlabel(r"Shared floor $\beta$")
    axis.set_title("(c) TF floor", fontsize=18, fontweight="bold", pad=10)


def panel_d(axis, data_dir: Path) -> None:
    results = load(data_dir / "ablation_4_anneal_vs_constant.json")["results"]
    standard = p_valid_mean_ci(canonical(data_dir))
    annealed_values = [p_valid_mean_ci(results["annealed to 0.4"]), standard]
    constant_values = [p_valid_mean_ci(results[c]) for c in ("constant 0.4", "constant 0.8")]

    positions = np.arange(2)
    chance_line(axis)
    for x, (annealed_mean, _), (constant_mean, _) in zip(positions, annealed_values,
                                                         constant_values):
        axis.plot([x - 0.10, x + 0.10], [annealed_mean, constant_mean],
                  color="#9E9E9E", linewidth=2.0, zorder=2)
    points(axis, positions - 0.10, annealed_values, color=INDIGO, marker="o", size=13,
           connected=False, label="annealed")
    points(axis, positions + 0.10, constant_values, color=CORAL, marker="s", size=12,
           connected=False, label="constant")
    axis.set_xlim(-0.5, 1.5)
    axis.set_xticks(positions, labels=[r"$\beta{=}0.4$", r"$\beta{=}0.8$"])
    axis.set_xlabel("TF floor")
    axis.set_title("(d) TF schedule", fontsize=18, fontweight="bold", pad=10)
    axis.legend(loc="lower right", frameon=True, fancybox=True, fontsize=13.5,
                handlelength=1.3, borderpad=0.5)


def rcparams() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 17,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.8,
        "grid.color": "#9E9E9E",
        "grid.linewidth": 1.2,
        "grid.alpha": 0.78,
    })


def render(data_dir: Path, output_path: Path) -> None:
    rcparams()
    figure, axes = plt.subplots(1, 4, figsize=(18.4, 4.8))
    panel_a(axes[0], data_dir)
    panel_b(axes[1], data_dir)
    panel_c(axes[2], data_dir)
    panel_d(axes[3], data_dir)

    for axis in axes:
        axis.set_ylim(*Y_RANGE)
        axis.tick_params(axis="both", length=6, width=1.6, color="#222222")
        axis.grid(True, which="major", zorder=0.5)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)

    figure.subplots_adjust(wspace=0.10, bottom=0.16, top=0.86, left=0.045, right=0.995)
    figure.text(0.005, 0.52, "Valid joint-action frequency", rotation=90, va="center",
                ha="left", fontsize=18)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    print(output_path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("figure/corridor_ablations.pdf"))
    args = parser.parse_args()
    render(args.data_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
