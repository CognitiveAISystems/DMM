"""Render the intent-communication ablation figure from the per-instance JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DOMAIN = "02-mazes"
AGENT_COUNTS = (8, 16, 24, 32, 48, 64, 80)
Z_95 = 1.959963984540054

# Reference first, then increasingly severe interventions; also the legend order.
VARIANTS = (
    ("full", "Full", dict(color="#54545C", marker="o", linestyle="-")),
    ("no-h", r"no-$h$", dict(color="#6772C8", marker="s", linestyle="--")),
    ("no-z", r"no-$z$", dict(color="#9C3A2E", marker="P", linestyle="--")),
    ("z0-only", r"$z^0$-only", dict(color="#D9A93F", marker="D", linestyle="--")),
    ("shuffled", "shuffled", dict(color="#4F9A9A", marker="^", linestyle="--")),
)

POLICIES = ("DMM-3M", "DMM-MICPO-3M")


def load_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return json.load(stream)


def wilson_interval(values: list[float]) -> tuple[float, float, float]:
    samples = np.asarray(values, dtype=float)
    mean = float(samples.mean())
    n = len(samples)
    denom = 1.0 + Z_95**2 / n
    center = (mean + Z_95**2 / (2.0 * n)) / denom
    radius = Z_95 * np.sqrt(mean * (1.0 - mean) / n + Z_95**2 / (4.0 * n**2)) / denom
    return mean, max(0.0, center - radius), min(1.0, center + radius)


def aggregate(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means, lower, upper = [], [], []
    for count in AGENT_COUNTS:
        values = [float(row["metrics"]["CSR"]) for row in rows
                  if int(row["env_grid_search"]["num_agents"]) == count]
        if len(values) != 128:
            raise ValueError(f"{len(values)} episodes at {count} agents, not 128")
        mean, lo, hi = wilson_interval(values)
        means.append(mean)
        lower.append(lo)
        upper.append(hi)
    return np.asarray(means), np.asarray(lower), np.asarray(upper)


def rcparams() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.15,
        "grid.color": "#9E9E9E",
        "grid.linewidth": 0.9,
        "grid.alpha": 0.78,
    })


def render(data_dir: Path, output_path: Path) -> None:
    rcparams()
    figure, axes = plt.subplots(1, len(POLICIES), figsize=(4.3 * len(POLICIES), 4.6),
                                squeeze=False)
    positions = np.arange(len(AGENT_COUNTS))
    legend_handles: dict[str, object] = {}

    for col, model in enumerate(POLICIES):
        axis = axes[0, col]
        for variant, label, style in VARIANTS:
            path = data_dir / DOMAIN / f"{model}-{variant}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            means, lower, upper = aggregate(load_rows(path))
            line = axis.plot(positions, means, label=label,
                             linewidth=2.7 if variant == "full" else 2.3,
                             markersize=8.0 if variant == "full" else 7.2,
                             markeredgecolor="white", markeredgewidth=0.8,
                             zorder=4 if variant == "full" else 3, **style)[0]
            axis.fill_between(positions, lower, upper, color=style["color"],
                              alpha=0.16, linewidth=0, zorder=1)
            legend_handles.setdefault(variant, line)

        axis.set_title(model, pad=10, fontweight="bold")
        if col == 0:
            axis.set_ylabel("Success Rate")
        axis.set_xticks(positions, labels=AGENT_COUNTS)
        axis.tick_params(axis="both", length=5, width=1.1, color="#222222")
        axis.grid(True, which="major", zorder=0.5)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.set_ylim(-0.03, 1.03)
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    figure.subplots_adjust(wspace=0.20, bottom=0.32, top=0.90, left=0.12, right=0.97)
    figure.supxlabel("Number of Agents", fontsize=15, y=0.19)
    figure.legend([legend_handles[variant] for variant, _, _ in VARIANTS],
                  [label for _, label, _ in VARIANTS],
                  loc="lower center", bbox_to_anchor=(0.53, 0.02), ncol=len(VARIANTS),
                  frameon=True, fancybox=True, borderpad=0.65, handlelength=2.2,
                  columnspacing=1.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    print(output_path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("figure/intent_ablation.pdf"))
    args = parser.parse_args()
    render(args.data_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
