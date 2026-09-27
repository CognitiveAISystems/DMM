"""Render the inference-time refinement-depth figure from the per-instance JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

K_TEST = (1, 2, 3, 4, 8, 12)
K_TRAIN = 4
K_TRAIN_LINE_COLOR = "#315159"
Z_95 = 1.959963984540054

DOMAINS = (
    ("01-random", "Random", 96),
    ("02-mazes", "Mazes", 80),
    ("03-warehouse", "Warehouse", 192),
    ("04-movingai", "Cities Tiles", 256),
)

# metric key, row label, whether a [0,1] proportion (Wilson instead of normal interval)
METRICS = (
    ("CSR", "Success Rate", True),
    ("makespan", "Makespan", False),
    ("a_collisions", "Collisions", False),
)

POLICIES = (
    ("DMM-3M", dict(color="#6768B7", fill="#BFC0EA", marker="o")),
    ("DMM-MICPO-3M", dict(color="#BB534F", fill="#ECB2B0", marker="s")),
)


def load_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return json.load(stream)


def normal_interval(values: list[float]) -> tuple[float, float, float]:
    samples = np.asarray(values, dtype=float)
    mean = float(samples.mean())
    n = len(samples)
    std = float(samples.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n)
    return mean, max(0.0, mean - Z_95 * sem), mean + Z_95 * sem


def wilson_interval(values: list[float]) -> tuple[float, float, float]:
    samples = np.asarray(values, dtype=float)
    mean = float(samples.mean())
    n = len(samples)
    denom = 1.0 + Z_95**2 / n
    center = (mean + Z_95**2 / (2.0 * n)) / denom
    radius = Z_95 * np.sqrt(mean * (1.0 - mean) / n + Z_95**2 / (4.0 * n**2)) / denom
    return mean, max(0.0, center - radius), min(1.0, center + radius)


def series(domain_dir: Path, model: str, max_agents: int, metric: str,
           is_proportion: bool) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Mean and 95% interval of one metric at the domain's largest team size."""
    interval = wilson_interval if is_proportion else normal_interval
    ks, means, lower, upper = [], [], [], []
    for k in K_TEST:
        path = domain_dir / f"{model}-K{k}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        values = [float(row["metrics"][metric]) for row in load_rows(path)
                  if int(row["env_grid_search"]["num_agents"]) == max_agents]
        if len(values) != 128:
            raise ValueError(f"{path} has {len(values)} episodes at {max_agents} agents, not 128")
        mean, lo, hi = interval(values)
        ks.append(k)
        means.append(mean)
        lower.append(lo)
        upper.append(hi)
    return ks, np.asarray(means), np.asarray(lower), np.asarray(upper)


def rcparams() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
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
    figure, axes = plt.subplots(len(METRICS), len(DOMAINS),
                                figsize=(4.7 * len(DOMAINS), 3.5 * len(METRICS)))
    policy_lines: dict[str, object] = {}
    k_train_line = None

    for col, (domain, title, max_agents) in enumerate(DOMAINS):
        for row, (metric, ylabel, is_proportion) in enumerate(METRICS):
            axis = axes[row, col]
            k_train_line = axis.axvline(K_TRAIN, color=K_TRAIN_LINE_COLOR, linestyle="--",
                                        linewidth=1.8, alpha=0.85, zorder=2)
            panel_values: list[float] = []
            for model, style in POLICIES:
                ks, means, lower, upper = series(
                    data_dir / domain, model, max_agents, metric, is_proportion
                )
                x_pos = np.asarray(ks, dtype=float)
                line = axis.plot(x_pos, means, label=model, color=style["color"],
                                 marker=style["marker"], linewidth=2.6, markersize=8.2,
                                 markeredgecolor="white", markeredgewidth=1.0, zorder=3)[0]
                axis.fill_between(x_pos, lower, upper, color=style["fill"], alpha=0.75,
                                  linewidth=0, zorder=1)
                policy_lines.setdefault(model, line)
                panel_values.extend(means.tolist() + lower.tolist() + upper.tolist())

            if row == 0:
                axis.set_title(f"{title} ({max_agents})", pad=10)
            if col == 0:
                axis.set_ylabel(ylabel)
            axis.set_xlim(K_TEST[0] - 0.4, K_TEST[-1] + 0.4)
            axis.set_xticks(K_TEST)
            axis.tick_params(axis="both", length=5, width=1.1, color="#222222")
            axis.grid(True, which="major", zorder=0.5)
            axis.set_axisbelow(True)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            if is_proportion:
                axis.set_ylim(-0.03, 1.03)
                axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            elif panel_values:
                lo, hi = min(panel_values), max(panel_values)
                margin = (hi - lo) * 0.10 + 1e-6
                axis.set_ylim(lo - margin, hi + margin)

    figure.subplots_adjust(wspace=0.26, hspace=0.32, bottom=0.21, top=0.95)
    figure.supxlabel(r"Test-time refinement rounds $K_{\mathrm{test}}$", fontsize=15, y=0.115)

    handles = [policy_lines[model] for model, _ in POLICIES]
    labels = [model for model, _ in POLICIES]
    handles.append(k_train_line)
    labels.append(rf"$K_{{\mathrm{{train}}}}={K_TRAIN}$")
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, 0.02),
                  ncol=len(handles), frameon=True, fancybox=True, borderpad=0.65,
                  handlelength=1.8, columnspacing=1.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    print(output_path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path,
                        default=Path("figure/round_depth_ablation.pdf"))
    args = parser.parse_args()
    render(args.data_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
