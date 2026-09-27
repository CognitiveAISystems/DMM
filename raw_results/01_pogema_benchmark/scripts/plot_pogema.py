#!/usr/bin/env python3
"""Reproduce the combined POGEMA CSR and SoC benchmark figure."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

ALGORITHMS = (
    "MAPF-GPT-85M",
    "MAPF-GPT-DDG-2M",
    "MAGAT+",
    "HMAGAT",
    "LC-MAPF-3M",
    "DMM-08M",
    "DMM-3M",
    "DMM-MICPO-08M",
    "DMM-MICPO-3M",
)

COLORS = {
    "MAPF-GPT-85M": "#674ea7",
    "MAPF-GPT-DDG-2M": "#005960",
    "MAGAT+": "#11c1ac",
    "HMAGAT": "#2e7d32",
    "LC-MAPF-3M": "#c1433c",
    "DMM-08M": "#f4a261",
    "DMM-3M": "#e67e22",
    "DMM-MICPO-08M": "#5b9bd5",
    "DMM-MICPO-3M": "#2471a3",
}

SPLITS = (
    ("01-random", "random", "Random Maps", (8, 16, 24, 32, 48, 64, 80, 96)),
    ("02-mazes", "mazes", "Mazes Maps", (8, 16, 24, 32, 48, 64, 80)),
    ("03-warehouse", "warehouse", "Warehouse", (32, 64, 96, 128, 160, 192)),
    ("04-movingai", "movingai", "Cities Tiles", (64, 128, 192, 256)),
)

VALID_TASK_COUNTS = {
    "01-random": 1020,
    "02-mazes": 896,
    "03-warehouse": 768,
    "04-movingai": 509,
}

INSET_POSITIONS = {
    "random": (0.63, 0.61, 0.37, 0.37),
    "mazes": (0.63, 0.61, 0.37, 0.37),
    "warehouse": (0.65, 0.64, 0.35, 0.35),
    "movingai": (0.63, 0.61, 0.37, 0.37),
}

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 13
LEGEND_SIZE = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PACKAGE_ROOT / "data",
        help="directory containing one subdirectory per benchmark split",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=PACKAGE_ROOT / "assets",
        help="directory containing the four map thumbnails",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "figure" / "CSR-SOC-dmmv2-combined.pdf",
        help="output PDF path",
    )
    return parser.parse_args()


def result_path(data_dir: Path, algorithm: str, split: str) -> Path:
    return data_dir / split / f"{algorithm}.json"


def load_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list")
    return payload


def instance_key(result: dict[str, Any]) -> tuple[str, int, int]:
    environment = result["env_grid_search"]
    return (
        environment.get("map_name", "unknown"),
        int(environment.get("seed", 0)),
        int(environment["num_agents"]),
    )


@lru_cache(maxsize=None)
def valid_instance_keys(data_dir: Path, split: str) -> set[tuple[str, int, int]]:
    """Return the valid task set used by the benchmark."""
    return {
        instance_key(result)
        for result in load_results(
            result_path(data_dir, "DMM-MICPO-08M", split)
        )
    }


def validate_inputs(data_dir: Path, assets_dir: Path) -> None:
    for split, asset_name, _, _ in SPLITS:
        if not (assets_dir / f"{asset_name}.png").is_file():
            raise FileNotFoundError(assets_dir / f"{asset_name}.png")

        for algorithm in (*ALGORITHMS, "LaCAM"):
            path = result_path(data_dir, algorithm, split)
            results = load_results(path)
            keys = [instance_key(result) for result in results]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{path}: duplicate logical instances")
            for result in results:
                if not isinstance(result.get("metrics"), dict):
                    raise ValueError(f"{path}: result is missing metrics")

        valid_count = len(valid_instance_keys(data_dir, split))
        expected_count = VALID_TASK_COUNTS[split]
        if valid_count != expected_count:
            raise ValueError(
                f"{split}: expected {expected_count} valid tasks, got {valid_count}"
            )


def load_csr_data(data_dir: Path) -> pd.DataFrame:
    rows = []
    for split, _, title, _ in SPLITS:
        allowed_keys = valid_instance_keys(data_dir, split)
        for algorithm in ALGORITHMS:
            for result in load_results(result_path(data_dir, algorithm, split)):
                if instance_key(result) not in allowed_keys:
                    continue
                rows.append(
                    {
                        "Algorithm": algorithm,
                        "Number of Agents": result["env_grid_search"]["num_agents"],
                        "Success Rate": result["metrics"]["CSR"],
                        "Dataset": title,
                    }
                )
    return pd.DataFrame(rows)


def common_instance_keys(
    data_dir: Path, split: str
) -> set[tuple[str, int, int]]:
    paths = [result_path(data_dir, "LaCAM", split)]
    paths.extend(result_path(data_dir, algorithm, split) for algorithm in ALGORITHMS)
    return set.intersection(
        *[
            {instance_key(result) for result in load_results(path)}
            for path in paths
        ]
    ) & valid_instance_keys(data_dir, split)


def relative_soc(
    data_dir: Path,
    split: str,
    algorithm: str,
    allowed_keys: set[tuple[str, int, int]],
) -> list[float]:
    reference = {
        instance_key(result): result["metrics"]
        for result in load_results(result_path(data_dir, "LaCAM", split))
    }
    ratios = []
    for result in load_results(result_path(data_dir, algorithm, split)):
        metrics = result["metrics"]
        key = instance_key(result)
        reference_metrics = reference.get(key)
        if (
            key not in allowed_keys
            or reference_metrics is None
            or reference_metrics.get("CSR", 0) <= 0
            or reference_metrics.get("SoC", 0) <= 0
            or "SoC" not in metrics
        ):
            continue
        ratios.append(metrics["SoC"] / reference_metrics["SoC"])
    return ratios


def add_csr_plot(
    axis: plt.Axes,
    data: pd.DataFrame,
    title: str,
    ticks: tuple[int, ...],
) -> None:
    split_data = data[data["Dataset"] == title].copy()
    positions = {agents: position for position, agents in enumerate(ticks)}
    split_data["Agent Category"] = split_data["Number of Agents"].map(positions)
    split_data = split_data.sort_values("Agent Category")

    plot = sns.lineplot(
        data=split_data,
        x="Agent Category",
        y="Success Rate",
        hue="Algorithm",
        style="Algorithm",
        hue_order=ALGORITHMS,
        style_order=ALGORITHMS,
        palette=COLORS,
        errorbar=("ci", 95),
        seed=0,
        markers=True,
        markersize=6.8,
        linewidth=2.1,
        sort=False,
        ax=axis,
    )
    if plot.get_legend() is not None:
        plot.get_legend().remove()

    axis.set_title(title, fontsize=TITLE_SIZE, pad=8)
    axis.set_xticks(range(len(ticks)), labels=[str(tick) for tick in ticks])
    axis.set_xlabel("Number of Agents", fontsize=LABEL_SIZE)
    axis.set_ylabel("Success Rate", fontsize=LABEL_SIZE)
    axis.tick_params(axis="both", labelsize=TICK_SIZE)
    axis.grid(True)


def add_soc_plot(
    axis: plt.Axes,
    data_dir: Path,
    assets_dir: Path,
    split: str,
    asset_name: str,
) -> None:
    allowed_keys = common_instance_keys(data_dir, split)
    values = [
        relative_soc(data_dir, split, algorithm, allowed_keys)
        for algorithm in ALGORITHMS
    ]
    if any(not value for value in values):
        raise ValueError(f"Missing SoC ratios for {split}")

    boxes = axis.boxplot(
        values,
        patch_artist=True,
        labels=[""] * len(values),
        showfliers=False,
        widths=0.62,
    )
    for patch, algorithm in zip(boxes["boxes"], ALGORITHMS):
        patch.set_facecolor(COLORS[algorithm])
    for median in boxes["medians"]:
        median.set_color("black")
        median.set_linewidth(2.0)
    for whisker in boxes["whiskers"]:
        whisker.set_linewidth(1.6)
    for cap in boxes["caps"]:
        cap.set_linewidth(1.6)

    axis.set_xlabel("")
    axis.set_ylabel("SoC Ratio", fontsize=LABEL_SIZE)
    axis.tick_params(axis="x", length=0)
    axis.tick_params(axis="y", labelsize=TICK_SIZE)

    ymin, ymax = axis.get_ylim()
    ticks = list(axis.get_yticks())
    if not any(abs(tick - 1.0) < 1e-9 for tick in ticks):
        ticks.append(1.0)
    ticks = sorted(tick for tick in ticks if ymin <= tick <= ymax)
    axis.set_yticks(ticks)
    axis.grid(False)
    for tick in ticks:
        axis.axhline(
            y=tick,
            color="black",
            linestyle="--",
            alpha=0.75,
            linewidth=0.8,
            zorder=0,
        )

    image = mpimg.imread(assets_dir / f"{asset_name}.png")
    image_axis = axis.inset_axes(INSET_POSITIONS[asset_name], frameon=True)
    image_axis.imshow(image)
    image_axis.set_xticks([])
    image_axis.set_yticks([])
    for spine in image_axis.spines.values():
        spine.set_visible(True)


def create_plot(data_dir: Path, assets_dir: Path, output_path: Path) -> None:
    validate_inputs(data_dir, assets_dir)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": TICK_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
    )
    csr_data = load_csr_data(data_dir)
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(16, 6.4),
        gridspec_kw={"height_ratios": [1, 1]},
    )

    for column, (split, asset_name, title, ticks) in enumerate(SPLITS):
        add_csr_plot(axes[0, column], csr_data, title, ticks)
        add_soc_plot(axes[1, column], data_dir, assets_dir, split, asset_name)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    display_labels = {
        "DMM-08M": "DMM-0.8M",
        "DMM-MICPO-08M": "DMM-MICPO-0.8M",
    }
    figure.legend(
        handles,
        [display_labels.get(label, label) for label in labels],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.032),
        ncol=len(ALGORITHMS),
        fancybox=True,
        frameon=True,
        fontsize=LEGEND_SIZE,
        handlelength=2.1,
        columnspacing=0.65,
        handletextpad=0.35,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.955,
        bottom=0.13,
        wspace=0.24,
        hspace=0.37,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(figure)


def main() -> None:
    arguments = parse_args()
    data_dir = arguments.data_dir.resolve()
    assets_dir = arguments.assets_dir.resolve()
    output_path = arguments.output.resolve()
    create_plot(data_dir, assets_dir, output_path)
    print(f"valid tasks: {sum(VALID_TASK_COUNTS.values())}")
    print(output_path)


if __name__ == "__main__":
    main()
