"""Render DMM spatial progress maps and per-step timing panels.

Use render.py for the paper figure from the data included in this package.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from pathlib import Path
from typing import BinaryIO

import numpy as np


MOVES = np.asarray(
    [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32
)
def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"expected {size:,} bytes, received {size - remaining:,}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def resolve_recorded_path(recorded: str, metadata_path: Path) -> Path:
    path = Path(recorded)
    if path.is_absolute() and path.exists():
        return path
    candidates = [Path.cwd() / path]
    candidates.extend(parent / path for parent in metadata_path.resolve().parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(recorded)


def apply_action_block(positions: np.ndarray, actions: np.ndarray) -> None:
    """Advance positions across a block without a [steps, agents, 2] array."""
    positions[:, 0] += (
        np.count_nonzero(actions == 2, axis=0)
        - np.count_nonzero(actions == 1, axis=0)
    ).astype(np.int32)
    positions[:, 1] += (
        np.count_nonzero(actions == 4, axis=0)
        - np.count_nonzero(actions == 3, axis=0)
    ).astype(np.int32)


def _binned_counts(indices: np.ndarray, mask: np.ndarray, bins: int) -> np.ndarray:
    return np.bincount(indices[mask], minlength=bins).astype(np.int64, copy=False)


def accumulate_pibt_locations(
    *,
    positions: np.ndarray,
    changed: np.ndarray,
    interval_changes: np.ndarray,
    location_counts: np.ndarray,
    bin_size: int,
    bin_columns: int,
) -> None:
    """Accumulate PIBT events at their pre-action spatial locations."""
    changed_ids = np.flatnonzero(changed)
    if not changed_ids.size:
        return
    interval_changes[changed_ids] += 1
    changed_positions = positions[changed_ids]
    event_bins = (
        (changed_positions[:, 0] // bin_size) * bin_columns
        + changed_positions[:, 1] // bin_size
    ).astype(np.int64, copy=False)
    location_counts += np.bincount(event_bins, minlength=len(location_counts))


def accumulate_moving_visits(
    *,
    positions: np.ndarray,
    actions: np.ndarray,
    location_counts: np.ndarray,
    bin_size: int,
    bin_columns: int,
) -> int:
    """Accumulate non-wait action destinations and return their count."""
    moving_ids = np.flatnonzero(actions)
    if not moving_ids.size:
        return 0
    destinations = positions[moving_ids] + MOVES[actions[moving_ids]]
    event_bins = (
        (destinations[:, 0] // bin_size) * bin_columns
        + destinations[:, 1] // bin_size
    ).astype(np.int64, copy=False)
    location_counts += np.bincount(event_bins, minlength=len(location_counts))
    return int(moving_ids.size)


def advance_phase_positions(
    *,
    positions: np.ndarray,
    actions: np.ndarray,
    position_bin_indices: np.ndarray,
    position_counts: np.ndarray,
    moving_visit_counts: np.ndarray,
    agent_presence_counts: np.ndarray,
    bin_size: int,
    bin_columns: int,
) -> int:
    """Advance one step while accumulating movement and agent occupancy.

    Position-bin counts are updated only for moving agents that cross a bin
    boundary.  Adding that compact count vector once per step is substantially
    cheaper than binning every agent again.
    """
    moving_ids = np.flatnonzero(actions)
    if moving_ids.size:
        destinations = positions[moving_ids] + MOVES[actions[moving_ids]]
        destination_bins = (
            (destinations[:, 0] // bin_size) * bin_columns
            + destinations[:, 1] // bin_size
        ).astype(np.int64, copy=False)
        moving_visit_counts += np.bincount(
            destination_bins, minlength=len(moving_visit_counts)
        )
        previous_bins = position_bin_indices[moving_ids]
        crossed = destination_bins != previous_bins
        if crossed.any():
            old_bins = previous_bins[crossed]
            new_bins = destination_bins[crossed]
            position_counts -= np.bincount(
                old_bins, minlength=len(position_counts)
            )
            position_counts += np.bincount(
                new_bins, minlength=len(position_counts)
            )
        positions[moving_ids] = destinations
        position_bin_indices[moving_ids] = destination_bins
    agent_presence_counts += position_counts
    return int(moving_ids.size)


def capture_snapshot(
    *,
    step: int,
    previous_step: int,
    positions: np.ndarray,
    goals: np.ndarray,
    goal_bin_indices: np.ndarray,
    goal_counts: np.ndarray,
    interval_changes: np.ndarray,
    current_bin_indices: np.ndarray,
    free_cell_counts: np.ndarray,
    pibt_location_counts: np.ndarray,
    target_isr: float | None = None,
    final: bool = False,
) -> dict:
    on_goal = np.all(positions == goals, axis=1)
    bins = len(goal_counts)
    on_goal_counts = _binned_counts(goal_bin_indices, on_goal, bins)
    unresolved_counts = goal_counts - on_goal_counts
    active_position_counts = _binned_counts(current_bin_indices, ~on_goal, bins)
    interval_length = step - previous_step
    if interval_length:
        changes = np.bincount(
            goal_bin_indices,
            weights=interval_changes,
            minlength=bins,
        ).astype(np.int64)
        changed_agents = _binned_counts(
            goal_bin_indices, interval_changes > 0, bins
        )
        pibt_location_rate = np.divide(
            pibt_location_counts,
            free_cell_counts * interval_length,
            out=np.zeros(bins, dtype=np.float64),
            where=free_cell_counts > 0,
        )
    else:
        changes = np.zeros(bins, dtype=np.int64)
        changed_agents = np.zeros(bins, dtype=np.int64)
        pibt_location_rate = np.zeros(bins, dtype=np.float64)
    return {
        "step": int(step),
        "previous_step": int(previous_step),
        "isr": float(on_goal.mean()),
        "on_goal_agents": int(on_goal.sum()),
        "unresolved_agents": int((~on_goal).sum()),
        "on_goal_counts": on_goal_counts,
        "unresolved_counts": unresolved_counts,
        "active_position_counts": active_position_counts,
        "pibt_changes": changes,
        "changed_agents": changed_agents,
        "pibt_location_counts": pibt_location_counts.copy(),
        "pibt_location_rate": pibt_location_rate,
        "pibt_changes_total": int(interval_changes.sum()),
        "pibt_changed_agents": int((interval_changes > 0).sum()),
        "target_isr": target_isr,
        "final": final,
    }


def capture_phase_traffic(
    *,
    step: int,
    previous_step: int,
    start_isr: float,
    end_isr: float,
    target_start_isr: float,
    target_end_isr: float,
    moving_visit_counts: np.ndarray,
    agent_presence_counts: np.ndarray,
    pibt_location_counts: np.ndarray,
    free_cell_counts: np.ndarray,
    pibt_changed_agents: int,
    final: bool = False,
) -> dict:
    """Capture spatial traffic and shielding statistics for one time phase."""
    interval_length = step - previous_step
    if interval_length <= 0:
        raise ValueError("phase intervals must contain at least one step")
    denominator = free_cell_counts.astype(np.float64) * interval_length
    moving_visit_rate = np.divide(
        moving_visit_counts,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    agent_density = np.divide(
        agent_presence_counts,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    pibt_location_rate = np.divide(
        pibt_location_counts,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return {
        "step": int(step),
        "previous_step": int(previous_step),
        "interval_steps": int(interval_length),
        "start_isr": float(start_isr),
        "end_isr": float(end_isr),
        "target_start_isr": float(target_start_isr),
        "target_end_isr": float(target_end_isr),
        "moving_visit_counts": moving_visit_counts.copy(),
        "moving_visit_rate": moving_visit_rate,
        "moving_visits_total": int(moving_visit_counts.sum()),
        "agent_presence_counts": agent_presence_counts.copy(),
        "agent_density": agent_density,
        "agent_presences_total": int(agent_presence_counts.sum()),
        "pibt_location_counts": pibt_location_counts.copy(),
        "pibt_location_rate": pibt_location_rate,
        "pibt_changes_total": int(pibt_location_counts.sum()),
        "pibt_changed_agents": int(pibt_changed_agents),
        "final": final,
    }


def smooth_spatial_grid(values: np.ndarray) -> np.ndarray:
    """Apply a small separable Gaussian-like filter without SciPy."""
    if values.ndim != 2:
        raise ValueError("spatial smoothing expects a two-dimensional grid")
    kernel = np.asarray([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    radius = len(kernel) // 2
    vertical = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    smoothed = sum(
        weight * vertical[index : index + values.shape[0]]
        for index, weight in enumerate(kernel)
    )
    horizontal = np.pad(smoothed, ((0, 0), (radius, radius)), mode="edge")
    return sum(
        weight * horizontal[:, index : index + values.shape[1]]
        for index, weight in enumerate(kernel)
    )


def pool_spatial_counts(
    values: np.ndarray,
    *,
    bin_rows: int,
    bin_columns: int,
    factor: int,
) -> np.ndarray:
    """Sum a flat fine-bin statistic into larger square regions."""
    if factor <= 0:
        raise ValueError("spatial pooling factor must be positive")
    if bin_rows % factor or bin_columns % factor:
        raise ValueError("fine-bin dimensions must be divisible by pooling factor")
    grid = np.asarray(values).reshape(bin_rows, bin_columns)
    return grid.reshape(
        bin_rows // factor,
        factor,
        bin_columns // factor,
        factor,
    ).sum(axis=(1, 3))


# Legacy palette sampled from the paper's graphical abstract. It is retained
# for reproducing earlier renders, but its diverging hues are a poor fit for
# ordered traffic intensity.
DMM_HEATMAP_COLORS = (
    "#eaf3f7",
    "#a7dfe6",
    "#56a0ae",
    "#1f5f6c",
    "#665885",
    "#ab2f2c",
)
DMM_HEATMAP_NAME = "dmm-paper"

# A restrained ColorBrewer-inspired sequential scale: low traffic stays pale,
# while increasingly active regions progress monotonically through blue.
TRAFFIC_HEATMAP_COLORS = (
    "#f7fbff",
    "#deebf7",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#3182bd",
    "#08519c",
    "#08306b",
)
TRAFFIC_HEATMAP_NAME = "dmm-traffic"
PIBT_ACCENT_COLOR = "#cc6677"

RUNTIME_COMPONENT_SPECS = (
    (
        "observation",
        "Observation preparation",
        "Observation",
        "#d39b5f",
    ),
    (
        "model_policy_and_message_comm",
        "Policy inference and message passing",
        "Policy",
        "#4477aa",
    ),
    (
        "logits_allgather",
        "Logits all-gather",
        "Logits all-gather",
        "#66ccee",
    ),
    (
        "shield_and_broadcast",
        "PIBT shielding and action broadcast",
        "CS-PIBT",
        PIBT_ACCENT_COLOR,
    ),
    (
        "state_and_metrics",
        "State update and metrics",
        "State + metrics",
        "#bbbbbb",
    ),
)
RUNTIME_FIGURE_OMITTED_KEYS = frozenset(
    {"logits_allgather", "state_and_metrics"}
)
RUNTIME_FIGURE_ORDER = (
    "model_policy_and_message_comm",
    "shield_and_broadcast",
    "observation",
)


def resolve_colormap(mpl, name: str):
    """Resolve a Matplotlib colormap or a DMM publication palette."""
    custom_palettes = {
        DMM_HEATMAP_NAME: DMM_HEATMAP_COLORS,
        TRAFFIC_HEATMAP_NAME: TRAFFIC_HEATMAP_COLORS,
    }
    if name in custom_palettes:
        return mpl.colors.LinearSegmentedColormap.from_list(
            name,
            custom_palettes[name],
            N=256,
        )
    return mpl.colormaps[name].copy()


def runtime_composition_from_result(result: dict) -> dict:
    """Extract max-rank, non-overlapping GPU-stage latencies from a run."""
    steady = result.get("timing_steady", {})
    raw_timing = result.get("timing_raw_ms", {})
    components = []
    per_step_count = None
    for key, label, short_label, color in RUNTIME_COMPONENT_SPECS:
        record = steady.get(key)
        if not isinstance(record, dict) or "mean_ms" not in record:
            raise ValueError(f"run result is missing timing_steady.{key}.mean_ms")
        mean_ms = float(record["mean_ms"])
        if mean_ms < 0:
            raise ValueError(f"negative steady-state timing for {key}")
        component = {
            "key": key,
            "label": label,
            "short_label": short_label,
            "color": color,
            "mean_ms": mean_ms,
        }
        raw_values = raw_timing.get(key)
        if raw_values is not None:
            values = [float(value) for value in raw_values]
            if any(value < 0 for value in values):
                raise ValueError(f"negative per-step timing for {key}")
            if per_step_count is None:
                per_step_count = len(values)
            elif len(values) != per_step_count:
                raise ValueError("per-step timing arrays have different lengths")
            component["per_step_ms"] = values
            component["episode_seconds"] = sum(values) / 1000.0
        components.append(component)
    component_sum_ms = sum(component["mean_ms"] for component in components)
    if component_sum_ms <= 0:
        raise ValueError("steady-state timing component sum must be positive")
    for component in components:
        component["percent"] = 100.0 * component["mean_ms"] / component_sum_ms
    gpu_total = steady.get("gpu_total", {})
    per_step_wall_seconds = result.get("step_seconds")
    if per_step_wall_seconds is not None:
        per_step_wall_seconds = [float(value) for value in per_step_wall_seconds]
        if any(value < 0 for value in per_step_wall_seconds):
            raise ValueError("negative per-step wall time")
        if per_step_count is not None and len(per_step_wall_seconds) != per_step_count:
            raise ValueError("per-step wall times and GPU timings have different lengths")
    elif per_step_count is not None:
        gpu_total_values = raw_timing.get("gpu_total")
        if gpu_total_values is not None:
            per_step_wall_seconds = [
                float(value) / 1000.0 for value in gpu_total_values
            ]
        else:
            per_step_wall_seconds = (
                np.asarray(
                    [component["per_step_ms"] for component in components],
                    dtype=np.float64,
                ).sum(axis=0)
                / 1000.0
            ).tolist()
    return {
        "basis": "max_rank_non_overlapping_gpu_stage_latencies",
        "component_sum_ms": component_sum_ms,
        "gpu_total_mean_ms": (
            float(gpu_total["mean_ms"])
            if isinstance(gpu_total, dict) and "mean_ms" in gpu_total
            else None
        ),
        "prefill_step_seconds": (
            float(result["prefill_step_seconds"])
            if result.get("prefill_step_seconds") is not None
            else None
        ),
        "prefill_observation_ms": (
            float(result["prefill_observation_ms"])
            if result.get("prefill_observation_ms") is not None
            else None
        ),
        "per_step_count": per_step_count,
        "per_step_wall_seconds": per_step_wall_seconds,
        "wall_seconds": (
            float(result["wall_seconds"])
            if result.get("wall_seconds") is not None
            else None
        ),
        "components": components,
    }


def runtime_composition_metadata(composition: dict | None) -> dict | None:
    """Return the scalar, publication-side subset of runtime composition."""
    if composition is None:
        return None
    metadata = {
        key: value
        for key, value in composition.items()
        if key not in {"components", "per_step_wall_seconds"}
    }
    metadata["components"] = [
        {key: value for key, value in component.items() if key != "per_step_ms"}
        for component in composition["components"]
    ]
    return metadata


def centered_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centered rolling mean without zero-padding edge bias."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("rolling mean input must be one-dimensional")
    if window <= 0:
        raise ValueError("rolling mean window must be positive")
    if not len(array):
        return array.copy()
    left = (window - 1) // 2
    right = window // 2 + 1
    indices = np.arange(len(array))
    starts = np.maximum(0, indices - left)
    ends = np.minimum(len(array), indices + right)
    prefix = np.concatenate(([0.0], np.cumsum(array, dtype=np.float64)))
    return (prefix[ends] - prefix[starts]) / (ends - starts)


def _format_step(step: int) -> str:
    return f"{step:,}"


def finalize_figure_output(temporary: Path, output: Path) -> None:
    """Install a rendered figure and keep generated SVG diffs clean."""
    temporary.replace(output)
    if output.suffix.lower() == ".svg":
        source = output.read_text()
        output.write_text(
            "\n".join(line.rstrip() for line in source.splitlines()) + "\n"
        )


def render_progress_svg(
    *,
    output: Path,
    snapshots: list[dict],
    bin_rows: int,
    bin_columns: int,
    bin_size: int,
    agents: int,
    seed: int | str,
    hotspot_quantile: float,
    annotate_counts: bool = False,
    show_pibt_hotspots: bool = True,
    colormap: str = "Reds",
) -> dict:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    columns = min(3, len(snapshots))
    rows = math.ceil(len(snapshots) / columns)
    active_scale_max = max(
        int(snapshot["active_position_counts"].max(initial=0))
        for snapshot in snapshots
    )
    color_scale_max = max(2, active_scale_max)
    cmap = mpl.colormaps[colormap].copy()
    cmap.set_bad("#eef1f4")
    norm = LogNorm(vmin=1, vmax=color_scale_max)
    figure_width = 14.4
    figure_height = 11.6 if rows == 2 else 6.2 * rows
    background = "#f7f8fa"
    text_color = "#17202a"
    note_color = "#52606d"
    panel_metadata = []

    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "svg.image_inline": True,
            "pdf.fonttype": 42,
        }
    ):
        figure, axes_array = plt.subplots(
            rows,
            columns,
            figsize=(figure_width, figure_height),
            facecolor=background,
            squeeze=False,
        )
        figure.subplots_adjust(
            left=0.035,
            right=0.985,
            top=0.875,
            bottom=0.125,
            wspace=0.10,
            hspace=0.26,
        )
        figure.suptitle(
            f"Spatial distribution of agents still off goal — {agents:,} agents",
            x=0.035,
            y=0.965,
            ha="left",
            fontsize=19,
            fontweight="bold",
            color=text_color,
        )
        figure.text(
            0.035,
            0.928,
            f"DMM, seed {seed}; each square aggregates a {bin_size}×{bin_size} "
            "map region. Color uses one shared log scale across all panels."
            + (
                " Orange rings mark exact PIBT-change hotspots."
                if show_pibt_hotspots
                else ""
            ),
            ha="left",
            fontsize=10.5,
            color=note_color,
        )

        flat_axes = axes_array.reshape(-1)
        for index, snapshot in enumerate(snapshots):
            axis = flat_axes[index]
            active_counts = snapshot["active_position_counts"]
            change_rate = snapshot["pibt_location_rate"]
            positive_rates = change_rate[change_rate > 0]
            threshold = (
                float(np.quantile(positive_rates, hotspot_quantile))
                if positive_rates.size
                else math.inf
            )
            hotspot_ids = (
                np.flatnonzero(change_rate >= threshold)
                if math.isfinite(threshold)
                else np.empty(0, dtype=np.int64)
            )
            heatmap = np.ma.masked_equal(
                active_counts.reshape(bin_rows, bin_columns), 0
            )
            axis.imshow(
                heatmap,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                origin="upper",
                rasterized=True,
            )
            if show_pibt_hotspots and hotspot_ids.size:
                hotspot_rows, hotspot_columns = np.divmod(
                    hotspot_ids, bin_columns
                )
                axis.scatter(
                    hotspot_columns,
                    hotspot_rows,
                    s=9,
                    facecolors="none",
                    edgecolors="#ff5a1f",
                    linewidths=0.7,
                    zorder=3,
                )
            if annotate_counts:
                count_grid = active_counts.reshape(bin_rows, bin_columns)
                font_size = max(4.0, min(7.0, 78.0 / bin_columns))
                for square_row in range(bin_rows):
                    for square_column in range(bin_columns):
                        count = int(count_grid[square_row, square_column])
                        if count == 0:
                            count_color = "#9ca3af"
                        else:
                            normalized = float(norm(count))
                            count_color = "white" if normalized > 0.58 else "#202124"
                        axis.text(
                            square_column,
                            square_row,
                            f"{count:,}",
                            ha="center",
                            va="center",
                            fontsize=font_size,
                            color=count_color,
                            zorder=4,
                        )
            if snapshot.get("step") == 0:
                panel_title = "Initial state"
            elif snapshot.get("final"):
                panel_title = (
                    f'Final state · step {_format_step(snapshot["step"])}'
                )
            elif snapshot.get("target_isr") is not None:
                panel_title = (
                    f'{100 * snapshot["target_isr"]:g}% settled · '
                    f'step {_format_step(snapshot["step"])}'
                )
            else:
                panel_title = f'Step {_format_step(snapshot["step"])}'
            axis.set_title(
                panel_title,
                loc="left",
                pad=22,
                fontsize=12,
                fontweight="bold",
                color=text_color,
            )
            axis.text(
                0,
                1.015,
                f'ISR {100 * snapshot["isr"]:.3f}% · '
                f'{snapshot["unresolved_agents"]:,} agents off goal',
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.2,
                color=note_color,
            )
            top_tiles = max(1, math.ceil(0.01 * len(active_counts)))
            active_total = int(active_counts.sum())
            top_tile_agents = int(
                np.partition(active_counts, len(active_counts) - top_tiles)[
                    -top_tiles:
                ].sum()
            )
            top_tile_fraction = (
                top_tile_agents / active_total if active_total else 0.0
            )
            axis.text(
                0,
                -0.045,
                f"densest square {int(active_counts.max(initial=0)):,} agents · "
                f'{snapshot["pibt_changes_total"]:,} PIBT changes',
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8.0,
                color=note_color,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if annotate_counts:
                axis.set_xticks(
                    np.arange(-0.5, bin_columns, 1), minor=True
                )
                axis.set_yticks(np.arange(-0.5, bin_rows, 1), minor=True)
                axis.grid(
                    which="minor",
                    color="#c7cdd4",
                    linewidth=0.45,
                    alpha=0.85,
                )
                axis.tick_params(which="minor", length=0)
            for spine in axis.spines.values():
                spine.set_color("#4b5563")
                spine.set_linewidth(0.65)
            panel_metadata.append(
                {
                    "step": snapshot["step"],
                    "isr": snapshot["isr"],
                    "unresolved_agents": snapshot["unresolved_agents"],
                    "interval_start": snapshot["previous_step"],
                    "pibt_changes": snapshot["pibt_changes_total"],
                    "pibt_changed_agents": snapshot["pibt_changed_agents"],
                    "hotspot_threshold": (
                        None if not math.isfinite(threshold) else threshold
                    ),
                    "hotspot_tiles": int(len(hotspot_ids)),
                    "active_tiles": int(np.count_nonzero(active_counts)),
                    "max_off_goal_agents_per_tile": int(
                        active_counts.max(initial=0)
                    ),
                    "top_1pct_tile_agent_fraction": top_tile_fraction,
                    "off_goal_agents_by_square": active_counts.reshape(
                        bin_rows, bin_columns
                    ).astype(int).tolist(),
                    "target_isr": snapshot.get("target_isr"),
                    "final": bool(snapshot.get("final")),
                }
            )
        for axis in flat_axes[len(snapshots) :]:
            axis.set_visible(False)

        color_axis = figure.add_axes((0.035, 0.048, 0.25, 0.014))
        colorbar = figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=color_axis,
            orientation="horizontal",
        )
        legend_counts = sorted(
            set(
                (
                    1,
                    max(1, round(math.sqrt(active_scale_max))),
                    active_scale_max,
                )
            )
        )
        colorbar.set_ticks(legend_counts)
        colorbar.set_ticklabels([f"{count:,}" for count in legend_counts])
        colorbar.ax.tick_params(labelsize=8, length=0, colors=note_color)
        colorbar.outline.set_visible(False)
        colorbar.ax.set_title(
            f"Off-goal agents per {bin_size}×{bin_size}-cell tile",
            loc="left",
            fontsize=9.2,
            fontweight="bold",
            color=text_color,
            pad=7,
        )
        ring = Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#ff5a1f",
            markeredgewidth=1,
            markersize=5,
        )
        if show_pibt_hotspots:
            figure.legend(
                [ring],
                [
                    f"top {100 * (1 - hotspot_quantile):.1f}% PIBT-change "
                    "density by event location"
                ],
                loc="lower left",
                bbox_to_anchor=(0.35, 0.043),
                frameon=False,
                handlelength=0.8,
                handletextpad=0.5,
                fontsize=8.5,
                labelcolor=note_color,
            )
        figure.text(
            0.35,
            0.043 if not show_pibt_hotspots else 0.028,
            f"Numbers report exact off-goal agent counts per square; "
            f"each square contains {bin_size}×{bin_size} original cells.",
            ha="left",
            fontsize=8.5,
            color=note_color,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
        figure.savefig(temporary, dpi=200, facecolor=background)
        finalize_figure_output(temporary, output)
        width = int(round(figure.get_figwidth() * 100))
        height = int(round(figure.get_figheight() * 100))
        plt.close(figure)
    return {
        "width": width,
        "height": height,
        "panels": panel_metadata,
        "active_scale_max": active_scale_max,
    }


def _format_compact_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _format_settlement(value: float) -> str:
    percentage = 100 * value
    if math.isclose(percentage, round(percentage), abs_tol=1e-8):
        return str(int(round(percentage)))
    return f"{percentage:.1f}"


def render_phase_traffic_figure(
    *,
    output: Path,
    phases: list[dict],
    bin_rows: int,
    bin_columns: int,
    bin_size: int,
    agents: int,
    seed: int | str,
    contour_quantiles: list[float],
    colormap: str = "Reds",
) -> dict:
    """Render five settlement phases as a compact paper figure."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    if not phases or len(phases) > 5:
        raise ValueError("phase-traffic figures require between one and five phases")
    positive_rates = np.concatenate(
        [phase["moving_visit_rate"][phase["moving_visit_rate"] > 0] for phase in phases]
    )
    if positive_rates.size:
        visit_scale_min = float(positive_rates.min())
        visit_scale_max = float(positive_rates.max())
    else:
        visit_scale_min, visit_scale_max = 1e-12, 1.0
    if visit_scale_max <= visit_scale_min:
        visit_scale_max = visit_scale_min * 10
    norm = LogNorm(vmin=visit_scale_min, vmax=visit_scale_max, clip=True)
    cmap = mpl.colormaps[colormap].copy()
    cmap.set_bad("#f2f2f2")
    contour_colors = ["#6baed6", "#08519c"]
    if len(contour_quantiles) != len(contour_colors):
        contour_colors = [
            mpl.colors.to_hex(mpl.colormaps["Blues"](value))
            for value in np.linspace(0.58, 0.90, len(contour_quantiles))
        ]
    panel_metadata = []

    with mpl.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "svg.image_inline": True,
            "pdf.fonttype": 42,
        }
    ):
        figure, axes_array = plt.subplots(
            2,
            3,
            figsize=(7.2, 5.05),
            facecolor="white",
            squeeze=False,
        )
        figure.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.94,
            bottom=0.035,
            wspace=0.18,
            hspace=0.28,
        )
        flat_axes = axes_array.reshape(-1)
        for index, phase in enumerate(phases):
            axis = flat_axes[index]
            rate_grid = phase["moving_visit_rate"].reshape(bin_rows, bin_columns)
            masked_rate = np.ma.masked_less_equal(rate_grid, 0)
            axis.imshow(
                masked_rate,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                origin="upper",
                rasterized=True,
            )

            pibt_rate = phase["pibt_location_rate"].reshape(
                bin_rows, bin_columns
            )
            smoothed_pibt_rate = smooth_spatial_grid(pibt_rate)
            positive_pibt = smoothed_pibt_rate[smoothed_pibt_rate > 0]
            contour_thresholds = []
            contour_levels = []
            contour_level_colors = []
            if positive_pibt.size:
                minimum = float(positive_pibt.min())
                maximum = float(positive_pibt.max())
                for quantile, color in zip(contour_quantiles, contour_colors):
                    threshold = float(np.quantile(positive_pibt, quantile))
                    contour_thresholds.append(threshold)
                    if minimum < threshold < maximum:
                        if not contour_levels or threshold > contour_levels[-1]:
                            contour_levels.append(threshold)
                            contour_level_colors.append(color)
                if contour_levels:
                    axis.contour(
                        smoothed_pibt_rate,
                        levels=contour_levels,
                        colors=contour_level_colors,
                        linewidths=np.linspace(0.55, 0.9, len(contour_levels)),
                        origin="upper",
                    )

            target_start = phase["target_start_isr"]
            target_end = phase["target_end_isr"]
            axis.set_title(
                f"({chr(ord('a') + index)}) "
                f"{_format_settlement(target_start)}–"
                f"{_format_settlement(target_end)}% settled",
                loc="left",
                y=1.065,
                fontsize=8.4,
                fontweight="bold",
                pad=0,
            )
            axis.text(
                0,
                1.018,
                f"steps {phase['previous_step']:,}–{phase['step']:,}  ·  "
                f"{_format_compact_count(phase['moving_visits_total'])} moves  ·  "
                f"{_format_compact_count(phase['pibt_changes_total'])} PIBT",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=5.9,
                color="#4d4d4d",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")
            for spine in axis.spines.values():
                spine.set_color("#202020")
                spine.set_linewidth(0.55)
            panel_metadata.append(
                {
                    "step_start": phase["previous_step"],
                    "step_end": phase["step"],
                    "target_start_isr": target_start,
                    "target_end_isr": target_end,
                    "actual_start_isr": phase["start_isr"],
                    "actual_end_isr": phase["end_isr"],
                    "moving_visits": phase["moving_visits_total"],
                    "pibt_changes": phase["pibt_changes_total"],
                    "pibt_changed_agents": phase["pibt_changed_agents"],
                    "pibt_contour_thresholds": contour_thresholds,
                }
            )

        legend_axis = flat_axes[5]
        legend_axis.set_axis_off()
        color_axis = legend_axis.inset_axes((0.08, 0.68, 0.84, 0.075))
        colorbar = figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=color_axis,
            orientation="horizontal",
        )
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(labelsize=5.8, length=2, pad=1)
        colorbar.ax.set_title(
            "Moving visits / free cell / step",
            loc="left",
            fontsize=7.2,
            fontweight="bold",
            pad=4,
        )
        contour_handles = [
            Line2D([], [], color=color, linewidth=width)
            for color, width in zip(
                contour_colors, np.linspace(0.8, 1.2, len(contour_colors))
            )
        ]
        legend_axis.legend(
            contour_handles,
            [
                f"top {100 * (1 - quantile):g}% PIBT-change density"
                for quantile in contour_quantiles
            ],
            loc="upper left",
            bbox_to_anchor=(0.06, 0.57),
            frameon=False,
            fontsize=6.5,
            handlelength=2.1,
            borderaxespad=0,
        )
        legend_axis.text(
            0.08,
            0.36,
            f"{agents:,} agents · seed {seed}\n"
            f"Each heatmap pixel aggregates {bin_size}×{bin_size} map cells.\n"
            "The red scale is shared across panels.\n"
            "Blue contours are phase-relative.\n"
            "PIBT events are located before the executed action.",
            ha="left",
            va="top",
            fontsize=6.3,
            color="#303030",
            linespacing=1.35,
        )
        for axis in flat_axes[len(phases) : 5]:
            axis.set_visible(False)

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
        figure.savefig(temporary, dpi=400, facecolor="white")
        finalize_figure_output(temporary, output)
        width = int(round(figure.get_figwidth() * 100))
        height = int(round(figure.get_figheight() * 100))
        plt.close(figure)
    return {
        "width": width,
        "height": height,
        "panels": panel_metadata,
        "moving_visit_rate_scale": [visit_scale_min, visit_scale_max],
    }


def render_pibt_regions_figure(
    *,
    output: Path,
    phases: list[dict],
    bin_rows: int,
    bin_columns: int,
    bin_size: int,
    free_cell_counts: np.ndarray,
    agents: int,
    seed: int | str,
    region_pool: int,
    hotspot_quantile: float,
    runtime_composition: dict | None = None,
    colormap: str = TRAFFIC_HEATMAP_NAME,
    phase_label_mode: str = "settlement",
    figure_title: str | None = None,
) -> dict:
    """Render movement traffic with exact PIBT shielding hotspot outlines."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Patch

    if not phases or len(phases) > 5:
        raise ValueError("PIBT-region figures require between one and five phases")
    if phase_label_mode not in {"settlement", "elapsed_time"}:
        raise ValueError(f"unknown phase label mode: {phase_label_mode}")
    pooled_rows = bin_rows // region_pool
    pooled_columns = bin_columns // region_pool
    region_size = bin_size * region_pool
    free_cell_counts = np.asarray(free_cell_counts, dtype=np.int64).reshape(-1)
    if free_cell_counts.size != bin_rows * bin_columns:
        raise ValueError(
            "free-cell counts must contain one value per spatial input bin"
        )
    pooled_free_cell_counts = pool_spatial_counts(
        free_cell_counts,
        bin_rows=bin_rows,
        bin_columns=bin_columns,
        factor=region_pool,
    ).astype(np.int64, copy=False)
    phase_spatial_data = []
    for phase in phases:
        interval_steps = int(phase["step"]) - int(phase["previous_step"])
        if interval_steps <= 0:
            raise ValueError("PIBT-region phases must contain at least one step")
        pooled_moving_counts = pool_spatial_counts(
            phase["moving_visit_counts"],
            bin_rows=bin_rows,
            bin_columns=bin_columns,
            factor=region_pool,
        ).astype(np.float64, copy=False)
        pooled_agent_presence_counts = pool_spatial_counts(
            phase["agent_presence_counts"],
            bin_rows=bin_rows,
            bin_columns=bin_columns,
            factor=region_pool,
        ).astype(np.float64, copy=False)
        pooled_pibt_counts = pool_spatial_counts(
            phase["pibt_location_counts"],
            bin_rows=bin_rows,
            bin_columns=bin_columns,
            factor=region_pool,
        ).astype(np.float64, copy=False)
        agent_density = np.divide(
            pooled_agent_presence_counts,
            pooled_free_cell_counts.astype(np.float64) * interval_steps,
            out=np.zeros_like(pooled_agent_presence_counts),
            where=pooled_free_cell_counts > 0,
        )
        phase_spatial_data.append(
            {
                "moving_counts": pooled_moving_counts,
                "agent_presence_counts": pooled_agent_presence_counts,
                "pibt_counts": pooled_pibt_counts,
                "agent_density": agent_density,
            }
        )

    positive_densities = np.concatenate(
        [
            spatial["agent_density"][spatial["agent_density"] > 0]
            for spatial in phase_spatial_data
        ]
    )
    if positive_densities.size:
        density_scale_min = float(np.quantile(positive_densities, 0.05))
        density_scale_max = float(positive_densities.max())
    else:
        density_scale_min, density_scale_max = 1e-6, 1e-5
    if density_scale_max <= density_scale_min:
        density_scale_max = density_scale_min * 10
    density_norm = LogNorm(
        vmin=density_scale_min,
        vmax=density_scale_max,
        clip=True,
    )
    cmap = resolve_colormap(mpl, colormap)
    cmap.set_bad("#edf3f4")
    panel_metadata = []
    elapsed_step_seconds = (
        np.cumsum(
            np.asarray(
                runtime_composition["per_step_wall_seconds"],
                dtype=np.float64,
            )
        )
        if runtime_composition
        and runtime_composition.get("per_step_wall_seconds")
        else None
    )

    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "svg.image_inline": True,
            "pdf.fonttype": 42,
        }
    ):
        figure = plt.figure(figsize=(7.2, 4.05), facecolor="white")
        if figure_title:
            figure.suptitle(figure_title, y=0.99, fontsize=8, fontweight="bold")
        outer_grid = figure.add_gridspec(
            5,
            1,
            height_ratios=(1.0, 0.005, 0.10, 0.145, 0.80),
            left=0.055,
            right=0.995,
            top=0.89 if figure_title else 0.925,
            bottom=0.15,
            hspace=0,
        )
        map_grid = outer_grid[0].subgridspec(1, 5, wspace=0.11)
        map_axes = [figure.add_subplot(map_grid[0, index]) for index in range(5)]
        for index, (phase, spatial) in enumerate(
            zip(phases, phase_spatial_data)
        ):
            axis = map_axes[index]
            pooled_moving_counts = spatial["moving_counts"]
            pooled_agent_presence_counts = spatial["agent_presence_counts"]
            pooled_pibt_counts = spatial["pibt_counts"]
            agent_density = spatial["agent_density"]
            axis.imshow(
                np.ma.masked_less_equal(agent_density, 0),
                cmap=cmap,
                norm=density_norm,
                interpolation="nearest",
                origin="upper",
                rasterized=True,
            )

            hotspot_threshold = None
            hotspot_regions = 0
            positive_pibt = pooled_pibt_counts[pooled_pibt_counts > 0]
            if positive_pibt.size:
                hotspot_threshold = float(
                    np.quantile(positive_pibt, hotspot_quantile)
                )
                hotspot_mask = pooled_pibt_counts >= hotspot_threshold
                hotspot_regions = int(hotspot_mask.sum())
                if hotspot_mask.any() and not hotspot_mask.all():
                    axis.contour(
                        hotspot_mask.astype(np.float64),
                        levels=[0.5],
                        colors=[PIBT_ACCENT_COLOR],
                        linewidths=1.05,
                        origin="upper",
                    )

            target_start = phase["target_start_isr"]
            target_end = phase["target_end_isr"]
            if phase_label_mode == "elapsed_time" and elapsed_step_seconds is not None:
                start_seconds = (
                    float(elapsed_step_seconds[int(phase["previous_step"]) - 1])
                    if phase["previous_step"]
                    else 0.0
                )
                end_seconds = float(elapsed_step_seconds[int(phase["step"]) - 1])
                panel_heading = (
                    f"steps {phase['previous_step']:,}–{phase['step']:,}"
                )
                panel_subtitle = (
                    f"{start_seconds / 60.0:.1f}–{end_seconds / 60.0:.1f} min"
                )
            else:
                start_seconds = None
                end_seconds = None
                panel_heading = (
                    f"{_format_settlement(target_start)}–"
                    f"{_format_settlement(target_end)}% settled"
                )
                panel_subtitle = (
                    f"steps {phase['previous_step']:,}–{phase['step']:,}"
                )
            axis.set_title(
                f"({chr(ord('a') + index)}) {panel_heading}",
                loc="left",
                y=1.13,
                fontsize=7.7,
                fontweight="bold",
                pad=0,
            )
            axis.text(
                0,
                1.018,
                panel_subtitle,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=6.25,
                color="#4d4d4d",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")
            for spine in axis.spines.values():
                spine.set_color("#202020")
                spine.set_linewidth(0.65)
            panel_metadata.append(
                {
                    "step_start": phase["previous_step"],
                    "step_end": phase["step"],
                    "target_start_isr": target_start,
                    "target_end_isr": target_end,
                    "actual_start_isr": phase["start_isr"],
                    "actual_end_isr": phase["end_isr"],
                    "elapsed_seconds_start": start_seconds,
                    "elapsed_seconds_end": end_seconds,
                    "moving_visits": phase["moving_visits_total"],
                    "pibt_changes": phase["pibt_changes_total"],
                    "pibt_changed_agents": phase["pibt_changed_agents"],
                    "pibt_hotspot_threshold": hotspot_threshold,
                    "pibt_hotspot_regions": hotspot_regions,
                    "agent_density_scale": [
                        density_scale_min,
                        density_scale_max,
                    ],
                    "moving_visits_by_region": pooled_moving_counts.astype(
                        int
                    ).tolist(),
                    "agent_presences": int(pooled_agent_presence_counts.sum()),
                    "agent_presence_by_region": pooled_agent_presence_counts.astype(
                        int
                    ).tolist(),
                    "agent_density_by_region": agent_density.tolist(),
                    "pibt_changes_by_region": pooled_pibt_counts.astype(int).tolist(),
                }
            )

        for axis in map_axes[len(phases) :]:
            axis.set_visible(False)

        heat_key_axis = figure.add_subplot(outer_grid[2])
        heat_key_axis.set_axis_off()
        gradient_axis = heat_key_axis.inset_axes((0.0, 0.56, 0.16, 0.30))
        gradient_axis.imshow(
            np.linspace(0, 1, 256).reshape(1, -1),
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
        )
        gradient_axis.set_xticks(
            [0, 255],
            labels=[f"{density_scale_min:.3f}", f"{density_scale_max:.3f}"],
        )
        gradient_axis.tick_params(axis="x", labelsize=6.2, length=0, pad=0.5)
        gradient_axis.set_yticks([])
        for spine in gradient_axis.spines.values():
            spine.set_visible(False)
        heat_key_axis.text(
            0.185,
            0.65,
            "Agent density  (agents / free cell)",
            transform=heat_key_axis.transAxes,
            ha="left",
            va="center",
            fontsize=6.4,
            fontweight="bold",
            color="#202020",
        )
        heat_key_axis.plot(
            [0.43, 0.465],
            [0.65, 0.65],
            color=PIBT_ACCENT_COLOR,
            linewidth=1.6,
            transform=heat_key_axis.transAxes,
            clip_on=False,
        )
        heat_key_axis.text(
            0.478,
            0.65,
            f"Top {100 * (1 - hotspot_quantile):g}% CS-PIBT changes",
            transform=heat_key_axis.transAxes,
            ha="left",
            va="center",
            fontsize=6.4,
            color="#303030",
        )
        heat_key_axis.text(
            0.995,
            0.65,
            f"Regions: {region_size}×{region_size} cells",
            transform=heat_key_axis.transAxes,
            ha="right",
            va="center",
            fontsize=6.4,
            color="#303030",
        )

        timing_grid = outer_grid[4].subgridspec(
            2,
            1,
            height_ratios=(0.16, 0.84),
            hspace=0.055,
        )
        prefill_axis = figure.add_subplot(timing_grid[0])
        timing_axis = figure.add_subplot(timing_grid[1], sharex=prefill_axis)
        phase_elapsed_seconds = None
        runtime_components_by_key = (
            {
                component["key"]: component
                for component in runtime_composition["components"]
            }
            if runtime_composition
            else {}
        )
        displayed_runtime_components = [
            runtime_components_by_key[key]
            for key in RUNTIME_FIGURE_ORDER
            if key in runtime_components_by_key
        ]
        has_timeline = bool(
            runtime_composition
            and runtime_composition.get("per_step_count")
            and runtime_composition.get("per_step_wall_seconds")
            and all(
                "per_step_ms" in component
                for component in displayed_runtime_components
            )
        )
        if not has_timeline:
            prefill_axis.set_axis_off()
            timing_axis.set_axis_off()
            timing_axis.text(
                0,
                0.55,
                "(f) Per-step runtime composition unavailable",
                fontsize=8.4,
                fontweight="bold",
                ha="left",
                va="center",
                transform=timing_axis.transAxes,
            )
        else:
            smoothing_window = 32
            component_series_seconds = np.asarray(
                [
                    component["per_step_ms"]
                    for component in displayed_runtime_components
                ],
                dtype=np.float64,
            ) / 1000.0
            wall_step_seconds = np.asarray(
                runtime_composition["per_step_wall_seconds"],
                dtype=np.float64,
            )
            elapsed_seconds = np.cumsum(wall_step_seconds)
            step_count = int(runtime_composition["per_step_count"])
            environment_steps = np.arange(1, step_count + 1, dtype=np.float64)
            if step_count > 1:
                steady_smoothed = np.asarray(
                    [
                        centered_rolling_mean(values[1:], smoothing_window)
                        for values in component_series_seconds
                    ]
                )
            else:
                steady_smoothed = np.empty(
                    (component_series_seconds.shape[0], 0),
                    dtype=np.float64,
                )
            plotted_series = np.concatenate(
                [component_series_seconds[:, :1], steady_smoothed],
                axis=1,
            )
            colors = [
                component["color"]
                for component in displayed_runtime_components
            ]
            for axis in (prefill_axis, timing_axis):
                axis.stackplot(
                    environment_steps,
                    *plotted_series,
                    colors=colors,
                    linewidth=0.2,
                    edgecolor="white",
                    alpha=0.94,
                )
            steady_total_seconds = steady_smoothed.sum(axis=0)
            y_limit = 1.05 * float(steady_total_seconds.max(initial=1.0))
            prefill_seconds = runtime_composition.get("prefill_step_seconds")
            prefill_stack_seconds = float(plotted_series[:, 0].sum())
            prefill_reference_seconds = (
                float(prefill_seconds)
                if prefill_seconds is not None
                else prefill_stack_seconds
            )
            prefill_lower = max(
                y_limit * 1.2,
                min(prefill_reference_seconds, prefill_stack_seconds) - 4.0,
            )
            prefill_upper = (
                max(prefill_reference_seconds, prefill_stack_seconds) + 1.5
            )
            prefill_axis.set_ylim(prefill_lower, prefill_upper)
            timing_axis.set_xlim(-0.004 * step_count, step_count)
            timing_axis.set_ylim(0, y_limit)
            timing_axis.set_ylabel(
                "Time per step (s)",
                fontsize=7.2,
                labelpad=2,
            )
            timing_axis.set_xlabel("environment step", fontsize=7.2, labelpad=1)
            phase_elapsed_seconds = [
                float(elapsed_seconds[int(phase["step"]) - 1])
                for phase in phases
            ]
            phase_steps = [0] + [int(phase["step"]) for phase in phases]
            timing_axis.set_xticks(phase_steps)
            timing_axis.set_xticklabels(
                [f"{step:,}" for step in phase_steps],
                fontsize=6.7,
            )
            timing_axis.get_xticklabels()[0].set_ha("left")
            timing_axis.get_xticklabels()[-1].set_ha("right")
            timing_axis.tick_params(axis="y", labelsize=6.7, length=2, pad=1)
            timing_axis.tick_params(axis="x", length=2, pad=1)
            timing_axis.grid(axis="y", color="#ffffff", linewidth=0.45, alpha=0.7)
            prefill_axis.set_yticks(
                [prefill_reference_seconds],
                labels=[f"{prefill_reference_seconds:.0f} s"],
            )
            prefill_axis.tick_params(
                axis="y",
                labelsize=6.3,
                length=2,
                pad=1,
                colors="#7b542b",
            )
            prefill_axis.tick_params(
                axis="x",
                bottom=False,
                labelbottom=False,
            )
            for spine in ("top", "right"):
                timing_axis.spines[spine].set_visible(False)
                prefill_axis.spines[spine].set_visible(False)
            prefill_axis.spines["bottom"].set_visible(False)
            timing_axis.spines["left"].set_linewidth(0.5)
            timing_axis.spines["bottom"].set_linewidth(0.5)
            prefill_axis.spines["left"].set_linewidth(0.5)
            break_size = 0.006
            prefill_axis.plot(
                (-break_size, break_size),
                (-break_size, break_size),
                transform=prefill_axis.transAxes,
                color="#202020",
                linewidth=0.55,
                clip_on=False,
            )
            timing_axis.plot(
                (-break_size, break_size),
                (1 - break_size, 1 + break_size),
                transform=timing_axis.transAxes,
                color="#202020",
                linewidth=0.55,
                clip_on=False,
            )
            for boundary_step in phase_steps[1:-1]:
                timing_axis.axvline(
                    boundary_step,
                    color="#3f3f46",
                    linewidth=0.45,
                    linestyle=(0, (2, 2)),
                    alpha=0.65,
                )
            observation_index = next(
                index
                for index, component in enumerate(displayed_runtime_components)
                if component["key"] == "observation"
            )
            observation_top = float(
                plotted_series[: observation_index + 1, 0].sum()
            )
            observation_color = runtime_components_by_key["observation"]["color"]
            timing_axis.vlines(
                1,
                0,
                y_limit,
                color=observation_color,
                linewidth=1.15,
                zorder=5,
            )
            prefill_axis.vlines(
                1,
                prefill_lower,
                min(observation_top, prefill_upper),
                color=observation_color,
                linewidth=1.15,
                zorder=5,
            )
            prefill_axis.scatter(
                [1],
                [observation_top],
                s=14,
                facecolor=observation_color,
                edgecolor="white",
                linewidth=0.4,
                clip_on=False,
                zorder=6,
            )
            prefill_axis.set_title(
                f"(f) Runtime per environment step (all {agents:,} agents)",
                loc="left",
                fontsize=8.3,
                fontweight="bold",
                pad=4,
            )
            handles = []
            labels = []
            for component in displayed_runtime_components:
                handles.append(Patch(facecolor=component["color"], edgecolor="none"))
                episode_seconds = float(component["episode_seconds"])
                displayed_seconds = (
                    f"{episode_seconds:,.0f} s"
                    if episode_seconds >= 10
                    else f"{episode_seconds:.1f} s"
                )
                short_label = component["short_label"].replace(
                    "Logits all-gather", "All-gather"
                )
                suffix = (
                    " (incl. prefill)"
                    if component["key"] == "observation"
                    and prefill_seconds is not None
                    else ""
                )
                labels.append(f"{short_label}  {displayed_seconds}{suffix}")
            timing_axis.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.29),
                ncol=len(handles),
                frameon=False,
                fontsize=6.05,
                handlelength=0.9,
                handletextpad=0.35,
                columnspacing=0.95,
                borderaxespad=0,
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
        figure.savefig(temporary, dpi=400, facecolor="white")
        finalize_figure_output(temporary, output)
        width = int(round(figure.get_figwidth() * 100))
        height = int(round(figure.get_figheight() * 100))
        plt.close(figure)
    return {
        "width": width,
        "height": height,
        "panels": panel_metadata,
        "region_pool": region_pool,
        "region_size": region_size,
        "region_shape": [pooled_rows, pooled_columns],
        "free_cells_by_region": pooled_free_cell_counts.tolist(),
        "agent_density_scale": [density_scale_min, density_scale_max],
        "phase_label_mode": phase_label_mode,
        "runtime_composition": runtime_composition_metadata(runtime_composition),
        "runtime_timeline": (
            {
                "steps": int(runtime_composition["per_step_count"]),
                "smoothing": "raw_step_1_then_centered_rolling_mean",
                "smoothing_window_steps": 32,
                "phase_boundaries": [int(phase["step"]) for phase in phases],
                "phase_boundaries_elapsed_seconds": phase_elapsed_seconds,
                "x_axis": "environment_step",
                "y_axis": "max_rank_gpu_stage_latency_seconds_per_step",
                "first_step_prefill_shown_in_broken_y_axis": True,
                "omitted_components": sorted(RUNTIME_FIGURE_OMITTED_KEYS),
            }
            if has_timeline
            else None
        ),
    }


def analyze_and_render(args: argparse.Namespace) -> dict:
    import torch

    started = time.perf_counter()
    phase_mode = args.mode in {"phase-traffic", "pibt-regions"}
    metadata_path = args.metadata.resolve()
    metadata = json.loads(metadata_path.read_text())
    run_result_arg = getattr(args, "run_result", None)
    run_result_path = run_result_arg.resolve() if run_result_arg else None
    runtime_composition = (
        runtime_composition_from_result(json.loads(run_result_path.read_text()))
        if run_result_path is not None
        else None
    )
    steps, agents = map(int, metadata["shape"])
    checkpoints = (
        sorted(set(map(int, args.checkpoints))) if args.checkpoints else None
    )
    milestones = (
        sorted(set(map(float, args.settlement_milestones)))
        if args.settlement_milestones
        else None
    )
    if checkpoints and (checkpoints[0] < 0 or checkpoints[-1] > steps):
        raise ValueError(f"checkpoints must be in [0, {steps}]")
    if phase_mode and checkpoints and checkpoints[0] == 0:
        raise ValueError("phase-traffic checkpoints must be greater than zero")

    master_path = (
        args.master.resolve()
        if args.master
        else resolve_recorded_path(metadata["master"], metadata_path)
    )
    master = torch.load(master_path, map_location="cpu", weights_only=False)
    grid = np.asarray(torch.as_tensor(master["grid"], dtype=torch.uint8).numpy())
    starts = np.asarray(master["starts"][:agents].numpy(), dtype=np.int32).copy()
    goals = np.asarray(master["goals"][:agents].numpy(), dtype=np.int32).copy()
    if grid.shape[0] % args.bin_size or grid.shape[1] % args.bin_size:
        raise ValueError("grid dimensions must be divisible by --bin-size")
    bin_rows = grid.shape[0] // args.bin_size
    bin_columns = grid.shape[1] // args.bin_size
    bins = bin_rows * bin_columns
    free_cell_counts = (
        (grid == 0)
        .reshape(bin_rows, args.bin_size, bin_columns, args.bin_size)
        .sum(axis=(1, 3), dtype=np.int64)
        .reshape(-1)
    )

    def position_bins(points: np.ndarray) -> np.ndarray:
        return (
            (points[:, 0] // args.bin_size) * bin_columns
            + points[:, 1] // args.bin_size
        ).astype(np.int64, copy=False)

    goal_bin_indices = (
        (goals[:, 0] // args.bin_size) * bin_columns
        + goals[:, 1] // args.bin_size
    ).astype(np.int64, copy=False)
    goal_counts = np.bincount(goal_bin_indices, minlength=bins).astype(np.int64)

    action_path = (
        args.actions.resolve()
        if args.actions
        else resolve_recorded_path(metadata["actions"], metadata_path)
    )
    change_metadata = metadata.get("pibt_changes")
    change_path = args.pibt_changes.resolve() if args.pibt_changes else None
    if change_path is None and change_metadata:
        change_path = resolve_recorded_path(change_metadata["path"], metadata_path)
    packed_bytes = int(change_metadata["packed_bytes_per_step"]) if change_metadata else 0
    positions = starts
    interval_changes = np.zeros(agents, dtype=np.uint32)
    pibt_location_counts = np.zeros(bins, dtype=np.int64)
    moving_visit_counts = np.zeros(bins, dtype=np.int64)
    agent_presence_counts = np.zeros(bins, dtype=np.int64)
    current_position_bin_indices = position_bins(positions)
    current_position_counts = np.bincount(
        current_position_bin_indices, minlength=bins
    ).astype(np.int64, copy=False)
    snapshots: list[dict] = []
    action_digest = hashlib.sha256()
    change_digest = hashlib.sha256()
    cursor = 0
    previous_checkpoint = 0
    phase_start_isr = float(np.all(positions == goals, axis=1).mean())
    target_start_isr = 0.0 if milestones else phase_start_isr
    next_progress = args.progress_every if args.progress_every else None

    def snapshot(
        step: int,
        *,
        target_isr: float | None = None,
        final: bool = False,
    ) -> None:
        nonlocal previous_checkpoint
        captured = capture_snapshot(
            step=step,
            previous_step=previous_checkpoint,
            positions=positions,
            goals=goals,
            goal_bin_indices=goal_bin_indices,
            goal_counts=goal_counts,
            interval_changes=interval_changes,
            current_bin_indices=position_bins(positions),
            free_cell_counts=free_cell_counts,
            pibt_location_counts=pibt_location_counts,
            target_isr=target_isr,
            final=final,
        )
        snapshots.append(captured)
        print(
            f"checkpoint={step:,}/{steps:,} isr={captured['isr']:.6f} "
            f"unresolved={captured['unresolved_agents']:,} "
            f"interval_pibt_changes={captured['pibt_changes_total']:,}",
            flush=True,
        )
        previous_checkpoint = step
        interval_changes.fill(0)
        pibt_location_counts.fill(0)

    def phase_snapshot(
        step: int,
        *,
        end_isr: float,
        target_end_isr: float,
        final: bool = False,
    ) -> None:
        nonlocal previous_checkpoint, phase_start_isr, target_start_isr
        captured = capture_phase_traffic(
            step=step,
            previous_step=previous_checkpoint,
            start_isr=phase_start_isr,
            end_isr=end_isr,
            target_start_isr=target_start_isr,
            target_end_isr=target_end_isr,
            moving_visit_counts=moving_visit_counts,
            agent_presence_counts=agent_presence_counts,
            pibt_location_counts=pibt_location_counts,
            free_cell_counts=free_cell_counts,
            pibt_changed_agents=int(np.count_nonzero(interval_changes)),
            final=final,
        )
        snapshots.append(captured)
        print(
            f"phase={100 * target_start_isr:g}-{100 * target_end_isr:g}% "
            f"steps={previous_checkpoint:,}-{step:,}/{steps:,} "
            f"isr={end_isr:.6f} "
            f"moving_visits={captured['moving_visits_total']:,} "
            f"pibt_changes={captured['pibt_changes_total']:,}",
            flush=True,
        )
        previous_checkpoint = step
        phase_start_isr = end_isr
        target_start_isr = target_end_isr
        interval_changes.fill(0)
        pibt_location_counts.fill(0)
        moving_visit_counts.fill(0)
        agent_presence_counts.fill(0)

    checkpoint_set = set(checkpoints or ())
    if not phase_mode and (milestones or 0 in checkpoint_set):
        snapshot(0)
    milestone_index = 0
    with gzip.open(action_path, "rb") as action_stream:
        change_context = gzip.open(change_path, "rb") if change_path else None
        try:
            while cursor < steps:
                count = min(args.chunk_steps, steps - cursor)
                action_raw = read_exact(action_stream, count * agents)
                action_digest.update(action_raw)
                actions = np.frombuffer(action_raw, dtype=np.uint8).reshape(
                    count, agents
                )
                if int(actions.max(initial=0)) > 4:
                    raise ValueError(
                        f"invalid action in steps {cursor}:{cursor + count}"
                    )
                unpacked = None
                if change_context is not None:
                    change_raw = read_exact(change_context, count * packed_bytes)
                    change_digest.update(change_raw)
                    packed = np.frombuffer(change_raw, dtype=np.uint8).reshape(
                        count, packed_bytes
                    )
                    unpacked = np.unpackbits(
                        packed,
                        axis=1,
                        count=agents,
                        bitorder=change_metadata.get("bitorder", "little"),
                    )
                for offset in range(count):
                    step_actions = actions[offset]
                    if unpacked is not None:
                        accumulate_pibt_locations(
                            positions=positions,
                            changed=unpacked[offset],
                            interval_changes=interval_changes,
                            location_counts=pibt_location_counts,
                            bin_size=args.bin_size,
                            bin_columns=bin_columns,
                        )
                    if phase_mode:
                        advance_phase_positions(
                            positions=positions,
                            actions=step_actions,
                            position_bin_indices=current_position_bin_indices,
                            position_counts=current_position_counts,
                            moving_visit_counts=moving_visit_counts,
                            agent_presence_counts=agent_presence_counts,
                            bin_size=args.bin_size,
                            bin_columns=bin_columns,
                        )
                    else:
                        positions += MOVES[step_actions]
                    cursor += 1

                    current_isr = None
                    if milestones:
                        current_isr = float(
                            np.all(positions == goals, axis=1).mean()
                        )
                        if (
                            milestone_index < len(milestones)
                            and current_isr >= milestones[milestone_index]
                        ):
                            target_isr = milestones[milestone_index]
                            while (
                                milestone_index < len(milestones)
                                and current_isr >= milestones[milestone_index]
                            ):
                                milestone_index += 1
                            if phase_mode:
                                phase_snapshot(
                                    cursor,
                                    end_isr=current_isr,
                                    target_end_isr=target_isr,
                                )
                            else:
                                snapshot(cursor, target_isr=target_isr)
                    elif cursor in checkpoint_set:
                        current_isr = float(
                            np.all(positions == goals, axis=1).mean()
                        )
                        if phase_mode:
                            phase_snapshot(
                                cursor,
                                end_isr=current_isr,
                                target_end_isr=current_isr,
                                final=cursor == steps,
                            )
                        else:
                            snapshot(cursor, final=cursor == steps)

                    if next_progress is not None and cursor >= next_progress:
                        if current_isr is None:
                            current_isr = float(
                                np.all(positions == goals, axis=1).mean()
                            )
                        print(
                            f"replay={cursor:,}/{steps:,} "
                            f"isr={current_isr:.6f} "
                            f"elapsed_s={time.perf_counter() - started:.1f}",
                            flush=True,
                        )
                        while next_progress <= cursor:
                            next_progress += args.progress_every

            if not snapshots or snapshots[-1]["step"] != steps:
                final_isr = float(np.all(positions == goals, axis=1).mean())
                if phase_mode:
                    phase_snapshot(
                        steps,
                        end_isr=final_isr,
                        target_end_isr=1.0 if math.isclose(final_isr, 1.0) else final_isr,
                        final=True,
                    )
                else:
                    snapshot(steps, final=True)
            else:
                snapshots[-1]["final"] = True
            if action_stream.read(1):
                raise ValueError("action stream has trailing uncompressed bytes")
            if change_context is not None and change_context.read(1):
                raise ValueError("PIBT-change stream has trailing uncompressed bytes")
        finally:
            if change_context is not None:
                change_context.close()

    action_sha = action_digest.hexdigest()
    expected_action_sha = metadata.get("uncompressed_sha256")
    if expected_action_sha and action_sha != expected_action_sha:
        raise ValueError(f"action SHA256 mismatch: {action_sha} != {expected_action_sha}")
    change_sha = change_digest.hexdigest() if change_path else None
    expected_change_sha = change_metadata.get("uncompressed_sha256") if change_metadata else None
    if expected_change_sha and change_sha != expected_change_sha:
        raise ValueError(f"PIBT SHA256 mismatch: {change_sha} != {expected_change_sha}")

    seed = master.get("metadata", {}).get("root_seed", "unknown")

    def render(output: Path) -> dict:
        if args.mode == "phase-traffic":
            return render_phase_traffic_figure(
                output=output,
                phases=snapshots,
                bin_rows=bin_rows,
                bin_columns=bin_columns,
                bin_size=args.bin_size,
                agents=agents,
                seed=seed,
                contour_quantiles=args.pibt_contour_quantiles,
                colormap=args.colormap,
            )
        if args.mode == "pibt-regions":
            return render_pibt_regions_figure(
                output=output,
                phases=snapshots,
                bin_rows=bin_rows,
                bin_columns=bin_columns,
                bin_size=args.bin_size,
                free_cell_counts=free_cell_counts,
                agents=agents,
                seed=seed,
                region_pool=args.region_pool,
                hotspot_quantile=args.region_hotspot_quantile,
                runtime_composition=runtime_composition,
                colormap=args.colormap,
                phase_label_mode=(
                    "elapsed_time" if checkpoints else "settlement"
                ),
            )
        return render_progress_svg(
            output=output,
            snapshots=snapshots,
            bin_rows=bin_rows,
            bin_columns=bin_columns,
            bin_size=args.bin_size,
            agents=agents,
            seed=seed,
            hotspot_quantile=args.hotspot_quantile,
            annotate_counts=args.annotate_counts,
            show_pibt_hotspots=not args.hide_pibt_hotspots,
            colormap=args.colormap,
        )

    render_metadata = render(args.output)
    additional_figures = []
    for additional_output in args.also_output:
        additional_metadata = render(additional_output)
        additional_figures.append(
            {
                "path": str(additional_output),
                "width": additional_metadata["width"],
                "height": additional_metadata["height"],
            }
        )
    summary = {
        "format": (
            "dmm-phase-traffic-map-v1"
            if args.mode == "phase-traffic"
            else (
                "dmm-pibt-regions-map-v1"
                if args.mode == "pibt-regions"
                else "dmm-progress-map-v2"
            )
        ),
        "source": {
            "metadata": str(metadata_path),
            "master": str(master_path),
            "actions": str(action_path),
            "pibt_changes": str(change_path) if change_path else None,
            "run_result": str(run_result_path) if run_result_path else None,
            "verified_sha256": {
                "actions": action_sha,
                "pibt_changes": change_sha,
            },
        },
        "agents": agents,
        "episode_steps": steps,
        "grid_shape": list(grid.shape),
        "bin_size": args.bin_size,
        "bin_shape": [bin_rows, bin_columns],
        "selection": (
            {"mode": "fixed_steps", "checkpoints": checkpoints}
            if checkpoints
            else {"mode": "settlement_milestones", "targets": milestones}
        ),
        "spatial_encoding": (
            {
                "color": "moving_action_destination_visits_per_free_cell_per_step",
                "color_scale": "shared_log",
                "color_scale_limits": render_metadata[
                    "moving_visit_rate_scale"
                ],
                "overlay": "smoothed_pibt_change_event_location_density_contours",
                "pibt_contour_quantiles": args.pibt_contour_quantiles,
                "pibt_contour_smoothing_kernel": [1, 4, 6, 4, 1],
                "colormap": args.colormap,
            }
            if args.mode == "phase-traffic"
            else (
                {
                    "color": "mean_agent_presence_per_free_cell_by_region",
                    "color_scale": "shared_log",
                    "color_scale_limits": render_metadata[
                        "agent_density_scale"
                    ],
                    "normalization": "traversable_cells_times_interval_steps",
                    "display_region_pool": args.region_pool,
                    "display_region_size": render_metadata["region_size"],
                    "display_region_shape": render_metadata["region_shape"],
                    "free_cells_by_region": render_metadata[
                        "free_cells_by_region"
                    ],
                    "overlay": "top_pibt_change_region_boundary",
                    "region_hotspot_quantile": args.region_hotspot_quantile,
                    "colormap": args.colormap,
                }
                if args.mode == "pibt-regions"
                else {
                    "color": "off_goal_agent_count",
                    "color_scale": "shared_log",
                    "color_scale_max": render_metadata["active_scale_max"],
                    "overlay": "pibt_change_event_location_density",
                    "hotspot_quantile": args.hotspot_quantile,
                    "annotated_counts": args.annotate_counts,
                    "show_pibt_hotspots": not args.hide_pibt_hotspots,
                    "colormap": args.colormap,
                }
            )
        ),
        "svg": {
            "path": str(args.output),
            "width": render_metadata["width"],
            "height": render_metadata["height"],
        },
        "additional_figures": additional_figures,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if args.mode == "pibt-regions":
        summary["runtime_composition"] = render_metadata["runtime_composition"]
        summary["runtime_timeline"] = render_metadata["runtime_timeline"]
    if phase_mode:
        summary["phases"] = []
        for phase, panel in zip(snapshots, render_metadata["panels"]):
            phase_record = dict(panel)
            phase_record.update(
                {
                    "interval_steps": phase["interval_steps"],
                    "agent_presences": phase["agent_presences_total"],
                    "moving_visits_by_tile": phase["moving_visit_counts"]
                    .reshape(bin_rows, bin_columns)
                    .astype(int)
                    .tolist(),
                    "moving_visit_rate_by_tile": phase["moving_visit_rate"]
                    .reshape(bin_rows, bin_columns)
                    .tolist(),
                    "agent_presence_by_tile": phase["agent_presence_counts"]
                    .reshape(bin_rows, bin_columns)
                    .astype(int)
                    .tolist(),
                    "agent_density_by_tile": phase["agent_density"]
                    .reshape(bin_rows, bin_columns)
                    .tolist(),
                    "pibt_changes_by_tile": phase["pibt_location_counts"]
                    .reshape(bin_rows, bin_columns)
                    .astype(int)
                    .tolist(),
                    "pibt_change_rate_by_tile": phase["pibt_location_rate"]
                    .reshape(bin_rows, bin_columns)
                    .tolist(),
                    "final": bool(phase["final"]),
                }
            )
            summary["phases"].append(phase_record)
    else:
        summary["checkpoints"] = render_metadata["panels"]
    summary_path = args.summary or args.output.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("off-goal-snapshots", "phase-traffic", "pibt-regions"),
        default="off-goal-snapshots",
        help="spatial statistic and publication layout to render",
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--run-result",
        type=Path,
        help="raw run-result JSON supplying steady-state timing for pibt-regions",
    )
    parser.add_argument(
        "--master",
        type=Path,
        help="override the scenario-master path recorded in relocated metadata",
    )
    parser.add_argument(
        "--actions",
        type=Path,
        help="override the action-trace path recorded in relocated metadata",
    )
    parser.add_argument(
        "--pibt-changes",
        type=Path,
        help="override the PIBT-change path recorded in relocated metadata",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--also-output",
        type=Path,
        nargs="*",
        default=[],
        help="also save the same Matplotlib figure (for example as PDF or PNG)",
    )
    parser.add_argument("--summary", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        help="fixed trajectory steps to render",
    )
    selection.add_argument(
        "--settlement-milestones",
        type=float,
        nargs="+",
        help="render the first step reaching each ISR fraction",
    )
    parser.add_argument("--bin-size", type=int, default=16)
    parser.add_argument(
        "--annotate-counts",
        action="store_true",
        help="print the exact off-goal agent count inside every spatial square",
    )
    parser.add_argument(
        "--hide-pibt-hotspots",
        action="store_true",
        help="omit PIBT hotspot rings from the figure while retaining totals",
    )
    parser.add_argument(
        "--colormap",
        help="Matplotlib colormap; pibt-regions defaults to a warm sequential scale",
    )
    parser.add_argument("--chunk-steps", type=int, default=16)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=256,
        help="print replay progress every this many steps; use 0 to disable",
    )
    parser.add_argument("--hotspot-quantile", type=float, default=0.995)
    parser.add_argument(
        "--pibt-contour-quantiles",
        type=float,
        nargs=2,
        default=[0.95, 0.99],
        metavar=("OUTER", "INNER"),
        help="phase-relative PIBT-density contour quantiles",
    )
    parser.add_argument(
        "--region-pool",
        type=int,
        default=4,
        help="number of fine heatmap bins pooled along each region dimension",
    )
    parser.add_argument(
        "--region-hotspot-quantile",
        type=float,
        default=0.99,
        help="phase-relative PIBT-region quantile outlined in black",
    )
    args = parser.parse_args()
    if args.colormap is None:
        args.colormap = TRAFFIC_HEATMAP_NAME if args.mode == "pibt-regions" else "Reds"
    if args.checkpoints is None and args.settlement_milestones is None:
        args.settlement_milestones = [0.1, 0.5, 0.9, 0.99]
    if (
        args.bin_size <= 0
        or args.chunk_steps <= 0
        or args.progress_every < 0
        or args.region_pool <= 0
    ):
        parser.error(
            "--bin-size, --chunk-steps, and --region-pool must be positive and "
            "--progress-every must be non-negative"
        )
    if not 0.0 <= args.hotspot_quantile <= 1.0:
        parser.error("--hotspot-quantile must be in [0, 1]")
    if not (
        0.0 < args.pibt_contour_quantiles[0]
        < args.pibt_contour_quantiles[1]
        < 1.0
    ):
        parser.error(
            "--pibt-contour-quantiles must be strictly increasing values in (0, 1)"
        )
    if not 0.0 < args.region_hotspot_quantile < 1.0:
        parser.error("--region-hotspot-quantile must be in (0, 1)")
    if args.settlement_milestones and not all(
        0.0 < value < 1.0 for value in args.settlement_milestones
    ):
        parser.error("--settlement-milestones values must be in (0, 1)")
    if (
        args.mode in {"phase-traffic", "pibt-regions"}
        and args.settlement_milestones
        and len(set(args.settlement_milestones)) > 4
    ):
        parser.error("phase-based modes support at most four milestones")
    return args


def main() -> None:
    summary = analyze_and_render(parse_args())
    print(
        f"rendered {summary['svg']['path']} in {summary['elapsed_seconds']:.1f}s",
        flush=True,
    )
    if "phases" in summary:
        for phase in summary["phases"]:
            print(
                f"phase={100 * phase['target_start_isr']:g}-"
                f"{100 * phase['target_end_isr']:g}% "
                f"steps={phase['step_start']:,}-{phase['step_end']:,} "
                f"moving_visits={phase['moving_visits']:,} "
                f"pibt_changes={phase['pibt_changes']:,}",
                flush=True,
            )
    else:
        for checkpoint in summary["checkpoints"]:
            print(
                f"step={checkpoint['step']:,} isr={checkpoint['isr']:.6f} "
                f"unresolved={checkpoint['unresolved_agents']:,} "
                f"interval_pibt_changes={checkpoint['pibt_changes']:,}",
                flush=True,
            )


if __name__ == "__main__":
    main()
