#!/usr/bin/env python3
"""Build the MovingAI solution-quality and runtime figure from raw JSON rows."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import reportlab
from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TASKS = 1600

# The figure is placed at single-column width (about 3.3 in), so text is
# sized to remain about 6 pt after scaling.
PAGE_W = 7.20 * 72
PAGE_H = 540
LEFT = 64
RIGHT = 6
TEXT_SIZE = 13
# Offset from a row's vertical center to the text baseline.
BASELINE_SHIFT = TEXT_SIZE * 0.35

SOC_PLOT_Y = 314
SOC_PLOT_H = 180
RUNTIME_PLOT_Y = 53
RUNTIME_PLOT_H = 225

TIME_LIMIT = 600.0
# The proposed methods are separated from the baselines by a vertical rule.
PROPOSED_PREFIX = "DMM-MICPO-"

FONT_REGULAR = "FigureSans"
FONT_BOLD = "FigureSans-Bold"


def register_fonts() -> None:
    """Register the redistributable fonts bundled with ReportLab."""
    font_dir = Path(reportlab.__file__).resolve().parent / "fonts"
    pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(font_dir / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(font_dir / "VeraBd.ttf")))


@dataclass(frozen=True)
class MethodSpec:
    key: str
    filename: str
    algorithm: str
    detail: str
    color: str
    display_label: str | None = None


METHODS = (
    MethodSpec("lglacam", "LG-LaCAM.json", "LG-LaCAM", "600 s", "#8C564B"),
    MethodSpec("lagat", "LaGAT.json", "LaGAT", "600 s", "#CC79A7"),
    MethodSpec("lns2", "MAPF-LNS2.json", "MAPF-LNS2", "600 s", "#6B7280"),
    MethodSpec(
        "hmagat", "HMAGAT.json", "HMAGAT+RSE", "5000 steps", "#2E7D32",
        display_label="HMAGAT",
    ),
    MethodSpec(
        "dmm08m",
        "DMM-MICPO-08M.json",
        "DMM-MICPO-08M",
        "5000 steps",
        "#5B9BD5",
        display_label="DMM-MICPO-0.8M",
    ),
    MethodSpec(
        "dmm",
        "DMM-MICPO-3M.json",
        "DMM-MICPO-3M",
        "5000 steps",
        "#174A7C",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PACKAGE_ROOT / "data",
        help="directory containing the six per-method JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "figure" / "mapf-soc-runtime-movingai1600.pdf",
        help="output PDF path",
    )
    return parser.parse_args()


def quantile(values: list[float], q: float) -> float:
    """Return the linearly interpolated quantile used in the paper figure."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def log_kde(values: list[float], count: int = 72) -> list[tuple[float, float]]:
    """Estimate a Gaussian KDE in log10 space for the runtime violin."""
    logs = [math.log10(value) for value in values if value > 0]
    if len(logs) < 2:
        raise ValueError("At least two positive runtimes are required")
    spread = statistics.stdev(logs)
    bandwidth = max(1.0e-6, 1.06 * spread * len(logs) ** (-0.2))
    start = min(logs) - 0.55
    stop = max(logs) + 0.55
    normalizer = bandwidth * math.sqrt(2.0 * math.pi) * len(logs)
    density = []
    for index in range(count):
        x = start + (stop - start) * index / (count - 1)
        weight = math.fsum(
            math.exp(-0.5 * ((x - sample) / bandwidth) ** 2)
            for sample in logs
        ) / normalizer
        density.append((10.0**x, weight))
    return density


def read_records(path: Path, spec: MethodSpec) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list")
    if len(payload) != EXPECTED_TASKS:
        raise ValueError(
            f"{path}: expected {EXPECTED_TASKS} records, got {len(payload)}"
        )

    records = {}
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} is not an object")
        if record.get("algorithm") != spec.algorithm:
            raise ValueError(
                f"{path}: record {index} has algorithm {record.get('algorithm')!r}"
            )
        metrics = record.get("metrics")
        environment = record.get("env_grid_search")
        if not isinstance(metrics, dict) or not isinstance(environment, dict):
            raise ValueError(f"{path}: record {index} has an invalid envelope")

        task_key = environment.get("task_key")
        if not isinstance(task_key, str) or not task_key:
            raise ValueError(f"{path}: record {index} has no task_key")
        if task_key in records:
            raise ValueError(f"{path}: duplicate task_key {task_key}")

        solved = metrics.get("CSR") == 1.0
        soc = metrics.get("SoC")
        runtime = metrics.get("runtime")
        if solved and (not isinstance(soc, (int, float)) or soc <= 0):
            raise ValueError(f"{path}: solved task {task_key} has no positive SoC")
        if not solved and soc is not None:
            raise ValueError(f"{path}: unsolved task {task_key} has a SoC")
        if not isinstance(runtime, (int, float)) or not math.isfinite(runtime):
            raise ValueError(f"{path}: task {task_key} has an invalid runtime")
        if runtime <= 0:
            raise ValueError(f"{path}: task {task_key} has a non-positive runtime")
        records[task_key] = record
    return records


def prepare_plot_data(data_dir: Path) -> tuple[list[dict[str, Any]], int]:
    records_by_method = {
        spec.key: read_records(data_dir / spec.filename, spec) for spec in METHODS
    }
    task_sets = [set(records) for records in records_by_method.values()]
    reference_tasks = task_sets[0]
    for spec, task_set in zip(METHODS[1:], task_sets[1:]):
        if task_set != reference_tasks:
            raise ValueError(
                f"{spec.filename}: task set differs from {METHODS[0].filename}"
            )

    virtual_best = {}
    for task_key in reference_tasks:
        solved_costs = [
            records_by_method[spec.key][task_key]["metrics"]["SoC"]
            for spec in METHODS
            if records_by_method[spec.key][task_key]["metrics"]["CSR"] == 1.0
        ]
        if solved_costs:
            virtual_best[task_key] = min(solved_costs)
    if len(virtual_best) != EXPECTED_TASKS:
        raise ValueError(
            f"Virtual best covers {len(virtual_best)} of {EXPECTED_TASKS} tasks"
        )

    plot_data = []
    for spec in METHODS:
        records = records_by_method[spec.key]
        solved_rows = {
            task_key: record
            for task_key, record in records.items()
            if record["metrics"]["CSR"] == 1.0
        }
        ratios = [
            100.0 * record["metrics"]["SoC"] / virtual_best[task_key]
            for task_key, record in solved_rows.items()
        ]
        runtimes = [float(record["metrics"]["runtime"]) for record in records.values()]
        points = [
            (
                float(record["metrics"]["runtime"]),
                record["metrics"]["CSR"] == 1.0,
            )
            for task_key, record in sorted(records.items())
        ]
        wins = sum(
            record["metrics"]["SoC"] == virtual_best[task_key]
            for task_key, record in solved_rows.items()
        )
        plot_data.append(
            {
                "key": spec.key,
                "label": spec.display_label or spec.algorithm,
                "detail": spec.detail,
                "color": spec.color,
                "solved": len(solved_rows),
                "wins": wins,
                "mean": statistics.fmean(ratios),
                "median": quantile(ratios, 0.50),
                "q75": quantile(ratios, 0.75),
                "q90": quantile(ratios, 0.90),
                "q95": quantile(ratios, 0.95),
                "runtime_median": statistics.median(runtimes),
                "runtime_density": log_kde(runtimes),
                "runtime_points": points,
            }
        )
    return plot_data, len(virtual_best)


def draw_centered(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    font: str = FONT_REGULAR,
    size: float = TEXT_SIZE,
) -> None:
    pdf.setFont(font, size)
    pdf.drawString(x - stringWidth(text, font, size) / 2, y, text)


def draw_centered_outlined(
    pdf: canvas.Canvas, text: str, x: float, y: float
) -> None:
    text_x = x - stringWidth(text, FONT_BOLD, TEXT_SIZE) / 2
    pdf.saveState()
    pdf.setFont(FONT_BOLD, TEXT_SIZE)
    pdf.setFillColor(white)
    for angle in range(0, 360, 30):
        radians = math.radians(angle)
        pdf.drawString(
            text_x + math.cos(radians) * 1.4,
            y + math.sin(radians) * 1.4,
            text,
        )
    pdf.setFillColor(black)
    pdf.drawString(text_x, y, text)
    pdf.restoreState()


def draw_diamond(
    pdf: canvas.Canvas, x: float, y: float, radius: float = 3.1
) -> None:
    path = pdf.beginPath()
    path.moveTo(x, y + radius)
    path.lineTo(x + radius, y)
    path.lineTo(x, y - radius)
    path.lineTo(x - radius, y)
    path.close()
    pdf.drawPath(path, fill=1, stroke=1)


def linear_y(
    value: float, y_min: float, y_max: float, plot_y: float, plot_h: float
) -> float:
    return plot_y + (value - y_min) / (y_max - y_min) * plot_h


def log_y(
    value: float, y_min: float, y_max: float, plot_y: float, plot_h: float
) -> float:
    lower = math.log10(y_min)
    upper = math.log10(y_max)
    return plot_y + (math.log10(value) - lower) / (upper - lower) * plot_h


def draw_panel_title(
    pdf: canvas.Canvas, text: str, plot_x: float, row_y: float
) -> None:
    pdf.setFillColor(black)
    pdf.setFont(FONT_BOLD, TEXT_SIZE)
    pdf.drawString(plot_x + 7, row_y - BASELINE_SHIFT, text)


def draw_soc_legend(
    pdf: canvas.Canvas, plot_x: float, plot_w: float, legend_y: float
) -> None:
    gap = 14
    pad = 4
    bands_text = "q95 / q90 / q75"
    bands_w = stringWidth(bands_text, FONT_REGULAR, TEXT_SIZE)
    median_w = stringWidth("median", FONT_REGULAR, TEXT_SIZE)
    mean_w = stringWidth("mean", FONT_REGULAR, TEXT_SIZE)
    legend_w = 8 + pad + bands_w + gap + 16 + pad + median_w + gap + 8 + pad + mean_w
    legend_x = plot_x + plot_w - legend_w - 7
    text_y = legend_y - BASELINE_SHIFT

    pdf.saveState()
    pdf.setFillColor(HexColor("#777777"))
    pdf.rect(legend_x, legend_y - 4, 8, 8, fill=1, stroke=0)
    pdf.setFillColor(black)
    pdf.setFont(FONT_REGULAR, TEXT_SIZE)
    pdf.drawString(legend_x + 8 + pad, text_y, bands_text)

    median_x = legend_x + 8 + pad + bands_w + gap
    pdf.setStrokeColor(black)
    pdf.setLineWidth(2.2)
    pdf.line(median_x, legend_y, median_x + 16, legend_y)
    pdf.drawString(median_x + 16 + pad, text_y, "median")

    mean_x = median_x + 16 + pad + median_w + gap
    pdf.setFillColor(white)
    pdf.setStrokeColor(black)
    pdf.setLineWidth(1.2)
    draw_diamond(pdf, mean_x + 4, legend_y, 3.8)
    pdf.setFillColor(black)
    pdf.drawString(mean_x + 8 + pad, text_y, "mean")
    pdf.restoreState()


def draw_runtime_legend(
    pdf: canvas.Canvas, plot_x: float, plot_w: float, legend_y: float
) -> None:
    gap = 14
    pad = 4
    solved_w = stringWidth("solved", FONT_REGULAR, TEXT_SIZE)
    unsolved_w = stringWidth("unsolved", FONT_REGULAR, TEXT_SIZE)
    median_w = stringWidth("median", FONT_REGULAR, TEXT_SIZE)
    legend_w = 6 + pad + solved_w + gap + 6 + pad + unsolved_w + gap + 16 + pad + median_w
    legend_x = plot_x + plot_w - legend_w - 7
    text_y = legend_y - BASELINE_SHIFT

    pdf.setFillColor(HexColor("#333333"))
    pdf.circle(legend_x + 3, legend_y, 3, fill=1, stroke=0)
    pdf.setFont(FONT_REGULAR, TEXT_SIZE)
    pdf.drawString(legend_x + 6 + pad, text_y, "solved")

    cross_x = legend_x + 6 + pad + solved_w + gap + 3
    pdf.setStrokeColor(HexColor("#333333"))
    pdf.setLineWidth(1.2)
    pdf.line(cross_x - 3, legend_y - 3, cross_x + 3, legend_y + 3)
    pdf.line(cross_x - 3, legend_y + 3, cross_x + 3, legend_y - 3)
    pdf.drawString(cross_x + 3 + pad, text_y, "unsolved")

    median_x = cross_x + 3 + pad + unsolved_w + gap
    pdf.setStrokeColor(black)
    pdf.setLineWidth(2.2)
    pdf.line(median_x, legend_y, median_x + 16, legend_y)
    pdf.drawString(median_x + 16 + pad, text_y, "median")


def draw_top_summary(
    pdf: canvas.Canvas,
    algorithms: list[dict[str, Any]],
    centers: list[float],
    reference_coverage: int,
) -> None:
    solved_y = PAGE_H - 18
    wins_y = PAGE_H - 36
    pdf.setFillColor(black)
    pdf.setFont(FONT_BOLD, TEXT_SIZE)
    pdf.drawRightString(LEFT - 5, solved_y, "Solved")
    pdf.drawRightString(LEFT - 5, wins_y, "Wins")
    for center, algorithm in zip(centers, algorithms):
        draw_centered(pdf, str(algorithm["solved"]), center, solved_y)
        draw_centered(pdf, str(algorithm["wins"]), center, wins_y, FONT_BOLD)


def draw_separator(
    pdf: canvas.Canvas,
    algorithms: list[dict[str, Any]],
    centers: list[float],
    plot_y: float,
    plot_h: float,
) -> None:
    """Draw a vertical rule between the baselines and the proposed methods."""
    first = next(
        index
        for index, algorithm in enumerate(algorithms)
        if algorithm["label"].startswith(PROPOSED_PREFIX)
    )
    x = (centers[first - 1] + centers[first]) / 2
    pdf.saveState()
    pdf.setStrokeColor(HexColor("#4A4A4A"))
    pdf.setStrokeAlpha(0.45)
    pdf.setLineWidth(0.9)
    pdf.setDash(4, 3)
    # Stop below the title/legend row so the rule does not cut through it.
    pdf.line(x, plot_y, x, plot_y + plot_h - 26)
    pdf.restoreState()


def draw_soc_panel(
    pdf: canvas.Canvas,
    algorithms: list[dict[str, Any]],
    centers: list[float],
) -> None:
    plot_x = LEFT
    plot_y = SOC_PLOT_Y
    plot_w = PAGE_W - LEFT - RIGHT
    plot_h = SOC_PLOT_H
    plot_top = plot_y + plot_h
    y_min = 100.0
    y_max = 178.0
    y_ticks = list(range(100, 161, 10))

    for tick in y_ticks:
        y = linear_y(tick, y_min, y_max, plot_y, plot_h)
        if tick == 100:
            pdf.setStrokeColor(HexColor("#737373"))
            pdf.setLineWidth(0.9)
            pdf.setDash(3, 2)
        else:
            pdf.setStrokeColor(HexColor("#D9D9D9"))
            pdf.setLineWidth(0.55)
            pdf.setDash()
        pdf.line(plot_x, y, plot_x + plot_w, y)
    pdf.setDash()

    pdf.setStrokeColor(HexColor("#4A4A4A"))
    pdf.setLineWidth(0.7)
    pdf.rect(plot_x, plot_y, plot_w, plot_h, fill=0, stroke=1)
    draw_separator(pdf, algorithms, centers, plot_y, plot_h)
    pdf.setFillColor(black)
    pdf.setFont(FONT_REGULAR, TEXT_SIZE)
    for tick in y_ticks:
        y = linear_y(tick, y_min, y_max, plot_y, plot_h)
        pdf.drawRightString(plot_x - 5, y - BASELINE_SHIFT, str(tick))

    pdf.saveState()
    pdf.translate(13, plot_y + plot_h / 2)
    pdf.rotate(90)
    draw_centered(pdf, "SoC vs. virtual best (%)", 0, 0, FONT_BOLD)
    pdf.restoreState()

    band = plot_w / len(algorithms)
    baseline_y = linear_y(100.0, y_min, y_max, plot_y, plot_h)
    for algorithm, center in zip(algorithms, centers):
        color = HexColor(algorithm["color"])
        q75 = float(algorithm["q75"])
        q90 = float(algorithm["q90"])
        q95 = float(algorithm["q95"])
        median = float(algorithm["median"])
        mean = float(algorithm["mean"])

        y75 = linear_y(q75, y_min, y_max, plot_y, plot_h)
        y90 = linear_y(q90, y_min, y_max, plot_y, plot_h)
        y95 = linear_y(q95, y_min, y_max, plot_y, plot_h)
        y50 = linear_y(median, y_min, y_max, plot_y, plot_h)
        mean_y = linear_y(mean, y_min, y_max, plot_y, plot_h)

        box_half = min(18.0, band * 0.24)
        middle_half = box_half * 0.72
        outer_half = box_half * 0.44

        pdf.saveState()
        pdf.setFillColor(color)
        pdf.setStrokeColor(color)
        pdf.setLineWidth(0.8)
        pdf.setFillAlpha(0.11)
        pdf.rect(
            center - outer_half,
            baseline_y,
            outer_half * 2,
            max(1.0, y95 - baseline_y),
            fill=1,
            stroke=0,
        )
        pdf.setFillAlpha(0.19)
        pdf.rect(
            center - middle_half,
            baseline_y,
            middle_half * 2,
            max(1.0, y90 - baseline_y),
            fill=1,
            stroke=0,
        )
        pdf.setFillAlpha(0.34)
        pdf.rect(
            center - box_half,
            baseline_y,
            box_half * 2,
            max(1.0, y75 - baseline_y),
            fill=1,
            stroke=1,
        )
        pdf.setFillAlpha(1)
        pdf.setStrokeColor(black)
        pdf.setLineWidth(1.8)
        pdf.line(center - box_half, y50, center + box_half, y50)
        pdf.setFillColor(white)
        pdf.setStrokeColor(black)
        pdf.setLineWidth(1.0)
        draw_diamond(pdf, center, mean_y)
        pdf.restoreState()

        # Anchor the mean value directly above its diamond marker.
        draw_centered_outlined(pdf, f"{mean:.1f}%", center, mean_y + 7)

    draw_panel_title(pdf, "(a) Solution quality", plot_x, plot_top - 13)
    draw_soc_legend(pdf, plot_x, plot_w, plot_top - 13)


def draw_runtime_panel(
    pdf: canvas.Canvas,
    algorithms: list[dict[str, Any]],
    centers: list[float],
) -> None:
    plot_x = LEFT
    plot_y = RUNTIME_PLOT_Y
    plot_w = PAGE_W - LEFT - RIGHT
    plot_h = RUNTIME_PLOT_H
    plot_top = plot_y + plot_h
    y_min = 2.0e-4
    y_max = 5.0e4
    y_ticks = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    pdf.setStrokeColor(HexColor("#D9D9D9"))
    pdf.setLineWidth(0.55)
    for tick in y_ticks:
        y = log_y(tick, y_min, y_max, plot_y, plot_h)
        pdf.line(plot_x, y, plot_x + plot_w, y)

    pdf.setStrokeColor(HexColor("#4A4A4A"))
    pdf.setLineWidth(0.7)
    pdf.rect(plot_x, plot_y, plot_w, plot_h, fill=0, stroke=1)
    draw_separator(pdf, algorithms, centers, plot_y, plot_h)
    pdf.setFillColor(black)
    pdf.setFont(FONT_REGULAR, TEXT_SIZE)
    for tick in y_ticks:
        y = log_y(tick, y_min, y_max, plot_y, plot_h)
        pdf.drawRightString(plot_x - 5, y - BASELINE_SHIFT, f"{tick:g}")

    pdf.saveState()
    pdf.translate(13, plot_y + plot_h / 2)
    pdf.rotate(90)
    draw_centered(pdf, "Runtime (s, log scale)", 0, 0, FONT_BOLD)
    pdf.restoreState()

    band = plot_w / len(algorithms)
    hit_seed = 43758.5453
    for algorithm_index, (algorithm, center) in enumerate(
        zip(algorithms, centers)
    ):
        color = HexColor(algorithm["color"])
        density = [
            (float(value), float(weight))
            for value, weight in algorithm["runtime_density"]
            if y_min <= float(value) <= y_max
        ]
        max_density = max(weight for _, weight in density)
        half_width = min(27.0, band * 0.36)

        left_points = [
            (
                center - half_width * weight / max_density,
                log_y(value, y_min, y_max, plot_y, plot_h),
            )
            for value, weight in density
        ]
        right_points = [
            (
                center + half_width * weight / max_density,
                log_y(value, y_min, y_max, plot_y, plot_h),
            )
            for value, weight in reversed(density)
        ]
        violin = pdf.beginPath()
        violin.moveTo(*left_points[0])
        for x, y in left_points[1:]:
            violin.lineTo(x, y)
        for x, y in right_points:
            violin.lineTo(x, y)
        violin.close()

        # Draw observations first so the violin remains visually dominant.
        pdf.saveState()
        pdf.setFillColor(color)
        pdf.setStrokeColor(color)
        pdf.setFillAlpha(0.21)
        pdf.setStrokeAlpha(0.38)
        for point_index, (runtime, solved) in enumerate(
            algorithm["runtime_points"]
        ):
            runtime = float(runtime)
            if runtime < y_min or runtime > y_max:
                continue
            jitter = (
                math.sin(
                    (point_index + 1) * 12.9898
                    + (algorithm_index + 11) * 78.233
                )
                * hit_seed
            ) % 1.0
            x = center + (jitter * 2 - 1) * half_width * 0.82
            y = log_y(runtime, y_min, y_max, plot_y, plot_h)
            if solved:
                pdf.circle(x, y, 1.45, fill=1, stroke=0)
            else:
                size = 2.15
                pdf.setLineWidth(1.05)
                pdf.line(x - size, y - size, x + size, y + size)
                pdf.line(x - size, y + size, x + size, y - size)
        pdf.restoreState()

        pdf.saveState()
        pdf.setFillColor(color)
        pdf.setStrokeColor(color)
        pdf.setLineWidth(1.15)
        pdf.setFillAlpha(0.17)
        pdf.setStrokeAlpha(0.85)
        pdf.drawPath(violin, fill=1, stroke=1)
        pdf.restoreState()

        median = float(algorithm["runtime_median"])
        median_y = log_y(median, y_min, y_max, plot_y, plot_h)
        pdf.setStrokeColor(black)
        pdf.setLineWidth(1.8)
        pdf.line(
            center - half_width * 0.78,
            median_y,
            center + half_width * 0.78,
            median_y,
        )
        if median < 1:
            median_text = f"{median:.3f} s"
        elif median < 100:
            median_text = f"{median:.2f} s"
        else:
            median_text = f"{median:.1f} s"
        label_y = min(plot_top - 30, median_y + 6)
        draw_centered_outlined(pdf, median_text, center, label_y)

    # Dashed time limit across the columns of the time-limited solvers.
    limited = [
        index
        for index, algorithm in enumerate(algorithms)
        if algorithm["detail"] == f"{TIME_LIMIT:g} s"
    ]
    limit_y = log_y(TIME_LIMIT, y_min, y_max, plot_y, plot_h)
    limit_right = (centers[limited[-1]] + centers[limited[-1] + 1]) / 2
    pdf.saveState()
    pdf.setStrokeColor(HexColor("#333333"))
    pdf.setLineWidth(1.1)
    pdf.setDash(4, 2.5)
    pdf.line(plot_x, limit_y, limit_right, limit_y)
    pdf.restoreState()
    limit_text = f"{TIME_LIMIT:g} s limit"
    draw_centered_outlined(
        pdf,
        limit_text,
        plot_x + 5 + stringWidth(limit_text, FONT_BOLD, TEXT_SIZE) / 2,
        limit_y + 5,
    )

    draw_panel_title(pdf, "(b) Solver runtime", plot_x, plot_top - 13)
    draw_runtime_legend(pdf, plot_x, plot_w, plot_top - 13)


def draw_bracket(
    pdf: canvas.Canvas,
    text: str,
    x0: float,
    x1: float,
    y: float,
    text_y: float,
    tick: float = 4,
) -> None:
    """Draw a bracket with end ticks of height ``tick`` (negative points down)."""
    pdf.setStrokeColor(HexColor("#4A4A4A"))
    pdf.setLineWidth(0.9)
    path = pdf.beginPath()
    path.moveTo(x0, y + tick)
    path.lineTo(x0, y)
    path.lineTo(x1, y)
    path.lineTo(x1, y + tick)
    pdf.drawPath(path, fill=0, stroke=1)
    pdf.setFillColor(black)
    draw_centered(pdf, text, (x0 + x1) / 2, text_y, FONT_BOLD)


def draw_shared_labels(
    pdf: canvas.Canvas,
    algorithms: list[dict[str, Any]],
    centers: list[float],
) -> None:
    # Group labels keep column labels short enough for single-column width.
    group_prefix = "DMM-MICPO-"
    half = (PAGE_W - LEFT - RIGHT) / len(algorithms) * 0.44
    pdf.setFillColor(black)
    for algorithm, center in zip(algorithms, centers):
        # Grouped methods show only their suffix below the group bracket.
        label = algorithm["label"].removeprefix(group_prefix)
        draw_centered(pdf, label, center, 35, FONT_BOLD)

    grouped = [
        index
        for index, algorithm in enumerate(algorithms)
        if algorithm["label"].startswith(group_prefix)
    ]
    # The group label sits inside the runtime panel, where these columns have
    # no observations near the lower axis limit.
    first = algorithms[grouped[0]]["label"].removeprefix(group_prefix)
    last = algorithms[grouped[-1]]["label"].removeprefix(group_prefix)
    draw_bracket(
        pdf,
        group_prefix.rstrip("-"),
        centers[grouped[0]] - stringWidth(first, FONT_BOLD, TEXT_SIZE) / 2 - 6,
        centers[grouped[-1]] + stringWidth(last, FONT_BOLD, TEXT_SIZE) / 2 + 6,
        RUNTIME_PLOT_Y + 12,
        RUNTIME_PLOT_Y + 16,
        tick=-4,
    )

    budgets: dict[str, list[int]] = {}
    for index, algorithm in enumerate(algorithms):
        budgets.setdefault(algorithm["detail"], []).append(index)
    for detail, indices in budgets.items():
        draw_bracket(
            pdf, detail, centers[indices[0]] - half, centers[indices[-1]] + half, 26, 12
        )


def draw_pdf(
    algorithms: list[dict[str, Any]],
    reference_coverage: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(
        str(output_path),
        pagesize=(PAGE_W, PAGE_H),
        pageCompression=1,
        initialFontName=FONT_REGULAR,
        initialFontSize=TEXT_SIZE,
        initialLeading=TEXT_SIZE * 1.2,
    )
    pdf.setTitle("MovingAI-1600 solution quality and solver runtime")
    pdf.setAuthor("DMM authors")
    pdf.setFillColor(white)
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    plot_w = PAGE_W - LEFT - RIGHT
    band = plot_w / len(algorithms)
    centers = [LEFT + band * (index + 0.5) for index in range(len(algorithms))]
    # Nudge columns apart where adjacent labels would otherwise touch.
    centers[1] -= 4.0
    centers[2] -= 5.0
    centers[3] += 5.0
    centers[-2] -= 3.5

    draw_top_summary(pdf, algorithms, centers, reference_coverage)
    draw_soc_panel(pdf, algorithms, centers)
    draw_runtime_panel(pdf, algorithms, centers)
    draw_shared_labels(pdf, algorithms, centers)

    pdf.showPage()
    pdf.save()


def print_summary(algorithms: list[dict[str, Any]]) -> None:
    print("algorithm              solved  wins  SoC mean/median  runtime median")
    for algorithm in algorithms:
        print(
            f"{algorithm['label']:<22} "
            f"{algorithm['solved']:>4}  "
            f"{algorithm['wins']:>4}  "
            f"{algorithm['mean']:>6.1f}/{algorithm['median']:<6.1f}  "
            f"{algorithm['runtime_median']:>9.3f} s"
        )


def main() -> None:
    arguments = parse_args()
    register_fonts()
    algorithms, reference_coverage = prepare_plot_data(arguments.data_dir)
    draw_pdf(algorithms, reference_coverage, arguments.output)
    print_summary(algorithms)
    print(arguments.output)


if __name__ == "__main__":
    main()
