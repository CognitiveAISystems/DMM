"""Render the data-driven panels of the million-agent paper figure."""

from __future__ import annotations

import argparse
import json
import lzma
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plotting import (  # noqa: E402
    TRAFFIC_HEATMAP_NAME,
    render_pibt_regions_figure,
    runtime_composition_from_result,
)


def load_json(path: Path) -> dict:
    with lzma.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figure" / "seed303_reproduced.pdf",
    )
    args = parser.parse_args()

    summary = load_json(ROOT / "data" / "seed303_spatial_fine.json.xz")
    result = load_json(
        ROOT / "data" / "dmm_08m" / "seed303_n1048576_4gpu.json.xz"
    )
    if summary["episode_steps"] != result["steps"] or result["steps"] != 16_345:
        raise ValueError("spatial summary and run result disagree")

    phases = []
    for saved in summary["phases"]:
        phases.append(
            {
                "step": saved["step_end"],
                "previous_step": saved["step_start"],
                "start_isr": saved["actual_start_isr"],
                "end_isr": saved["actual_end_isr"],
                "target_start_isr": saved["target_start_isr"],
                "target_end_isr": saved["target_end_isr"],
                "moving_visit_counts": np.asarray(
                    saved["moving_visits_by_region"], dtype=np.int64
                ).reshape(-1),
                "moving_visits_total": saved["moving_visits"],
                "agent_presence_counts": np.asarray(
                    saved["agent_presence_by_region"], dtype=np.int64
                ).reshape(-1),
                "pibt_location_counts": np.asarray(
                    saved["pibt_changes_by_region"], dtype=np.int64
                ).reshape(-1),
                "pibt_changes_total": saved["pibt_changes"],
                "pibt_changed_agents": saved["pibt_changed_agents"],
            }
        )

    encoding = summary["spatial_encoding"]
    region_rows, region_columns = encoding["display_region_shape"]
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    render_pibt_regions_figure(
        output=output,
        phases=phases,
        bin_rows=region_rows,
        bin_columns=region_columns,
        bin_size=encoding["display_region_size"],
        free_cell_counts=np.asarray(
            encoding["free_cells_by_region"], dtype=np.int64
        ).reshape(-1),
        agents=summary["agents"],
        seed=303,
        region_pool=1,
        hotspot_quantile=encoding["region_hotspot_quantile"],
        runtime_composition=runtime_composition_from_result(result),
        colormap=TRAFFIC_HEATMAP_NAME,
        phase_label_mode="elapsed_time",
    )
    print(output)


if __name__ == "__main__":
    main()
