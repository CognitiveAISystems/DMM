"""Recompute the paper's scalability table from the 32 run records."""

from __future__ import annotations

import csv
import json
import lzma
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIZES = (1_048_576, 524_288, 262_144, 131_072)
SEEDS = (101, 202, 303, 404)
DENSITY_PCT = {1_048_576: 27.86, 524_288: 13.93, 262_144: 6.97, 131_072: 3.48}
METHODS = (("DMM-MICPO-0.8M", "dmm_08m"), ("GPU-PIBT", "gpu_pibt"))


def load_record(method_dir: str, agents: int, seed: int) -> dict:
    path = (
        ROOT
        / "data"
        / method_dir
        / f"seed{seed}_n{agents}_4gpu.json.xz"
    )
    with lzma.open(path, "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    if record["agents"] != agents or record["horizon"] != 32_768:
        raise ValueError(f"unexpected scenario metadata: {path}")
    if record["status"] != "completed":
        raise ValueError(f"incomplete record: {path}")
    return record


def main() -> None:
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        (
            "method",
            "agents",
            "density_pct",
            "on_goal_pct",
            "mean_steps",
            "min_steps",
            "max_steps",
            "mean_total_wall_min",
        )
    )
    for agents in SIZES:
        for method, method_dir in METHODS:
            records = [load_record(method_dir, agents, seed) for seed in SEEDS]
            writer.writerow(
                (
                    method,
                    agents,
                    f"{DENSITY_PCT[agents]:.2f}",
                    f"{100 * statistics.mean(r['isr'] for r in records):.2f}",
                    f"{statistics.mean(r['steps'] for r in records):.0f}",
                    min(r["steps"] for r in records),
                    max(r["steps"] for r in records),
                    f"{statistics.mean(r['wall_seconds'] for r in records) / 60:.1f}",
                )
            )


if __name__ == "__main__":
    main()
