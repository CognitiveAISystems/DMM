"""Build one POGEMA family from the frozen instance archive."""

from __future__ import annotations

import csv
from pathlib import Path

from evaluation.assets import EVALUATION_ROOT, materialize
from evaluation.instances import read_instance, source_file
from evaluation.pogema.manifest import (
    DEFAULT_MAP_NAMES, FAMILY_HORIZONS, FAMILY_TOTAL_COUNTS,
)

FAMILIES = tuple(FAMILY_TOTAL_COUNTS)


def load_family(family: str, cache_root: Path) -> list:
    """Every episode of one family, in manifest order."""
    from pogema_gpu.tasks import Task

    if family not in FAMILY_TOTAL_COUNTS:
        raise ValueError(f"unknown POGEMA family: {family}")
    instance_root = materialize("pogema", cache_root)
    maps: dict[Path, tuple[str, ...]] = {}
    tasks = []
    with (EVALUATION_ROOT / "pogema" / "manifest.tsv").open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["suite"] != family:
                continue
            name = row["map_name"] or DEFAULT_MAP_NAMES[family]
            seed, agents = int(row["seed"]), int(row["num_agents"])
            obstacles, starts, goals = read_instance(
                source_file(instance_root, row["map_path"]),
                source_file(instance_root, row["scen_path"]), agents, maps,
            )
            tasks.append(Task(
                f"{family}/{name}/seed{seed}/n{agents}", obstacles, starts, goals,
                policy_seed=0, horizon=FAMILY_HORIZONS[family],
                provenance={"base_key": row["base_key"], "suite": family,
                            "map_name": name, "generation_seed": seed,
                            "named_map": bool(row["map_name"])},
            ))
    # The seven known-infeasible POGEMA episodes are kept: they fail identically for
    # every configuration and keep all depths on the same 128-episode task set.
    if len(tasks) != FAMILY_TOTAL_COUNTS[family]:
        raise ValueError(f"{family} must contain {FAMILY_TOTAL_COUNTS[family]} episodes")
    return tasks
