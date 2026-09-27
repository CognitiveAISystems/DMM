"""Validation of the fixed MovingAI-1600 task manifest."""

from __future__ import annotations

import csv
from pathlib import Path
import re

EXPECTED_TASKS = 1600
EXPECTED_MAPS = 32
SAFE_KEY = re.compile(r"[A-Za-z0-9._-]+\Z")
REQUIRED_COLUMNS = {
    "base_key", "map_name", "map_path", "scenario_path", "num_agents",
    "seed", "max_steps",
}


def load_manifest(
    path: Path,
    *,
    expected_tasks: int = EXPECTED_TASKS,
    expected_maps: int = EXPECTED_MAPS,
) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("MovingAI manifest is missing required columns")
        rows = list(reader)
    if len(rows) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} tasks, found {len(rows)}")
    keys = [row["base_key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("MovingAI task keys must be unique")
    if len({row["map_name"] for row in rows}) != expected_maps:
        raise ValueError(f"expected {expected_maps} distinct maps")
    for row in rows:
        key = row["base_key"]
        if not SAFE_KEY.fullmatch(key) or key in {".", ".."}:
            raise ValueError(f"unsafe task key: {key!r}")
        if int(row["seed"]) != 0 or int(row["max_steps"]) != 5000:
            raise ValueError(f"unexpected protocol for {key}")
        if not 0 < int(row["num_agents"]) <= 8192:
            raise ValueError(f"invalid agent count for {key}")
        for column in ("map_path", "scenario_path"):
            source = Path(row[column])
            if source.is_absolute() or ".." in source.parts:
                raise ValueError(f"unsafe {column} for {key}")
    return rows
