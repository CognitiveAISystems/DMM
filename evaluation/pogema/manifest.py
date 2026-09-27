"""Select the exact 3,193 feasible POGEMA episodes from 3,200 frozen layouts."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
import re

from evaluation.instances import read_instance, source_file


FAMILY_COUNTS = {
    "01-random": 1020,
    "02-mazes": 896,
    "03-warehouse": 768,
    "04-movingai": 509,
}
FAMILY_TOTAL_COUNTS = {
    "01-random": 1024,
    "02-mazes": 896,
    "03-warehouse": 768,
    "04-movingai": 512,
}
FAMILY_HORIZONS = {
    "01-random": 128, "02-mazes": 128,
    "03-warehouse": 128, "04-movingai": 256,
}
DEFAULT_MAP_NAMES = {"03-warehouse": "wfi_warehouse"}
EXCLUDED_INFEASIBLE = frozenset({
    "01-random_0775_N80", "01-random_0903_N96",
    "01-random_0909_N96", "01-random_0914_N96",
    "04-movingai_0144_N128", "04-movingai_0272_N192",
    "04-movingai_0400_N256",
})
SAFE_KEY = re.compile(r"[A-Za-z0-9._-]+\Z")


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"base_key", "suite", "map_name", "seed", "num_agents",
                    "max_steps", "map_path", "scen_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("POGEMA manifest lacks required columns")
        rows = list(reader)
    if len(rows) != 3200 or Counter(row["suite"] for row in rows) != FAMILY_TOTAL_COUNTS:
        raise ValueError("POGEMA source manifest must contain all 3,200 episodes")
    keys = [row["base_key"] for row in rows]
    if len(set(keys)) != 3200:
        raise ValueError("POGEMA source task keys are not unique")
    for row in rows:
        key, family = row["base_key"], row["suite"]
        if not SAFE_KEY.fullmatch(key) or not key.startswith(family + "_"):
            raise ValueError(f"unsafe POGEMA task key: {key!r}")
        if int(row["max_steps"]) != FAMILY_HORIZONS[family]:
            raise ValueError(f"POGEMA horizon mismatch: {key}")
        if row["map_path"] != f"instances/{key}/instance.map" or row["scen_path"] != f"instances/{key}/instance.scen":
            raise ValueError(f"POGEMA instance path mismatch: {key}")
        if not 0 < int(row["num_agents"]) <= 8192:
            raise ValueError(f"invalid POGEMA agent count: {key}")
    return rows


def _exclusions(path: Path) -> set[str]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["base_key", "reason"]:
            raise ValueError("invalid POGEMA exclusion columns")
        rows = list(reader)
    keys = [row["base_key"] for row in rows]
    if (len(rows) != 7 or set(keys) != EXCLUDED_INFEASIBLE
            or any(row["reason"] != "known_infeasible" for row in rows)):
        raise ValueError("POGEMA must exclude exactly the seven known infeasible episodes")
    return set(keys)


def load_benchmark(dataset_root: Path, *, instance_root: Path | None = None):
    """Return the feasible cohort in frozen manifest order."""
    from pogema_gpu.tasks import Task

    dataset_root = Path(dataset_root).resolve()
    instance_root = Path(instance_root).resolve() if instance_root is not None else dataset_root
    manifest_path = dataset_root / "manifest.tsv"
    exclusions_path = dataset_root / "excluded-infeasible.tsv"
    sources = {"manifest": manifest_path.name, "instances": "instances.tar.gz",
               "exclusions": exclusions_path.name}
    rows = _manifest_rows(manifest_path)
    excluded = _exclusions(exclusions_path)
    by_family = {family: set() for family in FAMILY_COUNTS}
    selected_rows = []
    for row in rows:
        family, key = row["suite"], row["base_key"]
        if key in excluded:
            continue
        name = row["map_name"] or DEFAULT_MAP_NAMES.get(family)
        if not isinstance(name, str) or not SAFE_KEY.fullmatch(name):
            raise ValueError(f"unsafe POGEMA map name: {key}")
        identity = (name, int(row["seed"]), int(row["num_agents"]))
        if identity in by_family[family]:
            raise ValueError(f"duplicate POGEMA episode identity: {key}")
        by_family[family].add(identity)
        selected_rows.append((row, name, identity))
    if {family: len(items) for family, items in by_family.items()} != FAMILY_COUNTS:
        raise ValueError("POGEMA feasible cohort has incorrect family counts")
    selected = []
    maps: dict[Path, tuple[str, ...]] = {}
    for row, name, identity in selected_rows:
        family, key = row["suite"], row["base_key"]
        obstacles, starts, goals = read_instance(
            source_file(instance_root, row["map_path"]),
            source_file(instance_root, row["scen_path"]),
            int(row["num_agents"]), maps,
        )
        selected.append(Task(
            f"{family}/{name}/seed{identity[1]}/n{identity[2]}",
            obstacles, starts, goals, policy_seed=0,
            provenance={"base_key": key, "suite": family,
                        "map_name": name, "generation_seed": identity[1]},
            horizon=FAMILY_HORIZONS[family],
        ))
    if len(selected) != 3193 or len({task.task_id for task in selected}) != 3193:
        raise ValueError("POGEMA benchmark must contain 3,193 unique tasks")
    return selected, sources
