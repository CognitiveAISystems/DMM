"""Read archived MovingAI-format maps and scenarios in row/column order."""

from __future__ import annotations

from pathlib import Path


def source_file(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"source path escapes benchmark root: {relative}")
    return target


def read_instance(map_path: Path, scenario_path: Path, num_agents: int,
                  maps: dict[Path, tuple[str, ...]]) -> tuple:
    if map_path not in maps:
        lines = map_path.read_text().splitlines()
        if "map" not in lines:
            raise ValueError(f"invalid MovingAI map: {map_path}")
        maps[map_path] = tuple(
            "".join("." if char in ".GS" else "#" for char in line)
            for line in lines[lines.index("map") + 1 :]
        )
    points = [line.split() for line in scenario_path.read_text().splitlines()[1:]
              if line.strip()][:num_agents]
    if len(points) != num_agents or any(len(point) < 8 for point in points):
        raise ValueError(f"scenario has insufficient agents: {scenario_path}")
    return (
        maps[map_path],
        tuple((int(point[5]), int(point[4])) for point in points),
        tuple((int(point[7]), int(point[6])) for point in points),
    )
