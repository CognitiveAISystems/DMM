"""Generate the four exact million-agent POGEMA maze scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.one_million.maze_generator import MazeGenerator
from evaluation.one_million.scenario import (
    FORMAT_VERSION, global_disjoint_endpoints,
    largest_free_component_mask, save_master,
)


SEEDS = (101, 202, 303, 404)
AGENTS = 1_048_576
GRID_SHA256 = {
    101: "01bf3e13f98072911a06b34cb6fa13e68a1963e0c016f36fc38b460369968461",
    202: "4f210d371cffd7087c706d6bba7f32b27eb0890c46f729ae677cef206af2720f",
    303: "04cd7da1c9a016479264dfc834fd7ff513b0de847c9b2766e30efd15cad27568",
    404: "138e6071ba20ef3cdc41f1d89416f28bc15bbbcfba7a74024c2fd500d1b11886",
}


def generate(out: Path, seeds=SEEDS) -> None:
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for seed in seeds:
        destination = out / f"master_seed{seed}.pt"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        parameters = dict(
            width=2304, height=2304, obstacle_density=0.3,
            wall_components=8, go_straight=0.8, seed=seed,
        )
        raw = MazeGenerator.string_to_array(MazeGenerator.generate_maze(**parameters))
        grid = raw[:2304, :2304].astype(np.uint8)
        grid_sha = hashlib.sha256(grid.tobytes()).hexdigest()
        if grid_sha != GRID_SHA256[seed]:
            raise RuntimeError(f"Maze generator differs for seed {seed}")
        connected = largest_free_component_mask(grid)
        sampling_grid = np.where(connected, 0, 1).astype(np.uint8)
        starts, goals = global_disjoint_endpoints(sampling_grid, AGENTS, seed + 1)
        metadata = dict(
            format_version=FORMAT_VERSION, size=2304, density=0.3,
            actual_density=float(grid.mean()), root_seed=seed,
            max_agents=AGENTS, goal_mode="global_uniform_disjoint",
            connected_component="endpoints_restricted_to_largest_free_component",
            map_type="pogema_maze", generator_parameters=parameters,
            grid_uint8_sha256=grid_sha, generator_raw_shape=list(raw.shape),
            postprocessing="first 2304 rows/columns; walls unchanged",
            free_cells=int((grid == 0).sum()),
            endpoint_component_free_cells=int(connected.sum()),
            agent_occupancy=AGENTS / int(connected.sum()),
        )
        master = dict(
            metadata=metadata, grid=torch.from_numpy(grid),
            starts=starts, goals=goals,
            manhattan=(starts - goals).abs().sum(dim=-1).to(torch.int16),
        )
        save_master(master, destination)
        metadata["master_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        (out / f"master_seed{seed}.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        entries.append(dict(root_seed=seed, n_agents=AGENTS, path=destination.name))
        print(f"Saved seed={seed}: {destination}", flush=True)
    (out / "index.json").write_text(json.dumps(dict(
        format="dmm_nested_global_v1", size=2304, density=0.3,
        map_type="pogema_maze", goal_mode="global_uniform_disjoint",
        horizon=32768, prefixes=[AGENTS], entries=entries,
    ), indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303,404")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not seeds or any(seed not in SEEDS for seed in seeds):
        parser.error("seeds must be drawn from 101,202,303,404")
    generate(args.out, seeds)


if __name__ == "__main__":
    main()
