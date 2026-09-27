"""Build the MAGAT+ graph dataset for the corridor scenario.

The underlying scenario is the one in scenario.py, the same states the tokenized
dataset encodes; MAGAT+ just consumes them through its own observation pipeline
instead of the shared tokenizer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from magat_observation import (NativeMagatPlusCostToGo,
                               build_native_magat_plus_observation,
                               native_magat_plus_to_pyg)
from scenario import A_GOAL, B_GOAL, GRID, TRAJECTORIES

OBS_RADIUS = 5
COMM_RADIUS = 7.0
EDGE_ATTR_FOR_MESSAGES = "positions+manhattan"

OBSTACLES = np.array(GRID, dtype=np.int8)


def window(grid: np.ndarray, centre: tuple[int, int], pad_value: int) -> np.ndarray:
    """The agent's local view, padded with `pad_value` outside the grid."""
    size = 2 * OBS_RADIUS + 1
    view = np.full((size, size), pad_value, dtype=grid.dtype)
    for i in range(size):
        row = centre[0] - OBS_RADIUS + i
        if not 0 <= row < grid.shape[0]:
            continue
        for j in range(size):
            col = centre[1] - OBS_RADIUS + j
            if 0 <= col < grid.shape[1]:
                view[i, j] = grid[row, col]
    return view


def neighbour_view(positions: list[tuple[int, int]], agent: int) -> np.ndarray:
    size = 2 * OBS_RADIUS + 1
    view = np.zeros((size, size), dtype=np.int8)
    row, col = positions[agent]
    for other, (other_row, other_col) in enumerate(positions):
        if other == agent:
            continue
        delta_row, delta_col = other_row - row, other_col - col
        if abs(delta_row) <= OBS_RADIUS and abs(delta_col) <= OBS_RADIUS:
            view[delta_row + OBS_RADIUS, delta_col + OBS_RADIUS] = 1
    return view


def step_graph(positions, targets, cost_to_go):
    local_obstacles = np.stack([window(OBSTACLES, position, 1) for position in positions])
    local_agents = np.stack([neighbour_view(positions, agent)
                             for agent in range(len(positions))])
    features, adjacency, coordinates = build_native_magat_plus_observation(
        local_obstacles, local_agents, positions, targets, cost_to_go=cost_to_go,
        obs_radius=OBS_RADIUS, comm_radius=COMM_RADIUS,
    )
    return native_magat_plus_to_pyg(features, adjacency, coordinates, use_edge_attr=True,
                                    edge_attr_for_messages=EDGE_ATTR_FOR_MESSAGES)


def build(output: Path) -> None:
    targets = [A_GOAL, B_GOAL]
    cost_to_go = NativeMagatPlusCostToGo(OBSTACLES.astype(bool), targets,
                                         obs_radius=OBS_RADIUS)
    graphs = []
    for trajectory in TRAJECTORIES:
        for step in range(trajectory.horizon):
            graph = step_graph(list(trajectory.positions[step]), targets, cost_to_go)
            graph.y = torch.tensor(trajectory.actions[step], dtype=torch.long)
            graphs.append(graph)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, output)
    print(f"{output}: {len(graphs)} graphs, node features {tuple(graphs[0].x.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "magat_graphs.pt")
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
