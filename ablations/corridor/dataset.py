"""Tokenize the two corridor trajectories into the Arrow file the trainers read.

DMM and LC-MAPF share one tokenizer, so both train on the file written here. The
observations are built the same way as at inference time — create the agents
once, then update positions, goals and last actions each step — so a trained
policy sees the same encoding it was trained on.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa

from scenario import A_GOAL, B_GOAL, GRID, TRAJECTORIES

THIRD_PARTY = Path(__file__).resolve().parent / "third_party"
OBS_RADIUS = 5


def observation_generator():
    """The vendored C++ tokenizer, compiled on first use."""
    sys.path.insert(0, str(THIRD_PARTY))
    import cppimport
    cppimport.imp("lc_mapf.observation_generator")
    from lc_mapf.observation_generator import InputParameters, ObservationGenerator

    parameters = InputParameters(
        20,           # cost2go_value_limit
        13,           # neighbor slots per observation
        5,            # previous actions
        256,          # context size
        OBS_RADIUS,   # observation radius
        5,            # agents radius
        64,           # grid step
        False,        # save cost2go
        -1,           # task type id
    )
    return ObservationGenerator, parameters


def padded_grid() -> list[list[int]]:
    """Wall-pad the grid by the observation radius, as the tokenizer expects."""
    rows, cols = len(GRID), len(GRID[0])
    padded = [[1] * (cols + 2 * OBS_RADIUS) for _ in range(rows + 2 * OBS_RADIUS)]
    for row in range(rows):
        for col in range(cols):
            padded[row + OBS_RADIUS][col + OBS_RADIUS] = GRID[row][col]
    return padded


def shift(position: tuple[int, int]) -> list[int]:
    return [position[0] + OBS_RADIUS, position[1] + OBS_RADIUS]


def tokenize(trajectory, grid, generator_type, parameters):
    """Returns observations [H, N, 256], neighbors [H, N, 13] and actions [H, N]."""
    generator = generator_type(grid, parameters)
    goals = [shift(A_GOAL), shift(B_GOAL)]
    generator.create_agents([shift(p) for p in trajectory.positions[0]], goals)

    last_actions = [-1, -1]
    observations, neighbors, actions = [], [], []
    for step in range(trajectory.horizon):
        generator.update_agents([shift(p) for p in trajectory.positions[step]],
                                goals, last_actions)
        observations.append(np.array(generator.generate_observations(), dtype=np.int8))
        neighbors.append(np.array(generator.get_agents_in_obs(), dtype=np.int8))
        actions.append(np.array(trajectory.actions[step], dtype=np.int8))
        last_actions = list(trajectory.actions[step])
    return np.stack(observations), np.stack(neighbors), np.stack(actions)


def build(output: Path) -> None:
    generator_type, parameters = observation_generator()
    grid = padded_grid()
    observations, neighbors, actions = zip(
        *(tokenize(trajectory, grid, generator_type, parameters)
          for trajectory in TRAJECTORIES)
    )
    table = pa.table({
        "input_tensors": pa.array(np.concatenate(observations).tolist(),
                                  type=pa.list_(pa.list_(pa.int8()))),
        "gt_actions": pa.array(np.concatenate(actions).tolist(), type=pa.list_(pa.int8())),
        "agents_in_obs": pa.array(np.concatenate(neighbors).tolist(),
                                  type=pa.list_(pa.list_(pa.int8()))),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(output), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    print(f"{output}: {table.num_rows} rows from {len(TRAJECTORIES)} trajectories")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "part_0_0.arrow")
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
