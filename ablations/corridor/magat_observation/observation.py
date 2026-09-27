from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence

import numpy as np


MAGAT_PLUS_COST_TO_GO_SCALE = 100


class NativeMagatPlusCostToGo:
    """Generate compact local shortest-path cost deltas for MAGAT+.

    Values are stored as signed int8 deltas clipped to [-100, 100].  The PyG
    adapter divides the channel by 100 before passing it to the network.
    Distance maps are computed once per episode/solution and reused at every
    timestep.
    """

    def __init__(
        self,
        global_obstacles,
        targets,
        *,
        obs_radius: int = 5,
        scale: int = MAGAT_PLUS_COST_TO_GO_SCALE,
    ) -> None:
        if obs_radius <= 0:
            raise ValueError("obs_radius must be positive")
        if not 1 <= scale <= 127:
            raise ValueError("cost-to-go scale must fit in signed int8")

        obstacles = np.asarray(global_obstacles, dtype=np.bool_)
        targets_array = np.asarray(targets, dtype=np.int32)
        if obstacles.ndim != 2:
            raise ValueError("global_obstacles must be a two-dimensional grid")
        if targets_array.ndim != 2 or targets_array.shape[1] != 2:
            raise ValueError("targets must have shape [num_agents, 2]")

        self.obstacles = obstacles
        self.targets = targets_array
        self.obs_radius = obs_radius
        self.scale = scale
        self._unreachable = int(obstacles.size + scale + 1)
        self._distance_tables = np.stack(
            [self._build_distance_table(tuple(target)) for target in targets_array]
        )
        self._padded_distance_tables = np.pad(
            self._distance_tables,
            ((0, 0), (obs_radius, obs_radius), (obs_radius, obs_radius)),
            constant_values=self._unreachable,
        )
        self._local_offsets = np.arange(-obs_radius, obs_radius + 1)

    def _build_distance_table(self, target: tuple[int, int]) -> np.ndarray:
        height, width = self.obstacles.shape
        target_x, target_y = target
        if (
            target_x < 0
            or target_x >= height
            or target_y < 0
            or target_y >= width
            or self.obstacles[target_x, target_y]
        ):
            raise ValueError(f"Invalid MAGAT+ target coordinate: {target}")

        distances = np.full(
            self.obstacles.shape, self._unreachable, dtype=np.int32
        )
        distances[target_x, target_y] = 0
        queue = deque([(target_x, target_y)])
        while queue:
            x, y = queue.popleft()
            next_distance = distances[x, y] + 1
            for next_x, next_y in (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
            ):
                if (
                    0 <= next_x < height
                    and 0 <= next_y < width
                    and not self.obstacles[next_x, next_y]
                    and next_distance < distances[next_x, next_y]
                ):
                    distances[next_x, next_y] = next_distance
                    queue.append((next_x, next_y))
        return distances

    def generate(self, positions) -> np.ndarray:
        positions_array = np.asarray(positions, dtype=np.int32)
        if positions_array.shape != self.targets.shape:
            raise ValueError(
                f"positions must have shape {self.targets.shape}, "
                f"got {positions_array.shape}"
            )

        radius = self.obs_radius
        height, width = self.obstacles.shape
        if (
            np.any(positions_array[:, 0] < 0)
            or np.any(positions_array[:, 0] >= height)
            or np.any(positions_array[:, 1] < 0)
            or np.any(positions_array[:, 1] >= width)
        ):
            raise ValueError("At least one agent is outside the global grid")

        agent_indices = np.arange(len(positions_array))
        base_distances = self._distance_tables[
            agent_indices, positions_array[:, 0], positions_array[:, 1]
        ]
        if np.any(base_distances >= self._unreachable):
            invalid_agents = np.flatnonzero(base_distances >= self._unreachable)
            raise ValueError(
                f"Agents cannot reach their targets: {invalid_agents.tolist()}"
            )

        rows = (
            positions_array[:, 0, None] + self._local_offsets[None, :] + radius
        )
        columns = (
            positions_array[:, 1, None] + self._local_offsets[None, :] + radius
        )
        local_distances = self._padded_distance_tables[
            agent_indices[:, None, None],
            rows[:, :, None],
            columns[:, None, :],
        ]
        valid = local_distances < self._unreachable
        deltas = np.where(
            valid,
            local_distances - base_distances[:, None, None],
            self.scale,
        )
        channels = np.clip(deltas, -self.scale, self.scale).astype(np.int8)
        return np.pad(channels, ((0, 0), (1, 1), (1, 1)))


def _goal_channel(position, target, obs_radius: int) -> np.ndarray:
    size = 2 * obs_radius + 3
    channel = np.zeros((size, size), dtype=np.int8)
    centre = (size // 2, size // 2)
    goal = np.asarray(target, dtype=np.int64) - np.asarray(position, dtype=np.int64)

    if np.all(np.abs(goal) <= obs_radius):
        channel[centre[0] + goal[0], centre[1] + goal[1]] = 1
        return channel

    angle = np.arctan2(goal[1], goal[0])
    goal_sign = np.sign(goal)
    distance = size // 2
    if np.pi / 4 <= angle <= np.pi * 3 / 4 or -np.pi * 3 / 4 <= angle <= -np.pi / 4:
        goal_y = int(distance * (goal_sign[1] + 1))
        goal_x = int(centre[0] + np.round(distance * goal[0] / np.abs(goal[1])))
    else:
        goal_x = int(distance * (goal_sign[0] + 1))
        goal_y = int(centre[1] + np.round(distance * goal[1] / np.abs(goal[0])))
    channel[goal_x, goal_y] = 1
    return channel


def _adjacency(positions: np.ndarray, comm_radius: float) -> np.ndarray:
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt(np.sum(differences.astype(np.float32) ** 2, axis=-1))
    adjacency = np.where(distances <= comm_radius, distances, 0.0).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def build_native_magat_plus_observation(
    local_obstacles,
    local_agents,
    positions,
    targets,
    *,
    cost_to_go: NativeMagatPlusCostToGo,
    obs_radius: int = 5,
    comm_radius: float = 7.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the native MAGAT+ CNN inputs and communication graph.

    Both offline dataset generation and online POGEMA evaluation call this
    function. Local obstacle/agent views use the native POGEMA MAPF contract:
    one ``(2 * obs_radius + 1)`` square per agent.
    """
    if obs_radius <= 0:
        raise ValueError("obs_radius must be positive")
    if comm_radius <= 0:
        raise ValueError("comm_radius must be positive")

    obstacles = np.asarray(local_obstacles, dtype=np.int8)
    agents = np.asarray(local_agents, dtype=np.int8)
    positions_array = np.asarray(positions, dtype=np.int32)
    targets_array = np.asarray(targets, dtype=np.int32)
    num_agents = len(positions_array)
    view_size = 2 * obs_radius + 1
    expected_views_shape = (num_agents, view_size, view_size)

    if obstacles.shape != expected_views_shape:
        raise ValueError(
            f"local_obstacles must have shape {expected_views_shape}, got {obstacles.shape}"
        )
    if agents.shape != expected_views_shape:
        raise ValueError(
            f"local_agents must have shape {expected_views_shape}, got {agents.shape}"
        )
    if positions_array.shape != (num_agents, 2):
        raise ValueError(f"positions must have shape ({num_agents}, 2)")
    if targets_array.shape != (num_agents, 2):
        raise ValueError(f"targets must have shape ({num_agents}, 2)")
    if cost_to_go.obs_radius != obs_radius:
        raise ValueError(
            f"cost-to-go obs_radius={cost_to_go.obs_radius} does not match "
            f"observation obs_radius={obs_radius}"
        )

    cost_to_go_channels = cost_to_go.generate(positions_array)
    node_features = np.stack(
        [
            np.stack(
                [
                    np.pad(obstacles[agent_idx], 1),
                    np.pad(agents[agent_idx], 1),
                    _goal_channel(
                        positions_array[agent_idx], targets_array[agent_idx], obs_radius
                    ),
                    cost_to_go_channels[agent_idx],
                ]
            )
            for agent_idx in range(num_agents)
        ]
    ).astype(np.int8)
    return node_features, _adjacency(positions_array, comm_radius), positions_array


def build_native_magat_plus_from_pogema(
    observations: Sequence[Mapping],
    *,
    cost_to_go: NativeMagatPlusCostToGo | None = None,
    obs_radius: int = 5,
    comm_radius: float = 7.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adapt native POGEMA MAPF observations to the shared MAGAT+ builder."""
    if not observations:
        raise ValueError("observations must contain at least one agent")
    required = {
        "obstacles",
        "agents",
        "global_obstacles",
        "global_xy",
        "global_target_xy",
    }
    for agent_idx, observation in enumerate(observations):
        missing = required.difference(observation)
        if missing:
            raise KeyError(
                f"POGEMA observation {agent_idx} is missing native fields: {sorted(missing)}"
            )

    targets = [observation["global_target_xy"] for observation in observations]
    if cost_to_go is None:
        cost_to_go = NativeMagatPlusCostToGo(
            observations[0]["global_obstacles"],
            targets,
            obs_radius=obs_radius,
        )

    return build_native_magat_plus_observation(
        [observation["obstacles"] for observation in observations],
        [observation["agents"] for observation in observations],
        [observation["global_xy"] for observation in observations],
        targets,
        cost_to_go=cost_to_go,
        obs_radius=obs_radius,
        comm_radius=comm_radius,
    )
