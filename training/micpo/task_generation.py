"""MICPO task generation without a POGEMA CPU dependency.

Provides maze grids, agent/goal placement, and the five-cell observation border.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def generate_maze_grid(
    height: int,
    width: int,
    wall_components_min: int,
    wall_components_max: int,
    seed: int,
) -> list[list[int]]:
    """Generate a numeric maze obstacle grid."""
    settings_rng = np.random.default_rng(seed)
    sampled_width = settings_rng.integers(width, width + 1)
    sampled_height = settings_rng.integers(height, height + 1)
    obstacle_density = settings_rng.uniform(0.0, 1.0)
    wall_components = settings_rng.integers(
        wall_components_min, wall_components_max + 1
    )
    go_straight = settings_rng.uniform(0.75, 0.85)

    rng = np.random.default_rng(seed)
    shape = (
        (int(sampled_height) // 2) * 2 + 3,
        (int(sampled_width) // 2) * 2 + 3,
    )
    density = (
        int(shape[0] * shape[1] * obstacle_density // wall_components)
        if wall_components != 0
        else 0
    )
    maze = np.zeros(shape, dtype=int)
    maze[0, :] = maze[-1, :] = 1
    maze[:, 0] = maze[:, -1] = 1

    for _ in range(density):
        x = rng.integers(0, shape[1] // 2) * 2
        y = rng.integers(0, shape[0] // 2) * 2
        maze[y, x] = 1
        last_direction = (0, 0)
        for _ in range(int(wall_components)):
            neighbors = []
            probabilities = []
            if x > 1:
                neighbors.append((y, x - 2))
                probabilities.append(
                    go_straight
                    if (y, x - 2) == (y + last_direction[0], x + last_direction[1])
                    else 1 - go_straight
                )
            if x < shape[1] - 2:
                neighbors.append((y, x + 2))
                probabilities.append(
                    go_straight
                    if (y, x + 2) == (y + last_direction[0], x + last_direction[1])
                    else 1 - go_straight
                )
            if y > 1:
                neighbors.append((y - 2, x))
                probabilities.append(
                    go_straight
                    if (y - 2, x) == (y + last_direction[0], x + last_direction[1])
                    else 1 - go_straight
                )
            if y < shape[0] - 2:
                neighbors.append((y + 2, x))
                probabilities.append(
                    go_straight
                    if (y + 2, x) == (y + last_direction[0], x + last_direction[1])
                    else 1 - go_straight
                )

            if not neighbors:
                continue
            if all(prob == go_straight for prob in probabilities):
                probabilities = [1 / len(probabilities)] * len(probabilities)
            else:
                total = sum(probabilities)
                probabilities = [prob / total for prob in probabilities]
            next_y, next_x = neighbors[rng.choice(range(len(neighbors)), p=probabilities)]
            last_direction = (next_y - y, next_x - x)
            if maze[next_y, next_x] == 0:
                maze[next_y, next_x] = 1
                maze[next_y + (y - next_y) // 2, next_x + (x - next_x) // 2] = 1
                x, y = next_x, next_y

    return maze[1:-1, 1:-1].tolist()


def _numeric_positions(
    obstacles: np.ndarray, num_agents: int, seed: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Match pogema 1.3.2a4's connected-component ``placing`` order."""
    grid = obstacles.copy()
    components = [0, 0]
    component_id = 2
    height, width = grid.shape
    moves = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    for row in range(height):
        for col in range(width):
            if grid[row, col] != 0:
                continue
            grid[row, col] = component_id
            components.append(1)
            queue = deque([(row, col)])
            while queue:
                current_row, current_col = queue.popleft()
                for drow, dcol in moves:
                    next_row, next_col = current_row + drow, current_col + dcol
                    if (0 <= next_row < height and 0 <= next_col < width
                            and grid[next_row, next_col] == 0):
                        grid[next_row, next_col] = component_id
                        components[component_id] += 1
                        queue.append((next_row, next_col))
            component_id += 1

    order = [
        (row, col)
        for row in range(height)
        for col in range(width)
        if grid[row, col] >= 2
    ]
    np.random.default_rng(seed).shuffle(order)
    requests: list[list[int]] = [[] for _ in components]
    starts: list[tuple[int, int]] = []
    goals: list[tuple[int, int]] = [(-1, -1) for _ in range(num_agents)]
    done = 0
    for row, col in order:
        color = int(grid[row, col])
        grid[row, col] = 0
        if requests[color]:
            goals[requests[color].pop()] = row, col
            done += 1
            continue
        if len(starts) >= num_agents:
            if done >= num_agents:
                break
            continue
        if components[color] >= 2:
            components[color] -= 2
            requests[color].append(len(starts))
            starts.append((row, col))
    if len(starts) != num_agents or done != num_agents:
        raise OverflowError("Not enough free cells to place all agents and targets")
    return starts, goals


def sample_scenario(
    grid: list[list[int]],
    num_agents: int,
    seed: int,
    *,
    obs_radius: int = 5,
) -> tuple[list[list[int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Place starts/goals and add the MAPF observation border."""
    obstacles = np.asarray(grid, dtype=np.int32)
    if obstacles.ndim != 2 or not np.isin(obstacles, (0, 1)).all():
        raise ValueError("Expected a rectangular binary obstacle map")
    starts, goals = _numeric_positions(obstacles, num_agents, seed)

    height, width = obstacles.shape
    bordered = np.zeros((height + 2 * obs_radius, width + 2 * obs_radius), dtype=np.int32)
    bordered[obs_radius - 1, obs_radius - 1:width + obs_radius + 1] = 1
    bordered[obs_radius - 1:height + obs_radius + 1, obs_radius - 1] = 1
    bordered[height + obs_radius, obs_radius - 1:width + obs_radius + 1] = 1
    bordered[obs_radius - 1:height + obs_radius + 1, width + obs_radius] = 1
    bordered[obs_radius:height + obs_radius, obs_radius:width + obs_radius] = obstacles
    shift = lambda points: [(row + obs_radius, col + obs_radius) for row, col in points]
    return bordered.tolist(), shift(starts), shift(goals)


def task_from_bordered(grid, positions, goals, horizon: int):
    """Convert a bordered MICPO scenario to a POGEMA-GPU task."""
    from pogema_gpu.tasks import Task

    radius = 5
    height = len(grid) - 2 * radius
    width = len(grid[0]) - 2 * radius
    if height <= 0 or width <= 0:
        raise ValueError("MICPO map must include the five-cell observation border")
    observation = tuple(
        "".join("#" if cell else "." for cell in row) for row in grid
    )
    interior = tuple(row[radius:-radius] for row in observation[radius:-radius])
    starts = tuple((int(row) - radius, int(col) - radius) for row, col in positions)
    targets = tuple((int(row) - radius, int(col) - radius) for row, col in goals)
    return Task(
        task_id="micpo-episode",
        obstacles=interior,
        starts=starts,
        goals=targets,
        observation_obstacles=observation,
        obs_radius=radius,
        policy_seed=0,
        horizon=horizon,
    )
