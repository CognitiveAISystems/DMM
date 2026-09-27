"""
Map generators for MICPO training.

The model configurations use distinct map profiles. All generators return
numeric grids:
  - Maze maps:   wall-component generation
  - Random maps: obstacle-scatter with connectivity verification
  - House maps:  connected room grids with one door per shared wall (DMM only)

All maps are returned as 2-D lists (0=free, 1=obstacle) for POGEMA-GPU.
"""

from __future__ import annotations

import numpy as np

from training.micpo.task_generation import generate_maze_grid


def _retain_largest_free_component(array: np.ndarray) -> tuple[np.ndarray, int]:
    """Turn crop-induced disconnected free fragments back into obstacles."""
    height, width = array.shape
    unseen = set(map(tuple, np.argwhere(array == 0)))
    components: list[list[tuple[int, int]]] = []
    while unseen:
        # Explicit minimum keeps map generation independent of set iteration
        # order and therefore reproducible across Python processes.
        start = min(unseen)
        unseen.remove(start)
        component = [start]
        stack = [start]
        while stack:
            row, col = stack.pop()
            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbour = row + drow, col + dcol
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    component.append(neighbour)
                    stack.append(neighbour)
        components.append(component)
    if not components:
        return array.copy(), 0
    components.sort(key=lambda component: (-len(component), min(component)))
    largest = components[0]
    fitted = np.ones_like(array)
    rows, cols = zip(*largest)
    fitted[np.asarray(rows), np.asarray(cols)] = 0
    removed = int(np.count_nonzero(array == 0)) - len(largest)
    return fitted, removed


def _fit_grid_shape(
    grid: list[list[int]] | np.ndarray,
    height: int,
    width: int,
) -> list[list[int]]:
    """Fit a generated grid to the requested external environment shape.

    Some procedural generators express room/maze sizes in terms of interior
    cells and consequently return one or more extra rows/columns.  MICPO must
    not silently change the environment size (and therefore the feasible
    agent-density distribution), so crop excess cells and extend a rare
    undersized result with free cells adjacent to the existing component.
    """
    array = np.asarray(grid, dtype=np.int32)
    if array.ndim != 2:
        raise RuntimeError(
            f"Map generator returned shape {array.shape}, expected a 2-D grid"
        )
    if array.shape[0] >= height and array.shape[1] >= width:
        # A crop through an exterior wall can isolate a sliver of free space.
        # Search the (normally very small) set of possible crop origins and
        # prefer the most centred connected window.
        row_extra = int(array.shape[0]) - height
        col_extra = int(array.shape[1]) - width
        candidates: list[tuple[int, float, int, np.ndarray]] = []
        for row in range(row_extra + 1):
            for col in range(col_extra + 1):
                crop = array[row:row + height, col:col + width]
                connected_crop, removed = _retain_largest_free_component(crop)
                centre_distance = abs(row - row_extra / 2.0) + abs(
                    col - col_extra / 2.0
                )
                free_cells = int(np.count_nonzero(connected_crop == 0))
                candidates.append(
                    (removed, centre_distance, -free_cells, connected_crop)
                )
        if candidates:
            candidates.sort(key=lambda item: item[:3])
            return candidates[0][3].tolist()
    else:
        fitted = np.zeros((height, width), dtype=np.int32)
        copy_height = min(height, int(array.shape[0]))
        copy_width = min(width, int(array.shape[1]))
        fitted[:copy_height, :copy_width] = array[:copy_height, :copy_width]
        if _is_fully_connected(fitted.tolist(), height, width):
            return fitted.tolist()
    raise RuntimeError(
        "No connected crop fits procedural map shape "
        f"{tuple(array.shape)} to requested {height}x{width}"
    )


# ------------------------------------------------------------------ #
# Maze maps
# ------------------------------------------------------------------ #

def generate_maze_map(
    height: int,
    width: int,
    wall_components_min: int,
    wall_components_max: int,
    seed: int,
    *,
    fit_shape: bool = True,
) -> list[list[int]]:
    """
    Generate a wall-component maze.

    Returns a list-of-lists where 1=obstacle, 0=free.
    """
    grid = generate_maze_grid(
        height, width, wall_components_min, wall_components_max, seed
    )
    return _fit_grid_shape(grid, height, width) if fit_shape else grid


# ------------------------------------------------------------------ #
# Random maps (obstacle scatter + connectivity check)
# ------------------------------------------------------------------ #

def generate_random_map_passable(
    height: int,
    width: int,
    obstacle_density: float,
    rng: np.random.Generator,
    max_retries: int = 100,
    *,
    fallback: str = "best",
) -> list[list[int]]:
    """
    Place obstacles uniformly at random at the given density.
    Verify that all free cells are reachable (single connected component).
    Retry up to max_retries times with a fresh RNG state each time. When all
    candidates are disconnected, DMM repairs the best candidate and DMM08M
    repairs the last one, matching their respective training distributions.

    Returns a list-of-lists where 1=obstacle, 0=free.
    Raises RuntimeError if no passable map is found within max_retries.
    """
    best: tuple[int, np.ndarray] | None = None
    last_grid: list[list[int]] | None = None
    for attempt in range(max_retries):
        grid = _scatter_obstacles(height, width, obstacle_density, rng)
        last_grid = grid
        if _is_fully_connected(grid, height, width):
            return grid
        if fallback == "best":
            connected, removed = _retain_largest_free_component(
                np.asarray(grid, dtype=np.int32)
            )
            if best is None or removed < best[0]:
                best = removed, connected
    if fallback == "last":
        assert last_grid is not None
        return _keep_largest_free_component(last_grid, height, width)
    if best is not None and np.count_nonzero(best[1] == 0) > 0:
        return best[1].tolist()
    raise RuntimeError(f"Could not generate a non-empty {height}x{width} map")


def _keep_largest_free_component(
    grid: list[list[int]], height: int, width: int
) -> list[list[int]]:
    """Preserve DMM08M's last-candidate fallback and component tie behavior."""
    unseen = {
        (row, col)
        for row in range(height)
        for col in range(width)
        if grid[row][col] == 0
    }
    largest: set[tuple[int, int]] = set()
    while unseen:
        start = unseen.pop()
        component = {start}
        stack = [start]
        while stack:
            row, col = stack.pop()
            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (row + drow, col + dcol)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        if len(component) > len(largest):
            largest = component

    connected = [row[:] for row in grid]
    for row in range(height):
        for col in range(width):
            if connected[row][col] == 0 and (row, col) not in largest:
                connected[row][col] = 1
    return connected


def _scatter_obstacles(
    height: int,
    width: int,
    obstacle_density: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    total = height * width
    n_obstacles = int(total * obstacle_density)
    flat = np.zeros(total, dtype=np.int32)
    indices = rng.choice(total, size=n_obstacles, replace=False)
    flat[indices] = 1
    grid = flat.reshape(height, width).tolist()
    return grid


def _is_fully_connected(grid: list[list[int]], height: int, width: int) -> bool:
    """BFS flood fill to check all free cells are reachable."""
    # find first free cell
    start = None
    for r in range(height):
        for c in range(width):
            if grid[r][c] == 0:
                start = (r, c)
                break
        if start:
            break

    if start is None:
        return False  # no free cells at all

    visited = set()
    queue = [start]
    visited.add(start)
    while queue:
        r, c = queue.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited:
                if grid[nr][nc] == 0:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    total_free = sum(grid[r][c] == 0 for r in range(height) for c in range(width))
    return len(visited) == total_free


def generate_house_map(
    height: int,
    width: int,
    obstacle_density: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Generate a connected room/house grid close to the requested size."""
    room_height = int(rng.integers(5, 9))
    room_width = int(rng.integers(5, 9))
    num_rows = max(2, int(np.ceil((height + 1) / (room_height + 1))))
    num_cols = max(2, int(np.ceil((width + 1) / (room_width + 1))))
    grid = np.zeros(
        (
            room_height * num_rows + num_rows - 1,
            room_width * num_cols + num_cols - 1,
        ),
        dtype=np.int32,
    )

    grid[room_height :: room_height + 1, :] = 1
    grid[:, room_width :: room_width + 1] = 1
    row_doors = rng.integers(
        0, room_width, size=(num_rows - 1, num_cols)
    ) + np.arange(num_cols)[None, :] * (room_width + 1)
    np.put_along_axis(
        grid[room_height :: room_height + 1, :], row_doors, 0, axis=1
    )
    col_doors = rng.integers(
        0, room_height, size=(num_rows, num_cols - 1)
    ) + np.arange(num_rows)[:, None] * (room_height + 1)
    np.put_along_axis(
        grid[:, room_width :: room_width + 1], col_doors, 0, axis=0
    )

    # Sparse room-centre clutter. If a rare obstacle arrangement disconnects
    # the free space, drop only that clutter while retaining walls and doors.
    clutter = min(max(float(obstacle_density), 0.0), 0.08)
    candidates = grid == 0
    candidates[0 :: room_height + 1, :] = False
    candidates[:, 0 :: room_width + 1] = False
    candidates[room_height - 1 :: room_height + 1, :] = False
    candidates[:, room_width - 1 :: room_width + 1] = False
    clutter_mask = (rng.random(grid.shape) < clutter) & candidates
    grid[clutter_mask] = 1
    if not _is_fully_connected(grid.tolist(), *grid.shape):
        grid[clutter_mask] = 0
    return _fit_grid_shape(grid, height, width)


# ------------------------------------------------------------------ #
# Unified sampler
# ------------------------------------------------------------------ #

def sample_map(
    map_type: str,
    height: int,
    width: int,
    obstacle_density: float,
    wall_components_min: int,
    wall_components_max: int,
    maze_fraction: float,
    seed: int,
    rng: np.random.Generator,
    policy_class: str,
    room_fraction: float = 0.0,
) -> list[list[int]]:
    """
    Sample one map according to map_type.

    map_type:
        "maze"   — always maze
        "random" — always random
        "house"  — always connected room/house map
        "mixed"  — maze with probability maze_fraction, else random
        "mixed3" — maze/house probabilities are explicit; random is remainder
    seed is used for the maze generator; rng is used for random maps.
    """
    if policy_class not in {"dmm", "dmm08m"}:
        raise ValueError(f"Unsupported policy_class: {policy_class!r}")
    if policy_class == "dmm08m" and map_type not in {"maze", "random", "mixed"}:
        raise ValueError(f"Unsupported DMM-08M map_type: {map_type!r}")

    if map_type == "maze":
        kind = "maze"
    elif map_type == "random":
        kind = "random"
    elif map_type == "house":
        kind = "house"
    elif map_type == "mixed":
        kind = "maze" if rng.random() < maze_fraction else "random"
    elif map_type == "mixed3":
        if min(maze_fraction, room_fraction) < 0.0:
            raise ValueError("mixed3 fractions must be non-negative")
        if maze_fraction + room_fraction > 1.0:
            raise ValueError("mixed3 fractions must sum to at most one")
        draw = float(rng.random())
        if draw < maze_fraction:
            kind = "maze"
        elif draw < maze_fraction + room_fraction:
            kind = "house"
        else:
            kind = "random"
    else:
        raise ValueError(f"Unknown map_type: {map_type!r}")

    if kind == "maze":
        return generate_maze_map(
            height, width,
            wall_components_min=wall_components_min,
            wall_components_max=wall_components_max,
            seed=seed,
            fit_shape=policy_class == "dmm",
        )
    elif kind == "random":
        return generate_random_map_passable(
            height, width,
            obstacle_density=obstacle_density,
            rng=rng,
            fallback="last" if policy_class == "dmm08m" else "best",
        )
    else:
        return generate_house_map(
            height,
            width,
            obstacle_density=obstacle_density,
            rng=rng,
        )
