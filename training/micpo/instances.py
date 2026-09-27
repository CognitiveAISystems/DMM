"""
Oneshot map / scenario generation.

build_env_instance(config, iteration_rng) -> EnvInstance
    Produces a concrete grid + start positions + goals.
    No lifelong goal sequences — on_target="nothing" throughout.

build_env_instance_from_seeds(config, map_seed, scenario_seed) -> EnvInstance
    Deterministic version for validation set generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from training.micpo.task_generation import sample_scenario


@dataclass
class EnvInstance:
    """Everything needed to construct one POGEMA-GPU episode."""
    grid:          list[list[int]]      # 2-D list, 1=obstacle 0=free
    positions:     list[list[int]]      # [N, 2]  [row, col]
    goals:         list[list[int]]      # [N, 2]
    height:        int
    width:         int
    map_seed:      Optional[int]
    scenario_seed: Optional[int]


def _sample_map(config, height: int, width: int, map_seed: Optional[int]):
    """Select the configured map sampler and dimensions."""
    from training.micpo.maps import sample_map

    kwargs = dict(
        map_type=config.map_type,
        height=height,
        width=width,
        obstacle_density=config.obstacle_density,
        wall_components_min=config.maze_wall_components_min,
        wall_components_max=config.maze_wall_components_max,
        maze_fraction=config.maze_fraction,
        seed=map_seed if map_seed is not None else 0,
        rng=np.random.default_rng(map_seed),
        policy_class=config.policy_class,
        room_fraction=getattr(config, "room_fraction", 0.0),
    )
    grid = sample_map(**kwargs)
    if config.policy_class == "dmm":
        return grid, len(grid), len(grid[0])
    if config.policy_class == "dmm08m":
        # DMM08M uses the configured dimensions of the uncropped maze.
        return grid, height, width
    raise ValueError(f"Unsupported policy_class: {config.policy_class!r}")


def build_env_instance(
    config,
    iteration_rng: np.random.Generator,
) -> EnvInstance:
    """
    Build one EnvInstance according to config.env_mode and config.map_type.
    """
    env_mode = config.env_mode
    if env_mode == "random":
        map_seed      = int(iteration_rng.integers(0, 2**31))
        scenario_seed = int(iteration_rng.integers(0, 2**31))

    elif env_mode == "fixed_map":
        if config.map_seed is None:
            raise ValueError("env_mode='fixed_map' requires map_seed")
        map_seed      = config.map_seed
        scenario_seed = int(iteration_rng.integers(0, 2**31))

    elif env_mode == "fixed_map_scenario":
        if config.map_seed is None or config.scenario_seed is None:
            raise ValueError("env_mode='fixed_map_scenario' requires map_seed and scenario_seed")
        map_seed      = config.map_seed
        scenario_seed = config.scenario_seed

    elif env_mode == "custom_map":
        if config.custom_map_path is None:
            raise ValueError("env_mode='custom_map' requires custom_map_path")
        map_seed      = None
        scenario_seed = int(iteration_rng.integers(0, 2**31))

    elif env_mode == "custom_map_scenario":
        if config.custom_map_path is None:
            raise ValueError("env_mode='custom_map_scenario' requires custom_map_path")
        if config.scenario_seed is None:
            raise ValueError("env_mode='custom_map_scenario' requires scenario_seed")
        map_seed      = None
        scenario_seed = config.scenario_seed

    else:
        raise ValueError(f"Unknown env_mode: {env_mode!r}")

    if env_mode in ("custom_map", "custom_map_scenario"):
        grid, height, width = _load_custom_map(config.custom_map_path)
    else:
        height = config.map_height
        width  = config.map_width
        if getattr(config, "map_height_min", None) is not None and getattr(config, "map_height_max", None) is not None:
            height = int(iteration_rng.integers(config.map_height_min, config.map_height_max + 1))
        if getattr(config, "map_width_min", None) is not None and getattr(config, "map_width_max", None) is not None:
            width = int(iteration_rng.integers(config.map_width_min, config.map_width_max + 1))
        grid, height, width = _sample_map(config, height, width, map_seed)

    try:
        env_grid, positions, goals = _sample_scenario(
            grid=grid,
            height=height,
            width=width,
            num_agents=config.num_agents,
            scenario_seed=scenario_seed,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Scenario placement failed on {height}x{width} map "
            f"with {config.num_agents} agents: {exc}"
        ) from exc

    return EnvInstance(
        grid=env_grid,
        positions=positions,
        goals=goals,
        height=len(env_grid),
        width=len(env_grid[0]),
        map_seed=map_seed,
        scenario_seed=scenario_seed,
    )


def build_env_instance_from_seeds(
    config,
    map_seed: Optional[int],
    scenario_seed: Optional[int],
    custom_map_path: Optional[str] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_agents: Optional[int] = None,
) -> EnvInstance:
    """Deterministic build from explicit seeds (validation set generation)."""
    if custom_map_path is not None:
        grid, height, width = _load_custom_map(custom_map_path)
    else:
        if height is None:
            height = config.map_height
        if width is None:
            width = config.map_width
        grid, height, width = _sample_map(config, height, width, map_seed)

    n_agents = num_agents if num_agents is not None else config.num_agents
    try:
        env_grid, positions, goals = _sample_scenario(
            grid=grid,
            height=height,
            width=width,
            num_agents=n_agents,
            scenario_seed=scenario_seed,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Scenario placement failed on {height}x{width} map "
            f"with {n_agents} agents: {exc}"
        ) from exc

    return EnvInstance(
        grid=env_grid,
        positions=positions,
        goals=goals,
        height=len(env_grid),
        width=len(env_grid[0]),
        map_seed=map_seed,
        scenario_seed=scenario_seed,
    )


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _sample_scenario(
    grid: list[list[int]],
    height: int,
    width: int,
    num_agents: int,
    scenario_seed: Optional[int],
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """
    Place agents using the configured MICPO sampling protocol.

    Returns:
        env_grid  [H'][W'] — full grid including obs_radius border
        positions [N][2]
        goals     [N][2]

    NOTE: coordinates are in the full (bordered) grid frame.
    """
    return sample_scenario(
        grid, num_agents, scenario_seed if scenario_seed is not None else 0
    )


def _load_custom_map(path: str) -> tuple[list[list[int]], int, int]:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    grid   = data["map"] if isinstance(data, dict) else data
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        raise ValueError("custom map must be a 2-D numeric obstacle grid")
    height = len(grid)
    width  = len(grid[0])
    return grid, height, width
