"""Exact map endpoints and master files for the million-agent experiment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

@dataclass
class EnvInstance:
    grid: object
    positions: object
    goals: object
    height: int
    width: int
    map_seed: int
    scenario_seed: int


FORMAT_VERSION = 1


def largest_free_component_mask(obstacles: np.ndarray) -> np.ndarray:
    """Return the largest 4-connected free component as a boolean mask."""
    height, width = obstacles.shape
    free = obstacles.reshape(-1) == 0
    labels = np.full(height * width, -1, dtype=np.int32)
    sizes: list[int] = []
    label = 0
    for root_value in np.flatnonzero(free):
        root = int(root_value)
        if labels[root] >= 0:
            continue
        labels[root] = label
        queue = deque([root])
        size = 0
        while queue:
            cell = queue.popleft()
            size += 1
            row, column = divmod(cell, width)
            for neighbor in (
                cell - width if row else -1,
                cell + width if row + 1 < height else -1,
                cell - 1 if column else -1,
                cell + 1 if column + 1 < width else -1,
            ):
                if neighbor >= 0 and free[neighbor] and labels[neighbor] < 0:
                    labels[neighbor] = label
                    queue.append(neighbor)
        sizes.append(size)
        label += 1
    if not sizes:
        return np.zeros_like(obstacles, dtype=np.bool_)
    return (labels == int(np.argmax(sizes))).reshape(height, width)


def global_disjoint_endpoints(
    grid: np.ndarray, n_agents: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform globally distributed, unique and mutually disjoint endpoints."""
    height, width = grid.shape
    free = np.flatnonzero(grid.reshape(-1) == 0)
    if free.size < 2 * n_agents:
        raise ValueError(
            f"need {2 * n_agents:,} free cells for endpoints, have {free.size:,}"
        )
    rng = np.random.default_rng(seed)
    selected = rng.choice(free, size=2 * n_agents, replace=False)
    starts_flat = selected[:n_agents]
    goals_flat = selected[n_agents:]
    starts = np.stack((starts_flat // width, starts_flat % width), axis=-1)
    goals = np.stack((goals_flat // width, goals_flat % width), axis=-1)
    return torch.from_numpy(starts.astype(np.int32)), torch.from_numpy(
        goals.astype(np.int32)
    )


def validate_master(master: dict) -> None:
    metadata = master["metadata"]
    if int(metadata["format_version"]) != FORMAT_VERSION:
        raise ValueError("unsupported scalability master format")
    grid = torch.as_tensor(master["grid"])
    starts = torch.as_tensor(master["starts"]).long()
    goals = torch.as_tensor(master["goals"]).long()
    if starts.shape != goals.shape or starts.ndim != 2 or starts.shape[1] != 2:
        raise ValueError("starts and goals must both be [N,2]")
    height, width = grid.shape
    endpoints = torch.cat(
        (starts[:, 0] * width + starts[:, 1], goals[:, 0] * width + goals[:, 1])
    )
    if endpoints.unique().numel() != endpoints.numel():
        raise ValueError("starts and goals must be globally unique and disjoint")
    if bool(grid[starts[:, 0], starts[:, 1]].any()):
        raise ValueError("a start lies on an obstacle")
    if bool(grid[goals[:, 0], goals[:, 1]].any()):
        raise ValueError("a goal lies on an obstacle")


def save_master(master: dict, path: str | Path) -> None:
    validate_master(master)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(master, path)


def load_master(path: str | Path) -> dict:
    master = torch.load(path, map_location="cpu", weights_only=False)
    validate_master(master)
    return master


def master_prefix(master: dict, n_agents: int) -> EnvInstance:
    validate_master(master)
    starts = torch.as_tensor(master["starts"])
    if not 0 < n_agents <= starts.shape[0]:
        raise ValueError(f"invalid prefix N={n_agents}")
    metadata = master["metadata"]
    grid = torch.as_tensor(master["grid"], dtype=torch.uint8)
    return EnvInstance(
        grid=grid,
        positions=starts[:n_agents],
        goals=torch.as_tensor(master["goals"])[:n_agents],
        height=grid.shape[0],
        width=grid.shape[1],
        map_seed=int(metadata["root_seed"]),
        scenario_seed=int(metadata["root_seed"]) + 1,
    )
