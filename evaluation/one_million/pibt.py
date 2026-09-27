"""Batched stateful PIBT for million-agent evaluation.

The policy supplies final consensus scores.  PIBT masks impossible actions,
ranks the remaining actions deterministically, and resolves vertex conflicts
and edge swaps.  Each environment keeps the standard lifelong PIBT priority:
unfinished agents gain one point per step, while agents on goal keep only the
fractional initial-distance tie break.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import Tensor


_MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _distance(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
    if start == goal:
        return 0
    height, width = grid.shape
    distance = np.full((height, width), -1, dtype=np.int64)
    distance[goal] = 0
    queue = deque([goal])
    while queue:
        row, col = queue.popleft()
        next_distance = int(distance[row, col]) + 1
        for drow, dcol in _MOVES[1:]:
            nr, nc = row + drow, col + dcol
            if (
                0 <= nr < height
                and 0 <= nc < width
                and not grid[nr, nc]
                and distance[nr, nc] < 0
            ):
                distance[nr, nc] = next_distance
                if (nr, nc) == start:
                    return next_distance
                queue.append((nr, nc))
    return height * width


class BatchedPIBT:
    """Exact deterministic PIBT over a heterogeneous batch of small maps."""

    def __init__(
        self,
        env_instances,
        device: torch.device,
        priority_mode: str = "exact_cpu",
        resolver_mode: str = "components",
    ):
        from evaluation.one_million.cuda_pibt import load_cuda_pibt

        self.device = device
        self.extension = load_cuda_pibt()
        self.num_envs = len(env_instances)
        self.num_agents = len(env_instances[0].positions)
        self.max_height = max(len(inst.grid) for inst in env_instances)
        self.max_width = max(len(inst.grid[0]) for inst in env_instances)
        self.max_num_cells = max(
            len(inst.grid) * len(inst.grid[0]) for inst in env_instances
        )

        obstacles = torch.ones(
            self.num_envs,
            self.max_height,
            self.max_width,
            dtype=torch.bool,
            device=device,
        )
        widths, heights, num_cells = [], [], []
        goals = []
        for env_index, inst in enumerate(env_instances):
            grid = np.asarray(inst.grid, dtype=np.bool_)
            height, width = grid.shape
            obstacles[env_index, :height, :width] = torch.as_tensor(
                grid, dtype=torch.bool, device=device
            )
            widths.append(width)
            heights.append(height)
            num_cells.append(height * width)
            goals.append(inst.goals)

        self.obstacles = obstacles
        self.widths = torch.tensor(widths, dtype=torch.long, device=device)
        self.heights = torch.tensor(heights, dtype=torch.long, device=device)
        self.num_cells = torch.tensor(num_cells, dtype=torch.long, device=device)
        self.goals = torch.stack([
            torch.as_tensor(value, dtype=torch.long, device=device)
            for value in goals
        ])
        self.goal_ids = (
            self.goals[..., 0] * self.widths[:, None] + self.goals[..., 1]
        )
        if priority_mode == "exact_cpu":
            initial_priorities = []
            for inst in env_instances:
                grid = np.asarray(inst.grid, dtype=np.bool_)
                scale = float(grid.shape[0] * grid.shape[1])
                initial_priorities.append([
                    _distance(
                        grid,
                        tuple(torch.as_tensor(start).tolist()),
                        tuple(torch.as_tensor(goal).tolist()),
                    ) / scale
                    for start, goal in zip(inst.positions, inst.goals)
                ])
            self.initial_priorities = torch.tensor(
                initial_priorities, dtype=torch.float32, device=device
            )
        elif priority_mode == "manhattan_gpu":
            starts = torch.stack([
                torch.as_tensor(inst.positions, dtype=torch.long, device=device)
                for inst in env_instances
            ])
            distance = (starts - self.goals).abs().sum(dim=-1).float()
            self.initial_priorities = distance / self.num_cells[:, None].float()
        else:
            raise ValueError(
                "priority_mode must be 'exact_cpu' or 'manhattan_gpu', "
                f"got {priority_mode!r}"
            )
        self.priority_mode = priority_mode
        if resolver_mode not in {
            "sequential", "parallel_occupancy", "compact", "components",
            "hybrid", "warp"
        }:
            raise ValueError(
                "resolver_mode must be 'sequential', 'parallel_occupancy', "
                "'compact', 'components', 'hybrid', or 'warp', "
                f"got {resolver_mode!r}"
            )
        self.resolver_mode = resolver_mode
        resolvers = {
            "sequential": self.extension.resolve_batched,
            "parallel_occupancy": (
                self.extension.resolve_batched_parallel_occupancy
            ),
            "compact": self.extension.resolve_batched_compact,
            "components": self.extension.resolve_batched_components,
            "hybrid": self.extension.resolve_batched_hybrid,
            "warp": self.extension.resolve_batched_warp,
        }
        self._resolve = resolvers[resolver_mode]
        self.moves = torch.tensor(_MOVES, dtype=torch.long, device=device)
        self.reset()

    def reset(self) -> None:
        self.priorities = self.initial_priorities.clone()

    @torch.no_grad()
    def set_initial_priorities_from_distances(self, distances: Tensor) -> None:
        """Initialize canonical PIBT priorities from exact GPU distances."""
        if distances.shape != (self.num_envs, self.num_agents):
            raise ValueError(
                "distances must be "
                f"{(self.num_envs, self.num_agents)}, got {tuple(distances.shape)}"
            )
        if bool(distances.lt(0).any().item()):
            raise ValueError("initial PIBT distances must be reachable")
        self.initial_priorities = (
            distances.to(device=self.device, dtype=torch.float32)
            / self.num_cells[:, None].to(torch.float32)
        )
        self.reset()

    @torch.no_grad()
    def step(
        self,
        scores: Tensor,
        positions: Tensor,
        env_done: Tensor | None = None,
        forbidden_actions: Tensor | None = None,
        prefer_unoccupied_ties: bool = False,
        commit: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return executed actions and a mask of actions changed by PIBT."""
        envs, agents, actions = scores.shape
        if (envs, agents, actions) != (self.num_envs, self.num_agents, 5):
            raise ValueError(
                f"scores must be {(self.num_envs, self.num_agents, 5)}, "
                f"got {tuple(scores.shape)}"
            )

        targets = positions.unsqueeze(2) + self.moves.view(1, 1, 5, 2)
        rows, cols = targets[..., 0], targets[..., 1]
        inside = (
            (rows >= 0)
            & (rows < self.heights[:, None, None])
            & (cols >= 0)
            & (cols < self.widths[:, None, None])
        )
        safe_rows = rows.clamp(0, self.max_height - 1)
        safe_cols = cols.clamp(0, self.max_width - 1)
        env_index = torch.arange(self.num_envs, device=self.device)[:, None, None]
        free = ~self.obstacles[env_index, safe_rows, safe_cols]
        valid = inside & free
        if forbidden_actions is not None:
            valid &= ~forbidden_actions
            # Keep the resolver well-defined if all five actions happened to
            # be constrained for an agent.
            no_candidate = ~valid.any(dim=-1)
            valid[..., 0] |= no_candidate
        valid = valid.contiguous()
        candidate_ids = (
            rows * self.widths[:, None, None] + cols
        ).masked_fill(~valid, -1).contiguous()
        current_ids = (
            positions[..., 0] * self.widths[:, None] + positions[..., 1]
        ).contiguous()

        if prefer_unoccupied_ties:
            if scores.dtype != torch.int32:
                raise ValueError(
                    "prefer_unoccupied_ties requires packed int32 distance scores"
                )
            occupied_now = torch.zeros(
                self.num_envs,
                self.max_num_cells,
                dtype=torch.bool,
                device=self.device,
            )
            occupied_now.scatter_(1, current_ids, True)
            candidate_occupied = occupied_now.gather(
                1, candidate_ids.clamp_min(0).view(self.num_envs, -1)
            ).view_as(candidate_ids)
            # distance_scores reserves bit 15 for the original PIBT paper's
            # unoccupied-cell tie preference.  The distance stride is 2^16,
            # so this bonus can never reverse an exact-distance comparison.
            scores = scores + (
                valid & ~candidate_occupied
            ).to(torch.int32) * (1 << 15)

        if scores.is_floating_point():
            masked_scores = (
                scores.float().nan_to_num(0.0).masked_fill(~valid, -torch.inf)
            )
        else:
            masked_scores = scores.masked_fill(
                ~valid, torch.iinfo(scores.dtype).min
            )
        policy_actions = masked_scores.argmax(dim=-1)
        preferences = torch.argsort(
            masked_scores, dim=-1, descending=True, stable=True
        ).contiguous()
        order = torch.argsort(
            self.priorities, dim=-1, descending=True, stable=True
        ).contiguous()

        executed, next_ids = self._resolve(
            current_ids,
            candidate_ids,
            preferences,
            valid,
            order,
            self.num_cells,
            self.max_num_cells,
        )
        if env_done is not None and env_done.any():
            executed = torch.where(env_done[:, None], torch.zeros_like(executed), executed)
            next_ids = torch.where(env_done[:, None], current_ids, next_ids)

        overridden = executed.ne(policy_actions)
        if env_done is not None:
            overridden &= ~env_done[:, None]

        if commit:
            self.commit(next_ids)
        return executed, overridden, next_ids

    @torch.no_grad()
    def commit(self, next_ids: Tensor) -> None:
        """Advance lifelong priorities once for the accepted joint action."""
        at_goal = next_ids.eq(self.goal_ids)
        self.priorities = torch.where(
            at_goal,
            self.priorities - torch.floor(self.priorities),
            self.priorities + 1.0,
        )

    @torch.no_grad()
    def priority_order(self) -> Tensor:
        return torch.argsort(
            self.priorities, dim=-1, descending=True, stable=True
        )

    def positions_from_ids(self, next_ids: Tensor) -> Tensor:
        """Convert resolved cell ids to positions without leaving the GPU."""
        return torch.stack(
            (
                torch.div(next_ids, self.widths[:, None], rounding_mode="floor"),
                next_ids.remainder(self.widths[:, None]),
            ),
            dim=-1,
        )
