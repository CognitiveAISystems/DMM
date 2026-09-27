"""Exact observation generation for a contiguous multi-GPU agent shard."""

from __future__ import annotations

from collections.abc import Callable

import torch

from evaluation.one_million.observation import GPUObservationGenerator


class ShardedGPUObservationGenerator(GPUObservationGenerator):
    """Own local BFS caches while querying neighbors from replicated state."""

    def __init__(self, *args, shard_start: int, shard_end: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.shard_start = int(shard_start)
        self.shard_end = int(shard_end)
        self.agent_chat_ids = None
        self.global_action_history = None

    def create_agents(self, positions, goals, **kwargs):
        positions = torch.as_tensor(positions)
        goals = torch.as_tensor(goals)
        super().create_agents(
            positions[self.shard_start:self.shard_end],
            goals[self.shard_start:self.shard_end],
            **kwargs,
        )
        self.global_action_history = torch.full(
            (positions.shape[0], self.cfg.num_previous_actions),
            self.default_hist_token,
            dtype=torch.long,
            device=self.device,
        )

    def update_global_agents(self, positions, goals, last_actions):
        positions = torch.as_tensor(positions, dtype=torch.long, device=self.device)
        goals = torch.as_tensor(goals, dtype=torch.long, device=self.device)
        self.pos_t = positions[self.shard_start:self.shard_end]
        self.goals_t = goals[self.shard_start:self.shard_end]
        if last_actions is not None:
            actions = torch.as_tensor(
                last_actions, dtype=torch.long, device=self.device
            )
            actions = torch.where(
                (actions >= 0) & (actions <= 4), actions,
                torch.full_like(actions, 5),
            )
            self.global_action_history = torch.roll(
                self.global_action_history, shifts=-1, dims=1
            )
            self.global_action_history[:, -1] = self.env_act_to_token[actions]

    def generate_sharded_observations(
        self,
        global_positions: torch.Tensor,
        global_goals: torch.Tensor,
        gather_local: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Return local observations and global-id chat connections."""
        from pogema_gpu.kernels.cuda_bfs import (
            get_chat_neighbors_spatial_sharded_cuda,
            get_neighbors_spatial_sharded_cuda,
        )

        cost_tokens, local_next_actions = self._compute_cost2go_and_actions(
            self.pos_t, self.goals_t
        )
        global_next_actions = gather_local(local_next_actions.contiguous())
        neighbor_tokens = get_neighbors_spatial_sharded_cuda(
            self.pos_t,
            global_positions,
            global_goals,
            self.global_action_history,
            global_next_actions,
            self.coord_lookup,
            self.height,
            self.width,
            self.cfg.agents_radius,
            self.cfg.cost2go_value_limit,
            self.coord_offset,
            self.pad_token,
            self.cfg.num_previous_actions,
            min(13, global_positions.shape[0]),
        )
        self.agent_chat_ids = get_chat_neighbors_spatial_sharded_cuda(
            self.pos_t,
            global_positions,
            self.height,
            self.width,
            self.cfg.agents_radius,
            13,
        )
        observations = torch.cat((cost_tokens, neighbor_tokens), dim=1)
        padding = self.cfg.context_size - observations.shape[1]
        if padding > 0:
            observations = torch.nn.functional.pad(
                observations, (0, padding), value=self.pad_token
            )
        return observations
