"""MICPO's one-episode interface backed by the bundled POGEMA-GPU simulator and tokenizer."""

from __future__ import annotations

from dataclasses import replace

import torch

from pogema_gpu.observations.cuda import CUDADMMTokenizer
from pogema_gpu.simulator import CUDABatch
from training.micpo.task_generation import task_from_bordered


_OBSERVATION_CONTRACT = {
    "num_previous_actions": 5,
    "cost2go_value_limit": 20,
    "agents_radius": 5,
    "cost2go_radius": 5,
    "context_size": 256,
    "max_num_neighbors": 13,
}


class MICPOEnvironment:
    """Thin MICPO interface using the same observation code as evaluation."""

    def __init__(self, grid, positions, goals, max_episode_steps=128,
                 device="cuda", on_target="nothing", observation_config=None):
        if on_target != "nothing":
            raise ValueError("MICPO only supports one-shot on_target='nothing'")
        if observation_config is not None:
            for name, expected in _OBSERVATION_CONTRACT.items():
                if getattr(observation_config, name, None) != expected:
                    raise ValueError(f"MICPO observation {name} must be {expected}")
        self.device = torch.device(device)
        self.max_episode_steps = max_episode_steps
        self._task = task_from_bordered(grid, positions, goals, max_episode_steps)
        self.num_agents = self._task.num_agents
        self._batch = CUDABatch([self._task], device=self.device)
        self.tokenizer = CUDADMMTokenizer(self._batch, cache_mode="vendor-window")
        self.pos = self._coordinates(self._batch.positions[0])
        self.goals = self._coordinates(self._batch.goals[0])

    def _coordinates(self, flat):
        return torch.stack((flat // self._batch.width, flat % self._batch.width), dim=-1)

    def reset(self, positions=None, goals=None):
        task = self._task
        if positions is not None or goals is not None:
            def unpad(points):
                if isinstance(points, torch.Tensor):
                    points = points.detach().cpu().tolist()
                return tuple((int(row) - 5, int(col) - 5)
                             for row, col in points)

            task = replace(
                task,
                starts=unpad(positions) if positions is not None else task.starts,
                goals=unpad(goals) if goals is not None else task.goals,
            )
        self._batch.reset_at([0], [task])
        self._task = task
        self.pos = self._coordinates(self._batch.positions[0])
        self.goals = self._coordinates(self._batch.goals[0])

    def observe(self):
        observations, neighbors = self.tokenizer.prepare([0])
        return observations[0], neighbors[0]

    def remember(self, actions):
        if actions.shape != (self.num_agents,):
            raise ValueError("MICPO actions must be [N]")
        proposed = actions.to(dtype=torch.long).unsqueeze(0).contiguous()
        # Validation and rollout keep producing logits for completed cohort
        # members until the whole cohort stops.
        self.tokenizer.remember(proposed, include_finished=True)
        return proposed

    def step(self, actions):
        proposed = self.remember(actions)
        self._batch.step(proposed)
        self.pos = self._coordinates(self._batch.positions[0])
        return None, {}
