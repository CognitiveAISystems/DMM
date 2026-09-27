"""
OneshotBuffer — pre-allocated storage for a variable-length oneshot episode.

DMM rollout-specific tensors:
  - Each MAPF step produces K categorical votes (K comm rounds, no pre-round).
  - votes         [G, H, N, K]   int64   — per-round vote g_t
  - log_pi_old_r  [G, H, N, K]   float32 — per-round log π_old(g_t)  (for IS ratio)
  - log_pi_old_i  [G, H, N]      float32 — sum over rounds (for SR)
  - z0            [G, H, N, 5]   float32 — Dirichlet-sampled initial state (for replay)

Subsampling is on the MAPF step (H) axis — all K rounds are kept together
for every selected step (rounds are causally chained, can't drop individually).

Shapes (H = max_horizon, K = n_comm_rounds, R = K):
    obs            [G, H, N, context_size]
    agent_chat_ids [G, H, N, max_num_neighbors]
    actions        [G, H, N]    int64   — final action = argmax(z_K)
    log_pi_old_i   [G, H, N]   float32  — sum of per-round log-probs (for SR)
    log_pi_old_r   [G, H, N, R] float32  — per-round log-probs (for IS ratio)
    votes          [G, H, N, R] int64   — per-round votes
    z0             [G, H, N, 5] float32  — initial z state from rollout
    pos_before     [G, H, N, 2] int64
    pos_after      [G, H, N, 2] int64
    first_arrival  [G, N]       int64
    episode_length [G]          int64
    goals          [G, N, 2]    int64
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


class OneshotBuffer:
    """
    Pre-allocated storage for G oneshot MAPF rollouts, each up to H steps.
    Extended for DMM with per-round vote and log-prob storage.
    """

    def __init__(
        self,
        G: int,
        H: int,
        N: int,
        n_rounds: int,           # K (comm rounds only, no pre-round)
        context_size: int,
        max_num_neighbors: int,
        device: torch.device,
    ):
        self.G = G
        self.H = H
        self.N = N
        self.n_rounds = n_rounds
        self.device = device

        R = n_rounds

        self.obs = torch.zeros(G, H, N, context_size, dtype=torch.long, device=device)
        self.agent_chat_ids = torch.full(
            (G, H, N, max_num_neighbors), -1, dtype=torch.long, device=device
        )
        self.actions      = torch.zeros(G, H, N,    dtype=torch.long,    device=device)
        self.log_pi_old_i = torch.zeros(G, H, N,    dtype=torch.float32, device=device)
        self.log_pi_old_r = torch.zeros(G, H, N, R, dtype=torch.float32, device=device)
        self.votes        = torch.zeros(G, H, N, R, dtype=torch.long,    device=device)
        self.z0           = torch.zeros(G, H, N, 5, dtype=torch.float32, device=device)
        self.pos_before   = torch.zeros(G, H, N, 2, dtype=torch.long,    device=device)
        self.pos_after    = torch.zeros(G, H, N, 2, dtype=torch.long,    device=device)

        self.first_arrival  = torch.full((G, N),  H, dtype=torch.long,  device=device)
        self.episode_length = torch.full((G,),     H, dtype=torch.long,  device=device)
        self.goals          = torch.zeros(G, N, 2, dtype=torch.long,    device=device)

        self.n_blocked_total: int = 0

    def store_step(
        self,
        step: int,
        obs_batch:      Tensor,   # [G, N, context_size]
        chat_ids_batch: Tensor,   # [G, N, max_num_neighbors]
        actions:        Tensor,   # [G, N]
        log_pi_rounds:  Tensor,   # [G, N, K]
        votes:          Tensor,   # [G, N, K]
        z0:             Tensor,   # [G, N, 5]
        pos_before:     Tensor,   # [G, N, 2]
        pos_after:      Tensor,   # [G, N, 2]
    ) -> None:
        self.obs[:, step]            = obs_batch
        self.agent_chat_ids[:, step] = chat_ids_batch
        self.actions[:, step]        = actions
        self.log_pi_old_r[:, step]   = log_pi_rounds
        self.log_pi_old_i[:, step]   = log_pi_rounds.sum(-1)   # for SR
        self.votes[:, step]          = votes
        self.z0[:, step]             = z0
        self.pos_before[:, step]     = pos_before
        self.pos_after[:, step]      = pos_after

    def filter_envs(self, indices: Tensor) -> "OneshotBuffer":
        """Return a new buffer containing only the rows at `indices` (G axis)."""
        nb = OneshotBuffer.__new__(OneshotBuffer)
        nb.G      = len(indices)
        nb.H      = self.H
        nb.N      = self.N
        nb.n_rounds = self.n_rounds
        nb.device = self.device
        nb.n_blocked_total = 0

        nb.obs            = self.obs[indices].contiguous()
        nb.agent_chat_ids = self.agent_chat_ids[indices].contiguous()
        nb.actions        = self.actions[indices].contiguous()
        nb.log_pi_old_i   = self.log_pi_old_i[indices].contiguous()
        nb.log_pi_old_r   = self.log_pi_old_r[indices].contiguous()
        nb.votes          = self.votes[indices].contiguous()
        nb.z0             = self.z0[indices].contiguous()
        nb.pos_before     = self.pos_before[indices].contiguous()
        nb.pos_after      = self.pos_after[indices].contiguous()
        nb.first_arrival  = self.first_arrival[indices].contiguous()
        nb.episode_length = self.episode_length[indices].contiguous()
        nb.goals          = self.goals[indices].contiguous()
        return nb


class SubsampledBuffer:
    """
    [G, k, ...] subset of an OneshotBuffer after MAPF-step subsampling.

    All K rounds are kept intact for each selected step —
    rounds are causally chained and cannot be dropped individually.
    """

    def __init__(
        self,
        G: int,
        k: int,
        N: int,
        n_rounds: int,
        context_size: int,
        max_num_neighbors: int,
        device: torch.device,
    ):
        self.G = G
        self.k = k
        self.N = N
        self.n_rounds = n_rounds
        self.device = device

        R = n_rounds

        self.obs            = torch.zeros(G, k, N, context_size,       dtype=torch.long,    device=device)
        self.agent_chat_ids = torch.full( (G, k, N, max_num_neighbors), -1, dtype=torch.long, device=device)
        self.actions        = torch.zeros(G, k, N,                     dtype=torch.long,    device=device)
        self.log_pi_old_i   = torch.zeros(G, k, N,                     dtype=torch.float32, device=device)
        self.log_pi_old_r   = torch.zeros(G, k, N, R,                  dtype=torch.float32, device=device)
        self.votes          = torch.zeros(G, k, N, R,                  dtype=torch.long,    device=device)
        self.z0             = torch.zeros(G, k, N, 5,                  dtype=torch.float32, device=device)
        self.timestep_indices = torch.zeros(G, k,                      dtype=torch.long,    device=device)


def subsample_buffer(
    buffer: OneshotBuffer,
    ratio: float,
    k_override: int = 0,
) -> SubsampledBuffer:
    """
    Sample exactly k MAPF-step indices per trajectory,
    drawn from [0, episode_length[g]) only.

    All K rounds are kept for each selected step.

    k = k_override if k_override > 0, else ceil(H * ratio).
    """
    G, H, N = buffer.G, buffer.H, buffer.N
    device   = buffer.device

    k = int(k_override) if k_override > 0 else max(1, math.ceil(H * ratio))

    sub = SubsampledBuffer(
        G, k, N,
        buffer.n_rounds,
        buffer.obs.shape[-1],
        buffer.agent_chat_ids.shape[-1],
        device,
    )

    for g in range(G):
        ep_len = max(1, min(int(buffer.episode_length[g].item()), H))

        if k <= ep_len:
            idx_g, _ = torch.randperm(ep_len, device=device)[:k].sort()
        else:
            idx_g, _ = torch.randint(0, ep_len, (k,), device=device).sort()

        sub.timestep_indices[g] = idx_g
        sub.obs[g]              = buffer.obs[g, idx_g]
        sub.agent_chat_ids[g]   = buffer.agent_chat_ids[g, idx_g]
        sub.actions[g]          = buffer.actions[g, idx_g]
        sub.log_pi_old_i[g]     = buffer.log_pi_old_i[g, idx_g]
        sub.log_pi_old_r[g]     = buffer.log_pi_old_r[g, idx_g]
        sub.votes[g]            = buffer.votes[g, idx_g]
        sub.z0[g]               = buffer.z0[g, idx_g]

    return sub
