"""Safe deployment mask for agents in locally settled neighborhoods."""

from __future__ import annotations

import torch
from torch import Tensor


def settled_observation_mask(
    global_positions: Tensor,
    global_goals: Tensor,
    local_chat_ids: Tensor,
    shard_start: int,
    shard_end: int,
) -> Tensor:
    """Return local agents whose ego and every encoded neighbor are on goal.

    Empty neighbor slots use ``-1`` and are treated as settled.  The mask is
    recomputed from the current state every step, so an agent becomes active
    again as soon as an off-goal agent enters its encoded neighborhood.
    """
    if local_chat_ids.ndim != 2:
        raise ValueError("local_chat_ids must have shape [local_agents, neighbors]")
    if shard_end - shard_start != local_chat_ids.shape[0]:
        raise ValueError("chat rows must match the contiguous local shard")
    on_goal = (global_positions == global_goals).all(dim=-1)
    neighbor_ids = local_chat_ids.long()
    valid = neighbor_ids.ge(0)
    safe_ids = neighbor_ids.clamp_min(0)
    neighbors_on_goal = on_goal[safe_ids] | ~valid
    local_on_goal = on_goal[shard_start:shard_end]
    return local_on_goal & neighbors_on_goal.all(dim=-1)
