"""Blocked-motion count for MICPO's POGEMA-GPU environment.

The collision resolver updates positions only after handling conflicts. An
agent is counted when it requested a non-stay action but remained in place;
this includes obstacle hits and agent-agent conflicts.
"""

from __future__ import annotations

import torch
from torch import Tensor


def count_blocked(
    pos_before: Tensor,   # [N, 2]  int64
    pos_after:  Tensor,   # [N, 2]  int64
    actions:    Tensor,   # [N]     int64  (0=stay, 1-4=move)
) -> int:
    """
    Number of agents that intended to move but stayed in place.
    Single environment, single step.
    """
    intended_to_move = (actions != 0)                          # [N] bool
    stayed           = (pos_after == pos_before).all(dim=-1)   # [N] bool
    return int((intended_to_move & stayed).sum().item())


def count_blocked_batched(
    pos_before: Tensor,   # [G, N, 2]  int64
    pos_after:  Tensor,   # [G, N, 2]  int64
    actions:    Tensor,   # [G, N]     int64
) -> Tensor:
    """
    [G] int64 tensor — blocked agents per environment.
    """
    intended_to_move = (actions != 0)                            # [G, N]
    stayed           = (pos_after == pos_before).all(dim=-1)     # [G, N]
    return (intended_to_move & stayed).sum(dim=-1)               # [G]
