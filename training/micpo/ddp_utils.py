"""
DDP utilities — process group init, advantage all-gather.

Single-GPU (WORLD_SIZE=1) works without any code change: all functions
degrade gracefully to no-ops.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor


def init_ddp(backend: str = "nccl") -> tuple[int, int, int, bool]:
    """
    Initialise process group from torchrun environment variables.

    Returns:
        (rank, local_rank, world_size, is_master)

    Single-GPU: returns (0, 0, 1, True) without calling init_process_group.
    """
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)

    is_master = rank == 0
    return rank, local_rank, world_size, is_master


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_gather_advantages(
    advantages: Tensor,
    world_size: int,
) -> Tensor:
    """
    Gather advantages from all ranks, concatenate along dim=0, and
    re-normalise globally.

    This ensures the normalisation uses the full G×world_size trajectory
    pool rather than each rank's local G subset.

    Args:
        advantages: [G_local] or [G_local, N] on the current rank
        world_size: total number of ranks

    Returns:
        Globally normalised advantages with shape [G_local] or [G_local, N].
    """
    if world_size == 1:
        return advantages   # no-op

    gathered = [torch.zeros_like(advantages) for _ in range(world_size)]
    dist.all_gather(gathered, advantages)
    global_advantages = torch.cat(gathered, dim=0)

    # Re-normalise over the full global pool
    if global_advantages.dim() == 1:
        mean = global_advantages.mean()
        std  = global_advantages.std()
        local_start = dist.get_rank() * advantages.shape[0]
        local_end   = local_start + advantages.shape[0]
        normed_global = (global_advantages - mean) / (std + 1e-8)
        return normed_global[local_start:local_end]
    else:
        # [G_total, N]: normalise per-agent across trajectories
        mean = global_advantages.mean(dim=0, keepdim=True)
        std  = global_advantages.std(dim=0, keepdim=True)
        local_start = dist.get_rank() * advantages.shape[0]
        local_end   = local_start + advantages.shape[0]
        normed_global = (global_advantages - mean) / (std + 1e-8)
        return normed_global[local_start:local_end]


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
