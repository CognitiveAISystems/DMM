"""Shared CUDA BFS extension and tensor operations."""

from functools import lru_cache
import hashlib
import os
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load_cuda_bfs():
    """Compile or load the shared BFS/observation CUDA extension."""
    from torch.utils.cpp_extension import load

    if torch.__version__.split(".")[:2] != ["2", "13"]:
        raise RuntimeError("CUDA BFS requires PyTorch 2.13")
    directory = Path(__file__).resolve().parent
    sources = [directory / name for name in (
        "bfs_bindings.cpp", "bfs_kernel.cu", "env_kernel.cu",
        "extract_kernel.cu", "neighbors_kernel.cu")]
    flags = ["-O2", "-U__CUDA_NO_HALF_CONVERSIONS__",
             "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"]
    identity = b"".join(path.read_bytes() for path in sources + [directory / "bfs_helpers.cuh"])
    identity += str((torch.__version__, torch.version.cuda, flags,
                     os.environ.get("CC"), os.environ.get("CXX"),
                     os.environ.get("TORCH_CUDA_ARCH_LIST"))).encode()
    name = "pogema_cuda_bfs_" + hashlib.sha256(identity).hexdigest()[:16]
    cache = Path(os.environ.get("TORCH_EXTENSIONS_DIR",
                                Path.home() / ".cache/pogema-gpu")) / name
    cache.mkdir(parents=True, exist_ok=True)
    return load(name, [str(path) for path in sources], build_directory=str(cache),
                extra_include_paths=[str(directory)], extra_cuda_cflags=flags,
                verbose=os.environ.get("POGEMA_GPU_BUILD_VERBOSE") == "1")


def is_available() -> bool:
    """Check that the CUDA extension can be loaded."""
    try:
        load_cuda_bfs()
    except Exception as exc:
        import warnings
        warnings.warn(f"CUDA BFS kernel compilation failed: {exc}")
        return False
    return True


def fused_bfs_cost2go(
    obstacles: torch.Tensor,
    agent_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    H: int, W: int,
    radius: int = 5,
    value_limit: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused BFS + window extraction + normalization + next-action.

    Returns:
        windows     : int16[N, (2*radius+1)^2]  normalized cost-to-go window
        action_codes: int32[N]                   4-bit next-action mask
    """
    module = load_cuda_bfs()
    obs_i8 = obstacles.to(torch.int8).contiguous()
    windows, actions = module.fused_bfs_cost2go(
        obs_i8,
        agent_pos.contiguous(),
        goal_pos.contiguous(),
        H, W, radius, value_limit,
    )
    return windows, actions


def raw_bfs_cost2go(
    obstacles: torch.Tensor,
    agent_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    H: int, W: int,
    radius: int = 31,
) -> torch.Tensor:
    """
    Raw BFS distances in a (2*radius+1)^2 window, no normalization.

    Returns:
        raw_windows: int16[N, (2*radius+1)^2]  (-1 = unreachable/wall)
    """
    module = load_cuda_bfs()
    obs_i8 = obstacles.to(torch.int8).contiguous()
    return module.raw_bfs_cost2go(
        obs_i8,
        agent_pos.contiguous(),
        goal_pos.contiguous(),
        H, W, radius,
    )
def extract_and_normalize_cuda(
    cache_windows: torch.Tensor,
    pos: torch.Tensor,
    cache_center: torch.Tensor,
    value_limit: int,
    obs_radius: int,
    cache_radius: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts small windows from cached large windows and normalizes them via CUDA.
    """
    module = load_cuda_bfs()
    windows, actions = module.extract_and_normalize_cuda(
        cache_windows.contiguous(),
        pos.contiguous(),
        cache_center.contiguous(),
        value_limit, obs_radius, cache_radius
    )
    return windows, actions

def get_neighbors_cuda(
    pos: torch.Tensor,
    goals: torch.Tensor,
    history: torch.Tensor,
    next_actions: torch.Tensor,
    coord_lookup: torch.Tensor,
    agents_radius: int,
    limit: int,
    coord_offset: int,
    pad_token: int,
    inf_dist: int,
    num_hist: int,
    num_neighbors: int
) -> torch.Tensor:
    """
    Gather tokens for the nearest N neighbors for each agent via CUDA.

    Returns:
        agents_indices: int64[N, num_neighbors * (5 + num_hist)]
    """
    module = load_cuda_bfs()

    return module.get_neighbors_cuda(
        pos.contiguous(),
        goals.contiguous(),
        history.contiguous(),
        next_actions.to(torch.int32).contiguous(),
        coord_lookup.contiguous(),
        agents_radius,
        limit,
        coord_offset,
        pad_token,
        inf_dist,
        num_hist,
        num_neighbors
    )


def get_neighbors_spatial_cuda(
    pos: torch.Tensor,
    goals: torch.Tensor,
    history: torch.Tensor,
    next_actions: torch.Tensor,
    coord_lookup: torch.Tensor,
    H: int,
    W: int,
    agents_radius: int,
    limit: int,
    coord_offset: int,
    pad_token: int,
    num_hist: int,
    num_neighbors: int,
) -> torch.Tensor:
    """Gather observation neighbors in O(N + local candidates) on CUDA."""
    module = load_cuda_bfs()
    return module.get_neighbors_spatial_cuda(
        pos.contiguous(),
        goals.contiguous(),
        history.contiguous(),
        next_actions.to(torch.int32).contiguous(),
        coord_lookup.contiguous(),
        H,
        W,
        agents_radius,
        limit,
        coord_offset,
        pad_token,
        num_hist,
        num_neighbors,
    )


def get_chat_neighbors_spatial_cuda(
    pos: torch.Tensor,
    H: int,
    W: int,
    agents_radius: int,
    max_neighbors: int = 13,
) -> torch.Tensor:
    """Gather communication neighbor ids with the same exact ordering."""
    module = load_cuda_bfs()
    return module.get_chat_neighbors_spatial_cuda(
        pos.contiguous(), H, W, agents_radius, max_neighbors
    )


def get_neighbors_spatial_sharded_cuda(
    ego_pos: torch.Tensor,
    pos: torch.Tensor,
    goals: torch.Tensor,
    history: torch.Tensor,
    next_actions: torch.Tensor,
    coord_lookup: torch.Tensor,
    H: int,
    W: int,
    agents_radius: int,
    limit: int,
    coord_offset: int,
    pad_token: int,
    num_hist: int,
    num_neighbors: int,
) -> torch.Tensor:
    """Gather local receiver observations from a global agent cell list."""
    module = load_cuda_bfs()
    return module.get_neighbors_spatial_sharded_cuda(
        ego_pos.contiguous(), pos.contiguous(), goals.contiguous(),
        history.contiguous(), next_actions.to(torch.int32).contiguous(),
        coord_lookup.contiguous(), H, W, agents_radius, limit, coord_offset,
        pad_token, num_hist, num_neighbors,
    )


def get_chat_neighbors_spatial_sharded_cuda(
    ego_pos: torch.Tensor,
    pos: torch.Tensor,
    H: int,
    W: int,
    agents_radius: int,
    max_neighbors: int = 13,
) -> torch.Tensor:
    """Gather global neighbor ids for a local receiver shard."""
    module = load_cuda_bfs()
    return module.get_chat_neighbors_spatial_sharded_cuda(
        ego_pos.contiguous(), pos.contiguous(), H, W, agents_radius,
        max_neighbors,
    )

def env_step_cuda(
    pos: torch.Tensor, actions: torch.Tensor, grid: torch.Tensor, goals: torch.Tensor,
    moves: torch.Tensor, next_flat: torch.Tensor, who_was_at: torch.Tensor,
    claimants: torch.Tensor, d_changed: torch.Tensor, d_all_on_goal: torch.Tensor,
    solve_time: torch.Tensor, current_step: int
) -> bool:
    module = load_cuda_bfs()
    return module.env_step_cuda(
        pos.contiguous(), actions.contiguous(), grid.contiguous(), goals.contiguous(),
        moves.contiguous(), next_flat.contiguous(), who_was_at.contiguous(),
        claimants.contiguous(), d_changed.contiguous(), d_all_on_goal.contiguous(),
        solve_time.contiguous(), current_step
    )
