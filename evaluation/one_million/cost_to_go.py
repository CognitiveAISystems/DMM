"""GPU-resident exact cost-to-go cache for standalone planners."""

from __future__ import annotations

import torch


_MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_HASH_MASK = 0xFFFF_FFFF
_TIE_MASK = 0x7FFF
_DISTANCE_STRIDE = 1 << 16


def _candidate_hashes(
    global_agent_ids: torch.Tensor,
    *,
    step: int,
    seed: int,
) -> torch.Tensor:
    """Return independently mixed deterministic keys for five candidates."""
    action_ids = torch.arange(
        len(_MOVES), dtype=torch.int64, device=global_agent_ids.device
    ).view(1, -1)
    values = (
        global_agent_ids.to(torch.int64).view(-1, 1) * 0x9E37_79B1
        + action_ids * 0x85EB_CA77
        + ((int(step) + 1) & _HASH_MASK) * 0xC2B2_AE3D
        + (int(seed) & _HASH_MASK) * 0x27D4_EB2F
    ).bitwise_and(_HASH_MASK)

    # Thomas Wang's 32-bit integer mixer.  Masking after each multiplication
    # keeps CPU and CUDA behavior identical without relying on signed overflow.
    values = (
        ((values ^ torch.bitwise_right_shift(values, 16)) * 0x045D_9F3B)
        .bitwise_and(_HASH_MASK)
    )
    values = (
        ((values ^ torch.bitwise_right_shift(values, 16)) * 0x045D_9F3B)
        .bitwise_and(_HASH_MASK)
    )
    return (values ^ torch.bitwise_right_shift(values, 16)).bitwise_and(
        _HASH_MASK
    )


def distance_scores(
    distances: torch.Tensor,
    global_agent_ids: torch.Tensor,
    *,
    step: int,
    seed: int,
) -> torch.Tensor:
    """Encode exact distance and well-mixed deterministic PIBT tie keys.

    Scores are signed int32 lexicographic keys.  Exact distance is the strict
    primary key, bit 15 is reserved for the paper's free-cell preference, and
    the low 15 bits provide a counter-hashed random candidate order.  Integer
    packing avoids losing tie information when distances become large.
    """
    if distances.ndim != 2 or distances.shape[1] != len(_MOVES):
        raise ValueError(f"distances must be [N,5], got {tuple(distances.shape)}")
    if global_agent_ids.shape != distances.shape[:1]:
        raise ValueError(
            "global_agent_ids must contain one id per distance row, got "
            f"{tuple(global_agent_ids.shape)} for {tuple(distances.shape)}"
        )

    tie_keys = _candidate_hashes(
        global_agent_ids,
        step=step,
        seed=seed,
    ).bitwise_and(_TIE_MASK)
    scores = (
        -distances.to(torch.int64) * _DISTANCE_STRIDE + tie_keys
    ).to(torch.int32)
    return scores.masked_fill(
        distances.lt(0), torch.iinfo(torch.int32).min
    )


class GPUCostToGoCache:
    """Own exact local BFS windows for a contiguous agent shard."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        grid,
        positions,
        goals,
        device: torch.device | str,
        cache_radius: int,
        bfs_chunk_size: int,
        shard_start: int = 0,
        shard_end: int | None = None,
    ) -> None:
        if cache_radius < 1:
            raise ValueError("cache_radius must be at least 1")
        if bfs_chunk_size < 1:
            raise ValueError("bfs_chunk_size must be positive")

        from pogema_gpu.kernels.cuda_bfs import is_available

        if not is_available():
            raise RuntimeError(
                "CUDA BFS kernel failed to compile or pass its smoke test"
            )

        self.width = int(width)
        self.height = int(height)
        self.device = torch.device(device)
        self.cache_radius = int(cache_radius)
        self.bfs_chunk_size = int(bfs_chunk_size)
        self.shard_start = int(shard_start)
        total_agents = len(positions)
        self.shard_end = total_agents if shard_end is None else int(shard_end)
        if not 0 <= self.shard_start <= self.shard_end <= total_agents:
            raise ValueError("invalid contiguous agent shard")

        self.obstacles = torch.as_tensor(
            grid, dtype=torch.bool, device=self.device
        ).reshape(self.height, self.width)
        all_positions = torch.as_tensor(positions)
        all_goals = torch.as_tensor(goals)
        self.goals = torch.as_tensor(
            all_goals[self.shard_start : self.shard_end],
            dtype=torch.long,
            device=self.device,
        ).contiguous()
        local_positions = torch.as_tensor(
            all_positions[self.shard_start : self.shard_end],
            dtype=torch.long,
            device=self.device,
        ).contiguous()

        local_agents = self.shard_end - self.shard_start
        side = 2 * self.cache_radius + 1
        self.cache_windows = torch.full(
            (local_agents, side * side),
            -1,
            dtype=torch.int16,
            device=self.device,
        )
        self.cache_centers = local_positions.clone()
        self.cache_valid = torch.zeros(
            local_agents, dtype=torch.bool, device=self.device
        )
        self.global_agent_ids = torch.arange(
            self.shard_start,
            self.shard_end,
            dtype=torch.int64,
            device=self.device,
        )
        self.last_refill_count = 0

    @torch.no_grad()
    def candidate_distances(self, positions: torch.Tensor) -> torch.Tensor:
        """Return exact distances for wait/up/down/left/right as ``[N,5]``."""
        positions = torch.as_tensor(
            positions, dtype=torch.long, device=self.device
        ).contiguous()
        if positions.shape != self.goals.shape:
            raise ValueError(
                f"local positions must be {tuple(self.goals.shape)}, got "
                f"{tuple(positions.shape)}"
            )

        # A one-cell candidate stencil must remain inside the cached window.
        margin = self.cache_radius - 1
        delta = (positions - self.cache_centers).abs()
        valid = self.cache_valid & delta.max(dim=1).values.le(margin)
        misses = (~valid).nonzero(as_tuple=True)[0]
        self.last_refill_count = int(misses.numel())
        if misses.numel():
            from pogema_gpu.kernels.cuda_bfs import raw_bfs_cost2go

            for start in range(0, misses.numel(), self.bfs_chunk_size):
                batch = misses[start : start + self.bfs_chunk_size]
                self.cache_windows[batch] = raw_bfs_cost2go(
                    self.obstacles,
                    positions[batch],
                    self.goals[batch],
                    self.height,
                    self.width,
                    self.cache_radius,
                )
                self.cache_centers[batch] = positions[batch]
                self.cache_valid[batch] = True

        side = 2 * self.cache_radius + 1
        offset = positions - self.cache_centers
        center = (
            (self.cache_radius + offset[:, 0]) * side
            + self.cache_radius
            + offset[:, 1]
        )
        candidate_offsets = torch.tensor(
            (0, -side, side, -1, 1),
            dtype=torch.long,
            device=self.device,
        )
        indices = center[:, None] + candidate_offsets[None, :]
        return self.cache_windows.gather(1, indices).contiguous()

    @torch.no_grad()
    def scores(self, positions: torch.Tensor, *, step: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return distance-ranked PIBT scores and their exact distances."""
        distances = self.candidate_distances(positions)
        return (
            distance_scores(
                distances,
                self.global_agent_ids,
                step=step,
                seed=seed,
            ),
            distances,
        )
