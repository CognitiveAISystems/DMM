"""Fixed observation contract of the million-agent evaluation."""

from dataclasses import dataclass


@dataclass
class MillionConfig:
    device: str = "cuda"
    n_comm_rounds: int = 4
    num_previous_actions: int = 5
    cost2go_value_limit: int = 20
    agents_radius: int = 5
    cost2go_radius: int = 5
    context_size: int = 256
    max_num_neighbors: int = 13
    max_horizon: int = 32768
    use_spatial_neighbors: bool = True
    bfs_chunk_size: int = 2048
    agent_chunk_size: int = 32768
    use_fp16: bool = True

class _ObsGenConfig:
    def __init__(self, config: MillionConfig):
        for name in (
            "device", "cost2go_radius", "cost2go_value_limit",
            "agents_radius", "num_previous_actions", "context_size",
            "use_spatial_neighbors", "bfs_chunk_size",
        ):
            setattr(self, name, getattr(config, name))
