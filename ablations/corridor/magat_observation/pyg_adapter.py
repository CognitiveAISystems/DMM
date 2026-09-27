from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


MAGAT_PLUS_COST_TO_GO_SCALE = 100.0


def native_magat_plus_to_pyg(
    node_features,
    adjacency,
    agent_positions,
    *,
    use_edge_attr: bool = True,
    edge_attr_for_messages: str | None = "positions+manhattan",
) -> Data:
    """Convert one native MAGAT+ observation into a PyG graph."""
    x = torch.tensor(np.asarray(node_features), dtype=torch.float32)
    if x.ndim != 4 or x.shape[1] != 4:
        raise ValueError(
            "MAGAT+ node features must have shape [agent, 4, height, width]"
        )
    x[:, 3].div_(MAGAT_PLUS_COST_TO_GO_SCALE)

    dense_adjacency = torch.tensor(np.asarray(adjacency), dtype=torch.float32)
    edge_index, edge_weight = dense_to_sparse(dense_adjacency)
    positions = torch.tensor(np.asarray(agent_positions), dtype=torch.float32)

    edge_attr = None
    if use_edge_attr:
        if edge_attr_for_messages is None:
            raise ValueError(
                "edge_attr_for_messages is required when use_edge_attr=True"
            )
        position_difference = positions[edge_index[0]] - positions[edge_index[1]]
        if edge_attr_for_messages == "positions":
            edge_attr = position_difference
        elif edge_attr_for_messages == "dist":
            edge_attr = torch.norm(position_difference, keepdim=True, dim=-1)
        elif edge_attr_for_messages == "manhattan":
            edge_attr = torch.sum(
                torch.abs(position_difference), dim=-1, keepdim=True
            )
        elif edge_attr_for_messages == "positions+dist":
            distance = torch.norm(position_difference, keepdim=True, dim=-1)
            edge_attr = torch.cat([position_difference, distance], dim=-1)
        elif edge_attr_for_messages == "positions+manhattan":
            manhattan = torch.sum(
                torch.abs(position_difference), dim=-1, keepdim=True
            )
            edge_attr = torch.cat([position_difference, manhattan], dim=-1)
        else:
            raise ValueError(
                f"Unsupported edge_attr_for_messages={edge_attr_for_messages!r}"
            )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_attr=edge_attr,
        agent_positions=positions,
    )

