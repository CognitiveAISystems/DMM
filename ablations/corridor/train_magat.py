"""Train MAGAT+ from scratch on the two corridor trajectories.

The architecture is MAGAT+'s production sizing rather than a shrunk variant, so
its capacity is not what limits it on this scenario. Only the optimiser budget is
cut down: the dataset is 14 graphs, so every step is full-batch.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

CORRIDOR = Path(__file__).resolve().parent
sys.path.insert(0, str(CORRIDOR / "third_party"))

from magat_plus.magat.agents import get_model  # noqa: E402

MODEL_ARGS = dict(
    obs_radius=5,
    embedding_size=128,
    num_attention_heads=1,
    cnn_mode="ResNetLarge_withMLP",
    num_gnn_layers=3,
    use_edge_weights=False,
    use_edge_attr=True,
    edge_dim=128,
    model_residuals="all",
    use_edge_attr_for_messages="positions+manhattan",
    edge_attr_processor="MLP",
    attention_mode="MAGAT_multiplicative",
    imitation_learning_model="MAGATPlus",
    scaled_product=False,
    add_data_cost_to_go=True,
    add_data_greedy_action=False,
    add_data_num_previous_actions=None,
    module_residual=None,
)


def train(data: Path, output: Path, iterations: int, seed: int, learning_rate: float,
          log_interval: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    graphs = torch.load(data, weights_only=False)
    batch = Batch.from_data_list(graphs).to(device)
    model, _ = get_model(Namespace(**MODEL_ARGS), device)
    model.train()
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.4f}M parameters, "
          f"{len(graphs)} graphs")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    for step in range(iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.x, batch)
        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % log_interval == 0:
            with torch.no_grad():
                accuracy = (logits.argmax(-1) == batch.y).float().mean().item()
            print(f"iter {step}: loss {loss.item():.4f}, train_acc {accuracy:.3f}")

    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_args": MODEL_ARGS, "iter": iterations},
               output / "ckpt_latest.pt")
    print(f"saved {output / 'ckpt_latest.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CORRIDOR / "data" / "magat_graphs.pt")
    parser.add_argument("--out-dir", type=Path, default=CORRIDOR / "runs" / "magat_corridor")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-interval", type=int, default=500)
    args = parser.parse_args()
    train(args.data, args.out_dir, args.iterations, args.seed, args.learning_rate,
          args.log_interval)


if __name__ == "__main__":
    main()
