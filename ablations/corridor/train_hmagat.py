"""Train HMAGAT from scratch on the two corridor trajectories.

The flags match dataset_hmagat.py, since the graphs are built for that exact
configuration, and the upstream repository fetched by `setup.sh` supplies the
network. As with the other baselines, only the optimiser budget is cut down.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from dataset_hmagat import build_args, upstream

CORRIDOR = Path(__file__).resolve().parent


def train(data: Path, output: Path, iterations: int, seed: int, learning_rate: float,
          log_interval: int) -> None:
    upstream()
    from hmagat.modules.agents import get_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    args = build_args()
    args.model_seed = seed
    graphs = torch.load(data, weights_only=False)
    batch = Batch.from_data_list(graphs).to(device)
    model, _, _ = get_model(args, device)
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
    torch.save({"model": model.state_dict(), "model_args": vars(args), "iter": iterations},
               output / "ckpt_latest.pt")
    print(f"saved {output / 'ckpt_latest.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CORRIDOR / "data" / "hmagat_graphs.pt")
    parser.add_argument("--out-dir", type=Path, default=CORRIDOR / "runs" / "hmagat_corridor")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-interval", type=int, default=500)
    args = parser.parse_args()
    train(args.data, args.out_dir, args.iterations, args.seed, args.learning_rate,
          args.log_interval)


if __name__ == "__main__":
    main()
