"""Train LC-MAPF from scratch on the two corridor trajectories.

The architecture matches the DMM corridor config, so the two models differ only
in how they commit to an action, not in capacity. The dataset is 14 rows, so
every step is full-batch.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import sys

import pyarrow as pa
import torch

CORRIDOR = Path(__file__).resolve().parent
sys.path.insert(0, str(CORRIDOR / "third_party"))

from lc_mapf.model import LCMAPF, Config  # noqa: E402

ARCHITECTURE = dict(
    block_size=256,
    field_of_view_size=121,
    agent_info_size=10,
    max_num_neighbors=13,
    n_encoder_layer=3,
    n_decoder_layer=3,
    n_head=3,
    n_embd=64 * 3,
    latent_embd=32 * 3,
    latent_tok_n=32,
    action_msg_feats=32 * 3,
    n_comm_rounds=4,
)


def load_dataset(path: Path, device: str):
    with pa.memory_map(str(path)) as source:
        table = pa.ipc.open_file(source).read_all()
    observations = torch.tensor(table["input_tensors"].to_pylist(), dtype=torch.int,
                                device=device)
    neighbors = torch.tensor(table["agents_in_obs"].to_pylist(), dtype=torch.int,
                             device=device)
    actions = torch.tensor(table["gt_actions"].to_pylist(), dtype=torch.long, device=device)
    return observations, neighbors, actions


def train(data: Path, output: Path, iterations: int, seed: int, learning_rate: float,
          log_interval: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    observations, neighbors, actions = load_dataset(data, device)
    allowed = {field.name for field in fields(Config)}
    model = LCMAPF(Config(**{k: v for k, v in ARCHITECTURE.items() if k in allowed}))
    model.to(device).train()
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.4f}M parameters, "
          f"{observations.shape[0]} rows")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    for step in range(iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(observations, neighbors, actions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % log_interval == 0:
            with torch.no_grad():
                accuracy = (logits.argmax(-1) == actions.view(-1)).float().mean().item()
            print(f"iter {step}: loss {loss.item():.4f}, train_acc {accuracy:.3f}")

    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_args": ARCHITECTURE, "iter": iterations},
               output / "ckpt_latest.pt")
    print(f"saved {output / 'ckpt_latest.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CORRIDOR / "data" / "part_0_0.arrow")
    parser.add_argument("--out-dir", type=Path, default=CORRIDOR / "runs" / "lcmapf_corridor")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-interval", type=int, default=500)
    args = parser.parse_args()
    train(args.data, args.out_dir, args.iterations, args.seed, args.learning_rate,
          args.log_interval)


if __name__ == "__main__":
    main()
