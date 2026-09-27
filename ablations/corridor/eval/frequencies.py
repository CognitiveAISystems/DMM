"""Sample joint actions at the ambiguous corridor state and report their frequencies.

Each column is one configuration, backed by a glob matching one checkpoint per
training seed. A column's frequency is the mean across seeds with a 95% Student-t
interval, so it reflects training variance rather than sampling noise alone.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.common import (ambiguous_graph, ambiguous_tokens, checkpoints, load_dmm,
                         load_hmagat, load_lcmapf, load_magat)
from eval.stats import format_mean_ci, mean_ci
from scenario import DOWN, JOINT_ACTION_LABELS, LEFT, RIGHT, UP, VALID_JOINT_ACTIONS, WAIT

JOINT_ACTIONS = tuple(JOINT_ACTION_LABELS)
ACTION_TEXT = {WAIT: "wait", UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}


def action_key(joint_action) -> str:
    """The key the published data files use for one joint action."""
    return f"({ACTION_TEXT[joint_action[0]]}, {ACTION_TEXT[joint_action[1]]})"


def p_valid_per_seed(rows: dict) -> list[float]:
    """Each seed's mass on the coordinated resolutions, summed before averaging."""
    columns = [rows[action_key(action)]["per_seed_freq"] for action in VALID_JOINT_ACTIONS]
    return [sum(seed) for seed in zip(*columns)]


@torch.no_grad()
def sample_tokenized(loader, paths, samples, seed, rounds=None):
    """DMM and LC-MAPF both sample a joint action per forward pass."""
    observation, neighbors = ambiguous_tokens()
    counts = []
    for path in paths:
        torch.manual_seed(seed)
        model = loader(Path(path), rounds) if rounds is not None else loader(Path(path))
        drawn = Counter()
        for _ in range(samples):
            actions = model.act(observation, neighbors, do_sample=True)
            drawn[tuple(actions.view(-1).tolist())] += 1
        counts.append(drawn)
    return counts


@torch.no_grad()
def sample_graph(loader, name, paths, samples, seed, temperature=1.0):
    """MAGAT+ and HMAGAT emit per-agent logits once; the draws are independent."""
    graph = ambiguous_graph(name)
    counts = []
    for path in paths:
        torch.manual_seed(seed)
        model = loader(Path(path))
        probabilities = F.softmax(model(graph.x, graph) / temperature, dim=-1)
        distribution = torch.distributions.Categorical(probs=probabilities)
        drawn = Counter()
        for _ in range(samples):
            drawn[tuple(distribution.sample().tolist())] += 1
        counts.append(drawn)
    return counts


def column(counts_by_seed, samples: int) -> dict:
    rows = {}
    for joint_action, label in JOINT_ACTION_LABELS.items():
        per_seed = [counts[joint_action] / samples for counts in counts_by_seed]
        mean, half_width = mean_ci(per_seed)
        rows[action_key(joint_action)] = {
            "label": label, "per_seed_freq": per_seed,
            "mean": mean, "ci_half_width": half_width,
        }
    valid = [sum(counts[action] for action in VALID_JOINT_ACTIONS) / samples
             for counts in counts_by_seed]
    mean, half_width = mean_ci(valid)
    cell = {"k_seeds": len(counts_by_seed), "rows": rows,
            "valid_rate": {"per_seed": valid, "mean": mean, "ci_half_width": half_width}}

    # Joint actions outside the four are reported rather than silently dropped.
    other: dict[str, float] = {}
    for counts in counts_by_seed:
        for joint_action, drawn in counts.items():
            if joint_action not in JOINT_ACTION_LABELS:
                key = action_key(joint_action)
                other[key] = other.get(key, 0.0) + drawn / samples / len(counts_by_seed)
    if other:
        cell["other"] = other
    return cell


SAMPLERS = {
    "dmm": lambda paths, n, seed, rounds: sample_tokenized(load_dmm, paths, n, seed, rounds),
    "lcmapf": lambda paths, n, seed, rounds: sample_tokenized(load_lcmapf, paths, n, seed),
    "magat": lambda paths, n, seed, rounds: sample_graph(load_magat, "magat", paths, n, seed),
    "hmagat": lambda paths, n, seed, rounds: sample_graph(load_hmagat, "hmagat", paths, n, seed),
}


def collect(requested, samples, seed, rounds=None) -> dict:
    results = {}
    for family, label, pattern in requested:
        paths = checkpoints(label, pattern)
        counts = SAMPLERS[family](paths, samples, seed, rounds)
        results[label] = column(counts, samples)
    return {"n": samples, "seed": seed, "columns": [label for _, label, _ in requested],
            "results": results}


def print_table(payload: dict) -> None:
    labels = payload["columns"]
    width = max(len(label) for label in JOINT_ACTION_LABELS.values())
    print(f"{'joint action'.ljust(width)}  " + "  ".join(f"{label:>18s}" for label in labels))
    for joint_action in JOINT_ACTIONS:
        key = action_key(joint_action)
        cells = []
        for label in labels:
            row = payload["results"][label]["rows"][key]
            cells.append(f"{format_mean_ci(row['mean'], row['ci_half_width']):>18s}")
        print(f"{JOINT_ACTION_LABELS[joint_action].ljust(width)}  " + "  ".join(cells))
    print()
    valid = []
    for label in labels:
        per_seed = p_valid_per_seed(payload["results"][label]["rows"])
        valid.append(f"{format_mean_ci(*mean_ci(per_seed)):>18s}")
    print("p_valid".ljust(width) + "  " + "  ".join(valid))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for family in SAMPLERS:
        parser.add_argument(f"--{family}", action="append", default=[], metavar="LABEL=GLOB",
                            help=f"{family} column, one checkpoint per seed")
    parser.add_argument("--n", type=int, default=1000, help="joint-action samples per checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed, reset per checkpoint")
    parser.add_argument("--rounds", type=int, help="override the DMM refinement depth")
    parser.add_argument("--out", type=Path, help="also write the table as JSON")
    args = parser.parse_args()

    # Split on the last '=' so column labels may contain one, as "DMM (tf=0.8)" does.
    requested = [(family, *spec.rsplit("=", 1))
                 for family in SAMPLERS for spec in getattr(args, family)]
    if not requested:
        parser.error("give at least one column, "
                     "e.g. --dmm 'tf=runs/corridor_tf08_seed*/ckpt_latest.pt'")

    payload = collect(requested, args.n, args.seed, args.rounds)
    print_table(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as stream:
            json.dump(payload, stream, indent=2)
        print(f"\n{args.out}")


if __name__ == "__main__":
    main()
