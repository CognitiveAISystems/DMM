"""Valid joint-action frequency for every pairing of trained and executed depth.

Rounds share weights, so a checkpoint trained at one depth runs at another. The
diagonal is the matched case; the off-diagonal cells say whether the learned
coordination survives a train/test mismatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.common import checkpoints, load_dmm
from eval.frequencies import column, p_valid_per_seed, sample_tokenized
from eval.stats import format_mean_ci, mean_ci


def p_valid(entry: dict) -> tuple[float, float]:
    return mean_ci(p_valid_per_seed(entry["rows"]))


def build(trained: list[tuple[str, str]], test_rounds: list[int], samples: int,
          seed: int) -> dict:
    results = {}
    for label, pattern in trained:
        paths = checkpoints(label, pattern)
        for rounds in test_rounds:
            counts = sample_tokenized(load_dmm, paths, samples, seed, rounds)
            results[f"train={label}/test={rounds}"] = column(counts, samples)
    return {"n": samples, "seed": seed,
            "train_labels": [label for label, _ in trained],
            "test_rounds": test_rounds, "results": results}


def print_matrix(payload: dict) -> None:
    test_rounds = payload["test_rounds"]
    print("p_valid".ljust(10) + "".join(f"{'test=' + str(r):>18s}" for r in test_rounds))
    for label in payload["train_labels"]:
        cells = [format_mean_ci(*p_valid(payload["results"][f"train={label}/test={r}"]))
                 for r in test_rounds]
        print(f"train={label}".ljust(10) + "".join(f"{cell:>18s}" for cell in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", action="append", required=True, metavar="ROUNDS=GLOB",
                        help="trained depth and its per-seed checkpoints, repeatable")
    parser.add_argument("--test-rounds", default="2,4,8,12")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    trained = [tuple(spec.rsplit("=", 1)) for spec in args.ckpt]
    test_rounds = [int(value) for value in args.test_rounds.split(",")]
    payload = build(trained, test_rounds, args.n, args.seed)
    print_matrix(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as stream:
            json.dump(payload, stream, indent=2)
        print(f"\n{args.out}")


if __name__ == "__main__":
    main()
