"""Generate the two corridor LaTeX tables as JSON, so their numbers trace to this data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from corridor_stats import action_mean_ci, canonical, invalid_mean_ci, load, p_valid_mean_ci

ACTIONS = {"WL": "(wait, left)", "RW": "(right, wait)",
           "WW": "(wait, wait)", "RL": "(right, left)"}
BASELINE_METHODS = ("DMM (tf=0)", "LC-MAPF", "MAGAT+", "HMAGAT")


def pct(value: tuple[float, float]) -> dict:
    mean, ci = value
    return {"mean": round(100 * mean, 1), "ci": round(100 * ci, 1)}


def joint_action_row(entry: dict, method: str, source: str) -> dict:
    row = {"method": method, "source": source}
    row.update({name: pct(action_mean_ci(entry, action)) for name, action in ACTIONS.items()})
    return row


def corridor_freq(data_dir: Path) -> dict:
    baseline = load(data_dir / "baseline.json")
    rows = [joint_action_row(canonical(data_dir), "DMM (beta=0.8)",
                             "round_generalization_matrix.json train=4/test=4")]
    rows += [joint_action_row(baseline["results"][method], method, "baseline.json")
             for method in BASELINE_METHODS]
    return {"table": "corridor_freq", "paper_label": "tab:corridor_freq", "unit": "percent",
            "precision_decimals": 1, "n_per_seed": baseline["n"], "k_seeds": 5, "rows": rows}


def corridor_depth_composition(data_dir: Path) -> dict:
    matrix = load(data_dir / "round_generalization_matrix.json")
    rows = []
    for k_test in matrix["test_rounds"]:
        entry = matrix["results"][f"train=4/test={k_test}"]
        rows.append({
            "k_test": k_test,
            "p_valid": pct(p_valid_mean_ci(entry)),
            "WL": round(100 * action_mean_ci(entry, ACTIONS["WL"])[0], 1),
            "RW": round(100 * action_mean_ci(entry, ACTIONS["RW"])[0], 1),
            "Invalid": pct(invalid_mean_ci(entry)),
        })
    return {"table": "corridor_depth_composition",
            "paper_label": "tab:corridor-depth-composition", "unit": "percent",
            "precision_decimals": 1, "k_train": 4, "n_per_seed": matrix["n"],
            "k_seeds": 5, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("tables"))
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("corridor_freq", corridor_freq(data_dir)),
                        ("corridor_depth_composition", corridor_depth_composition(data_dir))):
        target = output_dir / f"{name}.json"
        with target.open("w") as stream:
            json.dump(table, stream, indent=2)
        print(target)


if __name__ == "__main__":
    main()
