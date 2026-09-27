"""Validate and summarize a standalone MovingAI run from its shared task queue."""

from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from evaluation.movingai.manifest import load_manifest


def collect(manifest: Path, output_dir: Path) -> dict:
    rows = load_manifest(manifest)
    database = Path(output_dir) / "tasks.sqlite3"
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
        metadata_json, signature = connection.execute(
            "SELECT payload,signature FROM metadata WHERE id=1"
        ).fetchone()
        metadata = json.loads(metadata_json)
        tasks = connection.execute(
            "SELECT key,state FROM tasks ORDER BY ordinal"
        ).fetchall()
    if [key for key, _ in tasks] != [row["base_key"] for row in rows]:
        raise ValueError("queue task identities do not match the manifest")
    records_dir = Path(output_dir) / "records"
    if records_dir.exists():
        unassigned = {path.name for path in records_dir.iterdir()} - {
            key for key, _ in tasks
        }
        if unassigned:
            raise ValueError(f"unassigned result artifacts: {sorted(unassigned)[:5]}")

    counts = {"solved": 0, "no_solution": 0, "error": 0, "pending": 0, "running": 0}
    soc = []
    makespans = []
    for key, state in tasks:
        if state not in counts:
            raise ValueError(f"invalid task state: {key} {state}")
        counts[state] += 1
        if state not in {"solved", "no_solution"}:
            continue
        folder = records_dir / key
        if not (folder / "episode.json.gz").is_file():
            raise ValueError(f"missing episode artifact: {key}")
        record = json.loads((folder / "summary.json").read_text())
        if (
            record.get("base_key") != key
            or record.get("signature") != signature
            or record.get("status") != state
        ):
            raise ValueError(f"result does not match queue: {key}")
        if state == "solved":
            soc.append(int(record["soc"]))
            makespans.append(int(record["makespan"]))

    timings = []
    attempt_dir = Path(output_dir) / "attempts"
    if attempt_dir.exists():
        for path in sorted(attempt_dir.glob("*.json")):
            if len(path.stem) != 32 or any(char not in "0123456789abcdef" for char in path.stem):
                continue
            attempt = json.loads(path.read_text())
            timings.append({
                "owner": attempt["owner"],
                "gpu": attempt["gpu"],
                "pid": attempt["pid"],
                "finished": attempt["finished"],
                "model_initialization_seconds": attempt["model_initialization_seconds"],
                "evaluation_wall_seconds_sum": attempt["evaluation_wall_seconds_sum"],
                "process_wall_seconds": attempt.get("process_wall_seconds"),
            })
    return {
        "model": metadata["model"],
        "expected": len(rows),
        "completed": counts["solved"] + counts["no_solution"],
        "solved": counts["solved"],
        "no_solution": counts["no_solution"],
        "errors": counts["error"],
        "pending": counts["pending"],
        "running": counts["running"],
        "success_rate": counts["solved"] / len(rows),
        "mean_soc_solved": sum(soc) / len(soc) if soc else None,
        "mean_makespan_solved": sum(makespans) / len(makespans) if makespans else None,
        "sum_soc_solved": sum(soc),
        "timing_by_attempt": timings,
        "timing_note": (
            "GPU evaluator wall is summed over completed batches; process wall includes "
            "initialization and queue waiting. Interrupted attempts may have unrecorded "
            "time after their last completed batch. Neither is per-task solver runtime."
        ),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(collect(args.manifest, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
