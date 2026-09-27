"""Validate POGEMA task coverage and report quality separately from wall time."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from contextlib import closing


def collect(tasks, output_dir: Path, sources: dict) -> dict:
    database = Path(output_dir) / "tasks.sqlite3"
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
        payload, signature = connection.execute(
            "SELECT payload,signature FROM metadata WHERE id=1"
        ).fetchone()
        metadata = json.loads(payload)
        if metadata.get("sources") != sources or metadata.get("benchmark") != "pogema":
            raise ValueError("queue belongs to a different benchmark")
        rows = connection.execute("SELECT key,state FROM tasks ORDER BY ordinal").fetchall()
    if [key for key, _ in rows] != [task.task_id for task in tasks]:
        raise ValueError("queue task identities do not match the benchmark")
    records_dir = Path(output_dir) / "records"
    if records_dir.exists():
        extra = {
            path.parent.relative_to(records_dir).as_posix()
            for path in records_dir.rglob("summary.json")
        } - {key for key, _ in rows}
        if extra:
            raise ValueError(f"unassigned result artifacts: {sorted(extra)[:5]}")
    counts = {name: 0 for name in ("solved", "no_solution", "error", "pending", "running")}
    soc, makespan = [], []
    for key, state in rows:
        if state not in counts:
            raise ValueError(f"invalid task state: {key} {state}")
        counts[state] += 1
        if state in {"solved", "no_solution"}:
            folder = records_dir / key
            if not (folder / "episode.json.gz").is_file():
                raise ValueError(f"missing episode artifact: {key}")
            record = json.loads((folder / "summary.json").read_text())
            if (record.get("base_key"), record.get("signature"), record.get("status")) != (key, signature, state):
                raise ValueError(f"result does not match queue: {key}")
            if state == "solved":
                soc.append(int(record["soc"]))
                makespan.append(int(record["makespan"]))
    attempts = []
    for path in sorted((Path(output_dir) / "attempts").glob("*.json")):
        if len(path.stem) != 32 or any(char not in "0123456789abcdef" for char in path.stem):
            continue
        item = json.loads(path.read_text())
        attempts.append({key: item.get(key) for key in (
            "owner", "gpu", "pid", "finished", "model_initialization_seconds",
            "evaluation_wall_seconds_sum", "process_wall_seconds"
        )})
    return {"model": metadata["model"], "expected": len(tasks),
            "completed": counts["solved"] + counts["no_solution"],
            "solved": counts["solved"], "no_solution": counts["no_solution"],
            "errors": counts["error"], "pending": counts["pending"],
            "running": counts["running"],
            "success_rate": counts["solved"] / len(tasks),
            "mean_soc_solved": sum(soc) / len(soc) if soc else None,
            "mean_makespan_solved": sum(makespan) / len(makespan) if makespan else None,
            "timing_by_attempt": attempts,
            "timing_note": "GPU evaluator wall and process wall are not per-task solver runtime"}
