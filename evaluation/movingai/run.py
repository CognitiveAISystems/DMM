"""MovingAI-1600 evaluation with a fixed protocol."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

from evaluation.instances import read_instance, source_file
from evaluation.movingai.manifest import (
    load_manifest,
)
from evaluation.models import MODELS, package_precision
from evaluation.worker import atomic_json, execute


def make_tasks(rows: list[dict[str, str]], benchmark_root: Path, task_type) -> list:
    """Parse MovingAI maps and scenarios in manifest coordinate order."""
    maps = {}
    tasks = []
    for row in rows:
        map_path = source_file(benchmark_root, row["map_path"])
        scenario = source_file(benchmark_root, row["scenario_path"])
        obstacles, starts, goals = read_instance(
            map_path, scenario, int(row["num_agents"]), maps
        )
        tasks.append(task_type(
            row["base_key"],
            obstacles, starts, goals,
            policy_seed=0,
            horizon=5000,
        ))
    return tasks


def parity_preflight(policy, rows, benchmark_root, task_type, report_path: Path) -> None:
    from pogema_gpu.toolbox.cuda_evaluator import evaluate_cuda
    from pogema_gpu.toolbox.cuda_ragged import evaluate_ragged

    ordered = sorted(rows, key=lambda row: int(row["num_agents"]))
    first = ordered[0]
    medium = next(row for row in ordered if row["num_agents"] != first["num_agents"])
    selected = [first, medium, ordered[-1]]
    tasks = [
        replace(task, task_id=f"preflight_{index}", horizon=horizon)
        for index, (task, horizon) in enumerate(zip(
            make_tasks(selected, benchmark_root, task_type), (8, 16, 12)
        ))
    ]
    reference = {}
    for task in tasks:
        answer = evaluate_cuda(
            [task], copy.copy(policy), num_envs=1, record_trace=False,
            movement_backend="resolved", profile_steps=False,
        )
        reference[task.task_id] = answer["records"][0]
    result = evaluate_ragged(
        tasks, policy, num_envs=2, max_batch_agents=policy.max_agents,
        max_state_bytes=8 * 1024**3, reserve_bytes=16 * 1024**3,
        record_trace=False, movement_backend="resolved",
    )
    if result["completed_tasks"] != 3:
        raise RuntimeError("ragged preflight was incomplete")
    if not any(sample["active_envs"] == 2 for sample in result["step_samples"]):
        raise RuntimeError("ragged preflight missed parallel environments")
    if not any(item["tick"] > 0 for item in result["admissions"]):
        raise RuntimeError("ragged preflight missed refill")
    for record in result["records"]:
        for field in ("actions_hash", "metrics", "shield_statistics"):
            if record.get(field) != reference[record["task_id"]].get(field):
                raise RuntimeError(f"single/ragged mismatch: {record['task_id']} {field}")
    atomic_json(report_path, {
        "passed": True,
        "cases": [{"task_id": task.task_id, "agents": task.num_agents} for task in tasks],
        "admissions": result["admissions"],
        "evaluation_wall_seconds": result["evaluation_wall_seconds"],
    })


def evaluate(*, model: str, package: Path, output_dir: Path,
             manifest: Path, movingai_root: Path) -> dict:
    if model not in MODELS or "movingai" not in MODELS[model]["benchmarks"]:
        raise ValueError("MovingAI supports only the two MICPO checkpoints")
    manifest = manifest.resolve()
    package = package.resolve()
    benchmark_root = movingai_root.resolve()
    rows = load_manifest(manifest)
    rows_by_key = {row["base_key"]: row for row in rows}
    metadata = {
        "model": model,
        "movingai_root": str(benchmark_root),
        "package": str(package),
        "precision": package_precision(model, package),
        "protocol": "seed0-zero-z0-argmax4-pibt-sequential-rse16-5000steps",
    }
    output_dir = output_dir.resolve()
    from evaluation.movingai.summarize import collect
    return execute(
        model=model, benchmark="movingai", package=package, output_dir=output_dir,
        metadata=metadata, keys=[row["base_key"] for row in rows],
        task_factory=lambda keys, task_type: make_tasks(
            [rows_by_key[key] for key in keys], benchmark_root, task_type
        ),
        preflight=lambda policy, task_type, report_path: parity_preflight(
            policy, rows, benchmark_root, task_type, report_path
        ),
        summarize=lambda: collect(manifest, output_dir), shielded=True,
        torch_threads=8, claim_size=24,
    )
