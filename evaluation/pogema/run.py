"""Unshielded POGEMA benchmark using the shared persistent GPU worker."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

from evaluation.models import MODELS, package_precision
from evaluation.pogema.manifest import load_benchmark
from evaluation.pogema.summarize import collect
from evaluation.worker import atomic_json, execute


def parity_preflight(policy, tasks, report_path: Path) -> None:
    from pogema_gpu.toolbox.cuda_evaluator import evaluate_cuda
    from pogema_gpu.toolbox.cuda_ragged import evaluate_ragged

    ordered = sorted(tasks, key=lambda task: task.num_agents)
    selected = [replace(task, task_id=f"preflight_{i}", horizon=8)
                for i, task in enumerate((ordered[0], ordered[len(ordered) // 2], ordered[-1]))]
    reference = {}
    for task in selected:
        output = evaluate_cuda([task], copy.copy(policy), num_envs=1,
                               record_trace=False, movement_backend="resolved",
                               profile_steps=False)
        reference[task.task_id] = output["records"][0]
    packed = evaluate_ragged(selected, policy, num_envs=3,
                             max_batch_agents=policy.max_agents,
                             max_state_bytes=8 * 1024**3, reserve_bytes=16 * 1024**3,
                             record_trace=False, movement_backend="resolved")
    if packed["completed_tasks"] != len(selected):
        raise RuntimeError("POGEMA ragged preflight was incomplete")
    for record in packed["records"]:
        if any(record.get(field) != reference[record["task_id"]].get(field)
               for field in ("actions_hash", "metrics")):
            raise RuntimeError(f"POGEMA single/ragged mismatch: {record['task_id']}")
    atomic_json(report_path, {"passed": True, "shielded": False,
                              "cases": [{"task_id": t.task_id, "agents": t.num_agents}
                                        for t in selected]})


def evaluate(*, model: str, package: Path, dataset_root: Path,
             output_dir: Path, instance_root: Path | None = None) -> dict:
    tasks, sources = load_benchmark(dataset_root, instance_root=instance_root)
    package = package.resolve()
    spec = MODELS[model]
    if max(task.num_agents for task in tasks) > spec["max_num_agents"]:
        raise ValueError(f"{model} AOTI agent limit is below the benchmark maximum")
    by_key = {task.task_id: task for task in tasks}
    metadata = {
        "benchmark": "pogema", "model": model,
        "package": str(package),
        "precision": package_precision(model, package),
        "max_num_agents": spec["max_num_agents"],
        "sources": sources,
        "protocol": "seed0-dirichlet-z0-sample4-unshielded-frozen-pogema-horizons",
    }
    output_dir = output_dir.resolve()
    return execute(
        model=model, benchmark="pogema", package=package, output_dir=output_dir, metadata=metadata,
        keys=list(by_key), task_factory=lambda keys, _task_type: [by_key[key] for key in keys],
        preflight=lambda policy, _task_type, path: parity_preflight(policy, tasks, path),
        summarize=lambda: collect(tasks, output_dir, sources), shielded=False,
    )
