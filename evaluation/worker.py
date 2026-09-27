"""Shared single-GPU worker for POGEMA and MovingAI."""

from __future__ import annotations

import gzip
import inspect
import json
import os
from pathlib import Path
import time

from evaluation.queue import LocalTaskQueue


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_record(output_dir: Path, signature: str, policy, record: dict) -> str:
    key = record["task_id"]
    folder = output_dir / "records" / key
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "summary.json").exists():
        raise ValueError(f"refusing to replace an accepted result: {key}")
    temporary = folder / "episode.json.gz.tmp"
    with gzip.open(temporary, "wt", compresslevel=1) as stream:
        json.dump(record, stream, separators=(",", ":"))
    temporary.replace(folder / "episode.json.gz")
    metrics = record["metrics"]
    solved = bool(metrics["CSR"])
    status = "solved" if solved else "no_solution"
    atomic_json(folder / "summary.json", {
        "base_key": key,
        "signature": signature,
        "status": status,
        "solved": solved,
        "soc": int(metrics["SoC"]) if solved else None,
        "makespan": int(metrics["makespan"]) if solved else None,
        "runtime_sec": None,
        "environment_latency_sec": record.get("environment_latency_seconds"),
        "actions_hash": record["actions_hash"],
        "metrics": metrics,
        "shield_statistics": record.get("shield_statistics"),
        "policy": policy.metadata,
        "timing_note": "shared packed inference; no per-task solver runtime",
    })
    return status


def execute(*, model: str, benchmark: str, package: Path, output_dir: Path, metadata: dict,
            keys: list[str], task_factory, preflight, summarize,
            shielded: bool, torch_threads: int = 8, claim_size: int = 24) -> dict:
    """Resume a local task pool; one invocation owns exactly one visible GPU."""
    if torch_threads < 1 or claim_size < 1:
        raise ValueError("torch_threads and claim_size must be positive")
    from evaluation.models import policy_for

    queue = LocalTaskQueue(output_dir)
    signature = queue.initialize(metadata, keys)
    if queue.status()["completed"] == len(keys):
        return summarize()

    with queue.worker() as owner:
        import torch
        from pogema_gpu.kernels import extension
        from pogema_gpu.kernels.cuda_bfs import load_cuda_bfs
        from pogema_gpu.tasks import Task
        from pogema_gpu.toolbox.cuda_ragged import evaluate_ragged

        if str(torch.__version__) != "2.13.0+cu126":
            raise RuntimeError(f"PyTorch 2.13.0+cu126 required: {torch.__version__}")
        if "retain_records" not in inspect.signature(evaluate_ragged).parameters:
            raise RuntimeError("POGEMA-GPU evaluate_ragged lacks retain_records support")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("expose exactly one GPU to each evaluator process")
        torch.set_num_threads(torch_threads)
        started = time.perf_counter()
        extension()
        if shielded:
            extension("pibt")
        load_cuda_bfs()
        policy = policy_for(model, package, benchmark=benchmark, shielded=shielded)
        initialization_seconds = time.perf_counter() - started
        attempt_dir = output_dir / "attempts"
        attempt_path = attempt_dir / f"{owner}.json"
        if preflight is not None:
            preflight(policy, Task, attempt_dir / f"{owner}-preflight.json")
        attempt = {
            "owner": owner, "pid": os.getpid(), "gpu": torch.cuda.get_device_name(0),
            "model_initialization_seconds": initialization_seconds,
            "evaluation_wall_seconds_sum": 0.0, "batches": 0,
            "tasks_completed": 0, "finished": False,
        }
        atomic_json(attempt_path, attempt)
        print(json.dumps({"event": "evaluation_started", "pid": os.getpid(),
                          "owner": owner, "queue": queue.status(),
                          "initialization_seconds": initialization_seconds,
                          "gpu": torch.cuda.get_device_name(0)}), flush=True)
        try:
            while True:
                claimed = queue.claim(owner, signature, claim_size)
                if not claimed:
                    state = queue.status()
                    if not state["pending"] and not state["running"]:
                        break
                    time.sleep(5)
                    continue
                tasks = task_factory(claimed, Task)

                def on_result(record: dict, count: int, total: int) -> None:
                    key = record["task_id"]
                    status = save_record(output_dir, signature, policy, record)
                    queue.complete(owner, key, status)
                    attempt["tasks_completed"] += 1
                    print(json.dumps({"event": "task_completed", "task_id": key,
                                      "completed_in_batch": count,
                                      "remaining_in_batch": total - count}), flush=True)

                result = evaluate_ragged(
                    tasks, policy, num_envs=256, max_batch_agents=policy.max_agents,
                    max_state_bytes=8 * 1024**3, reserve_bytes=16 * 1024**3,
                    progress=on_result, record_trace=False, movement_backend="resolved",
                    retain_records=False,
                )
                attempt["batches"] += 1
                attempt["evaluation_wall_seconds_sum"] += result["evaluation_wall_seconds"]
                attempt["process_wall_seconds"] = time.perf_counter() - started
                atomic_json(attempt_dir / f"{owner}-batch-{attempt['batches']:04d}.json", result)
                atomic_json(attempt_path, attempt)
        except Exception as exc:
            queue.fail_claimed(owner, signature, str(exc))
            attempt["error"] = str(exc)
            attempt["process_wall_seconds"] = time.perf_counter() - started
            atomic_json(attempt_path, attempt)
            raise
        attempt["finished"] = True
        attempt["process_wall_seconds"] = time.perf_counter() - started
        atomic_json(attempt_path, attempt)
        report = summarize()
        if report["errors"]:
            raise RuntimeError(f"evaluation has {report['errors']} failed tasks")
        return report
