"""Run standalone GPU-PIBT on a nested large-agent scenario.

Exact CUDA BFS supplies each agent's candidate ordering.  The existing exact
CUDA PIBT implementation resolves the joint action.  No learned model is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist

from evaluation.one_million.cost_to_go import GPUCostToGoCache
from evaluation.one_million.scenario import load_master, master_prefix
from evaluation.one_million.resume import (EvaluationState, Trajectory, add_resume_arguments,
                                           rebuild_cache, restore_tensor)
from evaluation.one_million.pibt import BatchedPIBT


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def timing_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"total_ms": 0.0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    count = len(ordered)
    middle = count // 2
    median = (
        ordered[middle]
        if count % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    p95 = ordered[min(count - 1, max(0, int(0.95 * count + 0.999999) - 1))]
    return {
        "total_ms": float(sum(ordered)),
        "mean_ms": float(sum(ordered) / count),
        "median_ms": float(median),
        "p95_ms": float(p95),
    }


def code_state() -> tuple[str | None, str | None]:
    """Record source revision when available, including anonymous archives."""
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None
    return git("branch", "--show-current"), git("rev-parse", "HEAD")


def compact_completion_record(result: dict, output: str | Path) -> dict:
    steady = result["timing_steady"]
    return {
        "event": "completed",
        "algorithm": "GPU-PIBT",
        "output": str(output),
        "agents": result["agents"],
        "steps": result["steps"],
        "solved": result["solved"],
        "isr": result["isr"],
        "wall_seconds": result["wall_seconds"],
        "mean_step_seconds": result["mean_step_seconds"],
        "pibt_changes": result["pibt_overrides"],
        "steady_mean_ms": {
            key: steady[key]["mean_ms"]
            for key in (
                "distance_preferences",
                "preferences_allgather",
                "pibt_and_broadcast",
                "gpu_total",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-master", required=True)
    parser.add_argument("--agents", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=4096)
    parser.add_argument("--bfs-chunk", type=int, default=2048)
    parser.add_argument("--cache-radius", type=int, default=70)
    parser.add_argument("--preference-seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pibt-resolver",
        choices=(
            "sequential", "parallel_occupancy", "compact", "components",
            "hybrid", "warp",
        ),
        default="components",
    )
    parser.add_argument(
        "--trajectory-actions",
        help="optional gzip-compressed uint8 executed-action stream",
    )
    add_resume_arguments(parser)
    args = parser.parse_args()

    if args.agents < 1 or args.horizon < 1:
        raise ValueError("agents and horizon must be positive")

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.agents % world:
        raise ValueError("--agents must be divisible by the distributed world size")
    local_agents = args.agents // world
    shard_start, shard_end = rank * local_agents, (rank + 1) * local_agents
    branch, commit = code_state()
    state = EvaluationState(args, rank, world, device, 'GPU-PIBT')

    master = load_master(args.scenario_master)
    scenario = master_prefix(master, args.agents)
    global_positions = torch.as_tensor(
        scenario.positions, dtype=torch.long, device=device
    ).contiguous()
    global_goals = torch.as_tensor(
        scenario.goals, dtype=torch.long, device=device
    ).contiguous()
    cost_to_go = GPUCostToGoCache(
        width=scenario.width,
        height=scenario.height,
        grid=scenario.grid,
        positions=scenario.positions,
        goals=scenario.goals,
        device=device,
        cache_radius=args.cache_radius,
        bfs_chunk_size=args.bfs_chunk,
        shard_start=shard_start,
        shard_end=shard_end,
    )
    pibt = (
        BatchedPIBT(
            [scenario],
            device,
            priority_mode="manhattan_gpu",
            resolver_mode=args.pibt_resolver,
        )
        if rank == 0
        else None
    )

    def gather_local(local: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            (args.agents, *local.shape[1:]), dtype=local.dtype, device=device
        )
        dist.all_gather_into_tensor(output, local.contiguous())
        return output

    first_arrival = (
        torch.full((args.agents,), -1, dtype=torch.int32, device=device)
        if rank == 0
        else None
    )
    sum_cost = torch.zeros((), dtype=torch.int64, device=device) if rank == 0 else None
    override_count = (
        torch.zeros((), dtype=torch.int64, device=device) if rank == 0 else None
    )
    changes_per_step = (
        torch.zeros(args.horizon, dtype=torch.int64, device=device)
        if rank == 0
        else None
    )
    refill_agents_per_step: list[int] = []
    step_seconds: list[float] = []
    stage_raw = {
        name: []
        for name in (
            "distance_preferences",
            "preferences_allgather",
            "pibt_and_broadcast",
            "state_and_metrics",
            "gpu_total",
        )
    }
    initial_shortest = None
    solved_step = None

    restore_started = time.perf_counter()
    saved = state.loaded.get("payload", {})
    if saved:
        restore_tensor(global_positions, saved["global_positions"])
        initial_shortest = saved["initial_shortest"].to(device)
        rebuild_cache(cost_to_go, saved["cache"], observation=False)
        if rank == 0:
            restore_tensor(first_arrival, saved["first_arrival"])
            restore_tensor(sum_cost, saved["sum_cost"])
            restore_tensor(override_count, saved["override_count"])
            restore_tensor(changes_per_step, saved["changes_per_step"])
            solved_step = saved["solved_step"]
            step_seconds = saved["step_seconds"]
            stage_raw = saved["stage_raw"]
            refill_agents_per_step = saved["refill_agents_per_step"]
            pibt.priorities.copy_(saved["pibt_priorities"])
            pibt.initial_priorities.copy_(saved["pibt_initial_priorities"])
    trajectory = None
    if rank == 0 and args.trajectory_actions:
        trajectory = Trajectory(args.trajectory_actions, state.loaded.get("trajectory"))
    if rank == 0 and state.loaded.get("trajectory") and trajectory is None:
        raise ValueError("resuming a trajectory requires --trajectory-actions")
    trajectory_path = trajectory.path if trajectory else None
    if saved:
        torch.cuda.synchronize(device)
        state.restore_seconds = time.perf_counter() - restore_started
        state.restore_rng()

    def checkpoint_payload(completed_steps):
        payload = {
            "global_positions": global_positions,
            "initial_shortest": initial_shortest,
            "cache": {"centers": cost_to_go.cache_centers, "valid": cost_to_go.cache_valid},
        }
        if rank == 0:
            payload.update({
                "first_arrival": first_arrival,
                "sum_cost": sum_cost,
                "override_count": override_count,
                "changes_per_step": changes_per_step[:completed_steps],
                "solved_step": solved_step,
                "step_seconds": step_seconds,
                "stage_raw": stage_raw,
                "refill_agents_per_step": refill_agents_per_step,
                "pibt_priorities": pibt.priorities,
                "pibt_initial_priorities": pibt.initial_priorities,
            })
        return payload

    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    wall_start = time.perf_counter() - state.loaded.get("wall_seconds", 0.0)
    if rank == 0:
        print(
            f"GPU-PIBT N={args.agents:,} ranks={world} local={local_agents:,} "
            f"horizon={args.horizon} cache_radius={args.cache_radius} "
            f"bfs_chunk={args.bfs_chunk} resolver={args.pibt_resolver}",
            flush=True,
        )

    end_step = state.start_step if state.solved else args.horizon
    for step in range(state.start_step, end_step):
        dist.barrier()
        torch.cuda.synchronize(device)
        step_start = time.perf_counter()
        events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(5))
        events[0].record()
        local_positions = global_positions[shard_start:shard_end]
        local_scores, local_distances = cost_to_go.scores(
            local_positions, step=step, seed=args.preference_seed
        )
        events[1].record()
        global_scores = gather_local(local_scores)
        if step == 0:
            initial_shortest = gather_local(
                local_distances[:, 0].to(torch.int32)
            )
            if rank == 0:
                assert pibt is not None
                pibt.set_initial_priorities_from_distances(
                    initial_shortest.unsqueeze(0)
                )
        events[2].record()

        actions = torch.empty(args.agents, dtype=torch.long, device=device)
        next_ids = torch.empty_like(actions)
        if rank == 0:
            assert pibt is not None and override_count is not None
            assert changes_per_step is not None
            resolved, overridden, resolved_ids = pibt.step(
                global_scores.unsqueeze(0),
                global_positions.unsqueeze(0),
                prefer_unoccupied_ties=True,
            )
            actions.copy_(resolved[0])
            next_ids.copy_(resolved_ids[0])
            step_changes = overridden.sum()
            override_count += step_changes
            changes_per_step[step] = step_changes
        dist.broadcast(actions, src=0)
        dist.broadcast(next_ids, src=0)
        events[3].record()

        global_positions = torch.stack(
            (
                torch.div(next_ids, scenario.width, rounding_mode="floor"),
                next_ids.remainder(scenario.width),
            ),
            dim=-1,
        )
        done = torch.zeros((), dtype=torch.uint8, device=device)
        if rank == 0:
            assert first_arrival is not None and sum_cost is not None
            on_goal = (global_positions == global_goals).all(dim=-1)
            newly_arrived = on_goal & first_arrival.lt(0)
            first_arrival[newly_arrived] = step + 1
            sum_cost += (~on_goal).sum()
            if bool(on_goal.all().item()):
                solved_step = step + 1
                done.fill_(1)
        dist.broadcast(done, src=0)
        events[4].record()
        torch.cuda.synchronize(device)

        measurements = torch.tensor(
            [
                time.perf_counter() - step_start,
                events[0].elapsed_time(events[1]),
                events[1].elapsed_time(events[2]),
                events[2].elapsed_time(events[3]),
                events[3].elapsed_time(events[4]),
                events[0].elapsed_time(events[4]),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.reduce(measurements, dst=0, op=dist.ReduceOp.MAX)
        refill_agents = torch.tensor(
            cost_to_go.last_refill_count, dtype=torch.int64, device=device
        )
        dist.reduce(refill_agents, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            values = measurements.tolist()
            step_seconds.append(float(values[0]))
            for name, value in zip(stage_raw, values[1:]):
                stage_raw[name].append(float(value))
            refill_agents_per_step.append(int(refill_agents.item()))
            if step == 0 or (step + 1) % 32 == 0:
                steady = step_seconds[1:]
                steady_mean = sum(steady) / len(steady) if steady else float("nan")
                eta_hours = (
                    (args.horizon - step - 1) * steady_mean / 3600.0
                    if steady else float("nan")
                )
                assert changes_per_step is not None
                print(
                    f"step={step + 1}/{args.horizon} "
                    f"step_s={step_seconds[-1]:.3f} "
                    f"steady_mean_s={steady_mean:.3f} "
                    f"elapsed_min={(time.perf_counter() - wall_start) / 60.0:.1f} "
                    f"eta_h={eta_hours:.2f} "
                    f"isr={on_goal.float().mean().item():.6f} "
                    f"distance_ms={stage_raw['distance_preferences'][-1]:.1f} "
                    f"gather_ms={stage_raw['preferences_allgather'][-1]:.1f} "
                    f"pibt_ms={stage_raw['pibt_and_broadcast'][-1]:.1f} "
                    f"refills={refill_agents_per_step[-1]:,} "
                    f"pibt_changes={int(changes_per_step[step].item()):,} "
                    f"allocated_gib={torch.cuda.memory_allocated(device) / 1024**3:.1f} "
                    f"reserved_gib={torch.cuda.memory_reserved(device) / 1024**3:.1f}",
                    flush=True,
                )
            if trajectory is not None:
                trajectory.append(actions, overridden[0])
        stopping = state.should_stop()
        if state.due(step + 1, bool(done.item()), stopping):
            state.save(checkpoint_payload(step + 1), step + 1, bool(done.item()),
                       trajectory, time.perf_counter() - wall_start)
        if stopping:
            if rank == 0:
                print(f"STOPPED step={step + 1}; resumable state saved", flush=True)
            dist.destroy_process_group()
            raise SystemExit(128 + state.stop_signal)
        if bool(done.item()):
            break

    dist.barrier()
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    peak = torch.tensor(
        [max(state.peak[0], torch.cuda.max_memory_allocated(device)), max(state.peak[1], torch.cuda.max_memory_reserved(device))],
        dtype=torch.float64,
        device=device,
    )
    dist.reduce(peak, dst=0, op=dist.ReduceOp.MAX)

    if rank == 0:
        if trajectory is not None:
            trajectory.finish()
        assert first_arrival is not None and sum_cost is not None
        assert override_count is not None and changes_per_step is not None
        assert initial_shortest is not None
        if bool(initial_shortest.le(0).any().item()):
            raise RuntimeError("initial BFS distances must be positive and reachable")
        final_on_goal = (global_positions == global_goals).all(dim=-1)
        steps = len(step_seconds)
        arrival = torch.where(
            first_arrival.lt(0),
            torch.full_like(first_arrival, args.horizon),
            first_arrival,
        )
        changes_raw = changes_per_step[:steps].cpu().tolist()
        changes_total = int(override_count.item())
        stage_summary = {name: timing_stats(values) for name, values in stage_raw.items()}
        steady_summary = {
            name: timing_stats(values[1:]) for name, values in stage_raw.items()
        }
        mean_step = sum(step_seconds) / steps
        result = {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm": "GPU-PIBT",
            "distributed": True,
            "world_size": world,
            "agents": args.agents,
            "agents_per_rank": local_agents,
            "horizon": args.horizon,
            "steps": steps,
            "solved": solved_step is not None,
            "sr": 1.0 if solved_step is not None else 0.0,
            "isr": float(final_on_goal.float().mean().item()),
            "unresolved_agents": int(first_arrival.lt(0).sum().item()),
            "makespan": solved_step or args.horizon,
            "sum_cost": int(sum_cost.item()),
            "mean_cost": float(sum_cost.item() / args.agents),
            "mean_arrival_stretch": float(
                (arrival.float() / initial_shortest.float()).mean().item()
            ),
            "normalized_soc": float(
                arrival.double().sum().item()
                / initial_shortest.double().sum().item()
            ),
            "exact_shortest_path_sum": int(initial_shortest.sum().item()),
            "exact_shortest_path_mean": float(initial_shortest.float().mean().item()),
            "pibt_overrides": changes_total,
            "pibt_override_fraction": changes_total / (steps * args.agents),
            "pibt_changes": {
                "reference_action": "top exact-distance preference before PIBT",
                "total": changes_total,
                "fraction_of_agent_decisions": changes_total / (steps * args.agents),
                "mean_per_step": changes_total / steps,
                "max_per_step": max(changes_raw, default=0),
                "per_step": changes_raw,
            },
            "preference_construction": {
                "primary_key": "exact CUDA BFS distance to goal",
                "secondary_key": "prefer currently unoccupied candidate",
                "tie_break": "counter-hashed candidate permutation",
                "score_encoding": "int32 lexicographic distance/free/hash15",
                "seed": args.preference_seed,
                "world_size_invariant": True,
            },
            "priority_mode": "exact_gpu_bfs",
            "refill_agents": {
                "total": sum(refill_agents_per_step),
                "max_per_step": max(refill_agents_per_step, default=0),
                "per_step": refill_agents_per_step,
            },
            "wall_seconds": wall_seconds,
            "step_seconds": step_seconds,
            "mean_step_seconds": mean_step,
            "prefill_step_seconds": step_seconds[0],
            "prefill_gpu_ms": stage_raw["gpu_total"][0],
            "prefill_distance_preferences_ms": stage_raw["distance_preferences"][0],
            "timing": stage_summary,
            "timing_steady": steady_summary,
            "timing_raw_ms": stage_raw,
            "decision_us_per_agent": mean_step * 1e6 / args.agents,
            "throughput_k_agent_steps_per_s": args.agents / (1000.0 * mean_step),
            "peak_allocated_gib_per_rank": float(peak[0].item() / 1024**3),
            "peak_reserved_gib_per_rank": float(peak[1].item() / 1024**3),
            "bfs_chunk": args.bfs_chunk,
            "cache_radius": args.cache_radius,
            "master": args.scenario_master,
            "master_sha256": sha256(args.scenario_master),
            "master_metadata": master["metadata"],
            "git_branch": branch,
            "git_sha": commit,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(device)] * world,
            "shielding": "standalone_exact_cuda_pibt",
            "pibt_resolver": pibt.resolver_mode,
            "goal_mode": "global_uniform_disjoint",
        }
        result["evaluation_state"] = state.report()
        if trajectory is not None:
            result["trajectory"] = trajectory.metadata(steps, args.agents)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
        if trajectory_path is not None:
            metadata = Path(str(trajectory_path) + ".json")
            metadata_temporary = Path(str(metadata) + ".tmp")
            metadata_temporary.write_text(
                json.dumps(
                    {
                        **result["trajectory"],
                        "algorithm": "GPU-PIBT",
                        "agents": args.agents,
                        "steps": steps,
                        "width": scenario.width,
                        "height": scenario.height,
                        "master": args.scenario_master,
                        "master_sha256": result["master_sha256"],
                        "git_sha": commit,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            metadata_temporary.replace(metadata)
        print(
            "RESULT "
            + json.dumps(
                compact_completion_record(result, output),
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
