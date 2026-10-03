"""Run exact agent-sharded DMM-08M with centralized CUDA PIBT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from evaluation.one_million.sharded_observation import ShardedGPUObservationGenerator
from evaluation.one_million.scenario import load_master, master_prefix
from evaluation.one_million.resume import (EvaluationState, Trajectory, add_resume_arguments,
                                           rebuild_cache, restore_tensor)
from evaluation.one_million.pibt import BatchedPIBT
from evaluation.one_million.config import MillionConfig, _ObsGenConfig
from evaluation.one_million.policy import deployment_policy_from_checkpoint
from evaluation.one_million.settled_skip import settled_observation_mask


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def timing_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return {"total_ms": 0.0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
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


def compact_completion_record(result: dict, output: str | Path) -> dict:
    """Return scalar completion fields suitable for stdout and queue logs."""
    steady = result.get("timing_steady", {})
    return {
        "event": "completed",
        "output": str(output),
        "agents": result["agents"],
        "steps": result["steps"],
        "solved": result["solved"],
        "sr": result["sr"],
        "isr": result["isr"],
        "makespan": result["makespan"],
        "wall_seconds": result["wall_seconds"],
        "mean_step_seconds": result["mean_step_seconds"],
        "pibt_overrides": result["pibt_overrides"],
        "pibt_override_fraction": result["pibt_override_fraction"],
        "pibt_resolver": result["pibt_resolver"],
        "model_skip_fraction": result.get(
            "settled_neighborhood_skipping", {}
        ).get("fraction_of_agent_decisions", 0.0),
        "stochastic_tail_activation_step": result.get(
            "stochastic_tail", {}
        ).get("activation_step"),
        "stochastic_inference_mode": result.get(
            "stochastic_tail", {}
        ).get("mode", "disabled"),
        "timing_steady_mean_ms": {
            name: steady[name]["mean_ms"]
            for name in (
                "observation",
                "model",
                "shield_and_broadcast",
                "gpu_total",
            )
            if name in steady
        },
    }


def code_state() -> tuple[str | None, str | None]:
    """Record source revision when available, including anonymous archives."""
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None
    return git("branch", "--show-current"), git("rev-parse", "HEAD")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-master", required=True)
    parser.add_argument("--agents", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=4096)
    parser.add_argument("--agent-chunk", type=int, default=4096)
    parser.add_argument("--bfs-chunk", type=int, default=2048)
    parser.add_argument("--cache-radius", type=int, default=70)
    parser.add_argument("--checkpoint", default="weights/DMM-MICPO-08M.pt")
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
        help=(
            "optional gzip-compressed uint8 action stream; initial positions "
            "and these executed actions reconstruct the exact trajectory"
        ),
    )
    parser.add_argument(
        "--skip-settled-neighborhoods",
        action="store_true",
        help=(
            "force a wait proposal and skip model inference when ego and every "
            "encoded neighbor are currently on goal"
        ),
    )
    parser.add_argument(
        "--stochastic-policy",
        action="store_true",
        help=(
            "sample native DMM-08M actions for every active receiver from the "
            "first model step; settled receivers still propose wait"
        ),
    )
    parser.add_argument(
        "--stochastic-tail-patience",
        type=int,
        default=0,
        help=(
            "switch active receivers to native stochastic inference after this "
            "many steps without a new best unresolved-agent count; zero disables"
        ),
    )
    parser.add_argument(
        "--stochastic-tail-max-unresolved",
        type=int,
        default=4096,
        help="only activate the stochastic tail at or below this unresolved count",
    )
    parser.add_argument("--stochastic-tail-seed", type=int, default=0)
    add_resume_arguments(parser)
    args = parser.parse_args()
    if args.stochastic_tail_patience < 0:
        parser.error("--stochastic-tail-patience must be nonnegative")
    if args.stochastic_tail_max_unresolved <= 0:
        parser.error("--stochastic-tail-max-unresolved must be positive")
    if args.stochastic_policy and args.stochastic_tail_patience:
        parser.error(
            "--stochastic-policy and --stochastic-tail-patience are mutually "
            "exclusive"
        )
    if (
        args.stochastic_policy or args.stochastic_tail_patience
    ) and not args.skip_settled_neighborhoods:
        parser.error(
            "stochastic inference requires --skip-settled-neighborhoods"
        )

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
    state = EvaluationState(args, rank, world, device, 'DMM')

    master = load_master(args.scenario_master)
    scenario = master_prefix(master, args.agents)
    cfg = MillionConfig()
    cfg.device = str(device)
    cfg.max_horizon = args.horizon
    cfg.use_spatial_neighbors = True
    cfg.bfs_chunk_size = args.bfs_chunk
    cfg.agent_chunk_size = args.agent_chunk
    cfg.use_fp16 = True

    global_positions = torch.as_tensor(
        scenario.positions, dtype=torch.long, device=device
    ).contiguous()
    global_goals = torch.as_tensor(
        scenario.goals, dtype=torch.long, device=device
    ).contiguous()
    last_actions = torch.full(
        (args.agents,), -1, dtype=torch.long, device=device
    )

    from model.weights import resolve_weights

    args.checkpoint = str(resolve_weights(args.checkpoint))
    policy, model_architecture = deployment_policy_from_checkpoint(
        args.checkpoint,
        device=device,
        n_comm_rounds=cfg.n_comm_rounds,
        agent_chunk_size=args.agent_chunk,
    )
    if (
        args.stochastic_policy or args.stochastic_tail_patience
    ) and model_architecture != "dmm08m":
        raise ValueError("stochastic inference requires DMM-08M")
    stochastic_generator = torch.Generator(device=device)
    stochastic_generator.manual_seed(
        int(args.stochastic_tail_seed) + rank * 1_000_003
    )
    generator = ShardedGPUObservationGenerator(
        width=scenario.width, height=scenario.height, grid=scenario.grid,
        cfg=_ObsGenConfig(cfg), cache_radius=args.cache_radius,
        on_target="nothing", shard_start=shard_start, shard_end=shard_end,
    )
    generator.create_agents(scenario.positions, scenario.goals)
    pibt = (
        BatchedPIBT(
            [scenario], device, priority_mode="manhattan_gpu",
            resolver_mode=args.pibt_resolver,
        )
        if rank == 0 else None
    )

    def gather_local(local: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            (args.agents, *local.shape[1:]), dtype=local.dtype, device=device
        )
        dist.all_gather_into_tensor(output, local.contiguous())
        return output

    def gather_sender(local: torch.Tensor) -> torch.Tensor:
        return gather_local(local.squeeze(0)).unsqueeze(0)

    first_arrival = torch.full(
        (args.agents,), -1, dtype=torch.int32, device=device
    ) if rank == 0 else None
    sum_cost = torch.zeros((), dtype=torch.int64, device=device) if rank == 0 else None
    override_count = torch.zeros((), dtype=torch.int64, device=device) if rank == 0 else None
    pibt_changes_per_step = (
        torch.zeros(args.horizon, dtype=torch.int64, device=device)
        if rank == 0 else None
    )
    skipped_agents_per_step = (
        torch.zeros(args.horizon, dtype=torch.int64, device=device)
        if rank == 0 else None
    )
    skipped_pibt_changes_per_step = (
        torch.zeros(args.horizon, dtype=torch.int64, device=device)
        if rank == 0 else None
    )
    stochastic_tail_per_step = (
        torch.zeros(args.horizon, dtype=torch.uint8, device=device)
        if rank == 0 else None
    )
    step_seconds: list[float] = []
    stage_raw = {
        name: [] for name in (
            "observation",
            "model_policy_and_message_comm",
            "logits_allgather",
            "model",
            "shield_and_broadcast",
            "state_and_metrics",
            "gpu_total",
        )
    }
    solved_step = None
    stochastic_tail_flag = torch.tensor(
        int(args.stochastic_policy), dtype=torch.uint8, device=device
    )
    stochastic_tail_activation_step = 1 if args.stochastic_policy else None
    best_unresolved = args.agents
    last_unresolved_improvement_step = 0
    restore_started = time.perf_counter()
    saved = state.loaded.get("payload", {})
    if saved:
        restore_tensor(global_positions, saved["global_positions"])
        restore_tensor(last_actions, saved["last_actions"])
        restore_tensor(stochastic_tail_flag, saved["stochastic_tail_flag"])
        generator.global_action_history.copy_(saved["action_history"])
        stochastic_generator.set_state(saved["stochastic_rng"])
        if saved["cache"] is not None:
            rebuild_cache(generator, saved["cache"], observation=True)
        if rank == 0:
            restore_tensor(first_arrival, saved["first_arrival"])
            restore_tensor(sum_cost, saved["sum_cost"])
            restore_tensor(override_count, saved["override_count"])
            restore_tensor(pibt_changes_per_step, saved["pibt_changes_per_step"])
            restore_tensor(skipped_agents_per_step, saved["skipped_agents_per_step"])
            restore_tensor(skipped_pibt_changes_per_step, saved["skipped_pibt_changes_per_step"])
            restore_tensor(stochastic_tail_per_step, saved["stochastic_tail_per_step"])
            solved_step = saved["solved_step"]
            stochastic_tail_activation_step = saved["stochastic_tail_activation_step"]
            best_unresolved = saved["best_unresolved"]
            last_unresolved_improvement_step = saved["last_unresolved_improvement_step"]
            step_seconds = saved["step_seconds"]
            stage_raw = saved["stage_raw"]
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
            "last_actions": last_actions,
            "stochastic_tail_flag": stochastic_tail_flag,
            "action_history": generator.global_action_history,
            "stochastic_rng": stochastic_generator.get_state(),
            "cache": ({"centers": generator._cache_center, "valid": generator._cache_valid}
                      if generator.cache_enabled else None),
        }
        if rank == 0:
            payload.update({
                "first_arrival": first_arrival,
                "sum_cost": sum_cost,
                "override_count": override_count,
                "pibt_changes_per_step": pibt_changes_per_step[:completed_steps],
                "skipped_agents_per_step": skipped_agents_per_step[:completed_steps],
                "skipped_pibt_changes_per_step": skipped_pibt_changes_per_step[:completed_steps],
                "stochastic_tail_per_step": stochastic_tail_per_step[:completed_steps],
                "solved_step": solved_step,
                "stochastic_tail_activation_step": stochastic_tail_activation_step,
                "best_unresolved": best_unresolved,
                "last_unresolved_improvement_step": last_unresolved_improvement_step,
                "step_seconds": step_seconds,
                "stage_raw": stage_raw,
                "pibt_priorities": pibt.priorities,
                "pibt_initial_priorities": pibt.initial_priorities,
            })
        return payload

    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    wall_start = time.perf_counter() - state.loaded.get("wall_seconds", 0.0)

    if rank == 0:
        stochastic_mode = (
            "always" if args.stochastic_policy else
            "stagnation_tail" if args.stochastic_tail_patience else
            "disabled"
        )
        print(
            f"Distributed {model_architecture} N={args.agents:,} ranks={world} "
            f"local={local_agents:,} horizon={args.horizon} "
            f"stochastic={stochastic_mode}", flush=True,
        )
    end_step = state.start_step if state.solved else args.horizon
    for step in range(state.start_step, end_step):
        dist.barrier()
        torch.cuda.synchronize(device)
        step_start = time.perf_counter()
        events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(6))
        events[0].record()
        generator.update_global_agents(global_positions, global_goals, last_actions)
        observations = generator.generate_sharded_observations(
            global_positions, global_goals, gather_local
        )
        if args.skip_settled_neighborhoods:
            settled_mask = settled_observation_mask(
                global_positions,
                global_goals,
                generator.agent_chat_ids,
                shard_start,
                shard_end,
            )
        else:
            settled_mask = torch.zeros(
                local_agents, dtype=torch.bool, device=device
            )
        global_settled_mask = (
            gather_local(settled_mask)
            if args.skip_settled_neighborhoods else None
        )
        if rank == 0 and global_settled_mask is not None:
            skipped_agents_per_step[step] = global_settled_mask.sum()
        events[1].record()
        use_stochastic_tail = bool(stochastic_tail_flag.item())
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if use_stochastic_tail:
                local_logits = policy.stochastic_act_sharded(
                    observations.unsqueeze(0),
                    generator.agent_chat_ids.unsqueeze(0),
                    gather_sender,
                    stochastic_generator,
                    args.agent_chunk,
                    active_mask=~settled_mask,
                ).logits.squeeze(0)
            else:
                local_logits = policy.deterministic_zero_act_sharded(
                    observations.unsqueeze(0), generator.agent_chat_ids.unsqueeze(0),
                    gather_sender, args.agent_chunk,
                    active_mask=(
                        ~settled_mask if args.skip_settled_neighborhoods else None
                    ),
                ).logits.squeeze(0)
        events[2].record()
        global_logits = gather_local(local_logits)
        events[3].record()

        actions = torch.empty(
            args.agents, dtype=torch.long, device=device
        )
        next_ids = torch.empty_like(actions)
        if rank == 0:
            resolved, overridden, resolved_ids = pibt.step(
                global_logits.unsqueeze(0), global_positions.unsqueeze(0)
            )
            actions.copy_(resolved[0])
            next_ids.copy_(resolved_ids[0])
            step_pibt_changes = overridden.sum()
            override_count += step_pibt_changes
            pibt_changes_per_step[step] = step_pibt_changes
            if global_settled_mask is not None:
                skipped_pibt_changes_per_step[step] = (
                    overridden[0] & global_settled_mask
                ).sum()
        dist.broadcast(actions, src=0)
        dist.broadcast(next_ids, src=0)
        events[4].record()
        global_positions = torch.stack(
            (
                torch.div(next_ids, scenario.width, rounding_mode="floor"),
                next_ids.remainder(scenario.width),
            ), dim=-1,
        )
        last_actions = actions

        done = torch.zeros((), dtype=torch.uint8, device=device)
        if rank == 0:
            on_goal = (global_positions == global_goals).all(dim=-1)
            unresolved = int((~on_goal).sum().item())
            newly_arrived = on_goal & first_arrival.lt(0)
            first_arrival[newly_arrived] = step + 1
            sum_cost += (~on_goal).sum()
            stochastic_tail_per_step[step] = int(use_stochastic_tail)
            if unresolved < best_unresolved:
                best_unresolved = unresolved
                last_unresolved_improvement_step = step + 1
            stalled_steps = step + 1 - last_unresolved_improvement_step
            if (
                args.stochastic_tail_patience
                and not bool(stochastic_tail_flag.item())
                and unresolved > 0
                and unresolved <= args.stochastic_tail_max_unresolved
                and stalled_steps >= args.stochastic_tail_patience
            ):
                stochastic_tail_flag.fill_(1)
                stochastic_tail_activation_step = step + 2
                print(
                    f"stochastic_tail_activated model_step="
                    f"{stochastic_tail_activation_step} unresolved={unresolved:,} "
                    f"best_unresolved={best_unresolved:,} "
                    f"stalled_steps={stalled_steps}",
                    flush=True,
                )
            if bool(on_goal.all().item()):
                solved_step = step + 1
                done.fill_(1)
        dist.broadcast(stochastic_tail_flag, src=0)
        dist.broadcast(done, src=0)
        events[5].record()
        torch.cuda.synchronize(device)
        local_measurements = torch.tensor(
            [
                time.perf_counter() - step_start,
                events[0].elapsed_time(events[1]),
                events[1].elapsed_time(events[2]),
                events[2].elapsed_time(events[3]),
                events[1].elapsed_time(events[3]),
                events[3].elapsed_time(events[4]),
                events[4].elapsed_time(events[5]),
                events[0].elapsed_time(events[5]),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.reduce(local_measurements, dst=0, op=dist.ReduceOp.MAX)
        if rank == 0:
            measured = local_measurements.tolist()
            step_seconds.append(float(measured[0]))
            for name, value in zip(stage_raw, measured[1:]):
                stage_raw[name].append(float(value))
            if step == 0 or (step + 1) % 32 == 0:
                steady = step_seconds[1:]
                steady_mean = sum(steady) / len(steady) if steady else None
                model_steady = stage_raw["model"][1:]
                model_mean_ms = (
                    sum(model_steady) / len(model_steady) if model_steady else None
                )
                eta_hours = (
                    (args.horizon - step - 1) * steady_mean / 3600.0
                    if steady_mean is not None else None
                )
                print(
                    f"step={step + 1}/{args.horizon} "
                    f"step_s={step_seconds[-1]:.3f} "
                    f"steady_mean_s="
                    f"{steady_mean if steady_mean is not None else float('nan'):.3f} "
                    f"elapsed_min={(time.perf_counter() - wall_start) / 60.0:.1f} "
                    f"eta_h={eta_hours if eta_hours is not None else float('nan'):.2f} "
                    f"isr={on_goal.float().mean().item():.6f} "
                    f"unresolved={int((~on_goal).sum().item()):,} "
                    f"stochastic_tail={int(use_stochastic_tail)} "
                    f"obs_ms={stage_raw['observation'][-1]:.1f} "
                    f"model_ms={stage_raw['model'][-1]:.1f} "
                    f"model_steady_ms="
                    f"{model_mean_ms if model_mean_ms is not None else float('nan'):.1f} "
                    f"shield_ms={stage_raw['shield_and_broadcast'][-1]:.1f} "
                    f"pibt_changes={int(pibt_changes_per_step[step].item()):,} "
                    f"model_skipped={int(skipped_agents_per_step[step].item()):,} "
                    f"skipped_pibt_changes="
                    f"{int(skipped_pibt_changes_per_step[step].item()):,} "
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
        [max(state.peak[0], torch.cuda.max_memory_allocated(device)),
         max(state.peak[1], torch.cuda.max_memory_reserved(device))],
        dtype=torch.float64, device=device,
    )
    dist.reduce(peak, dst=0, op=dist.ReduceOp.MAX)
    if rank == 0:
        if trajectory is not None:
            trajectory.finish()
        final_on_goal = (global_positions == global_goals).all(dim=-1)
        steps = len(step_seconds)
        stage_summary = {
            name: timing_stats(values) for name, values in stage_raw.items()
        }
        steady_summary = {
            name: timing_stats(values[1:]) for name, values in stage_raw.items()
        }
        pibt_changes_total = int(override_count.item())
        pibt_changes_raw = pibt_changes_per_step[:steps].cpu().tolist()
        pibt_change_fraction = pibt_changes_total / (steps * args.agents)
        skipped_agents_raw = skipped_agents_per_step[:steps].cpu().tolist()
        skipped_agent_decisions = sum(skipped_agents_raw)
        skipped_agent_fraction = skipped_agent_decisions / (steps * args.agents)
        skipped_pibt_changes_raw = (
            skipped_pibt_changes_per_step[:steps].cpu().tolist()
        )
        skipped_pibt_changes = sum(skipped_pibt_changes_raw)
        stochastic_tail_raw = stochastic_tail_per_step[:steps].cpu().tolist()
        stochastic_tail_steps = sum(stochastic_tail_raw)
        result = {
            "status": "completed", "distributed": True,
            "world_size": world, "agents": args.agents,
            "agents_per_rank": local_agents, "horizon": args.horizon,
            "steps": steps, "solved": solved_step is not None,
            "sr": 1.0 if solved_step is not None else 0.0,
            "isr": float(final_on_goal.float().mean().item()),
            "makespan": solved_step or args.horizon,
            "sum_cost": int(sum_cost.item()),
            "mean_cost": float(sum_cost.item() / args.agents),
            "pibt_overrides": pibt_changes_total,
            "pibt_override_fraction": pibt_change_fraction,
            "pibt_changes": {
                "total": pibt_changes_total,
                "fraction_of_agent_decisions": pibt_change_fraction,
                "mean_per_step": pibt_changes_total / steps,
                "max_per_step": max(pibt_changes_raw, default=0),
                "per_step": pibt_changes_raw,
            },
            "settled_neighborhood_skipping": {
                "enabled": args.skip_settled_neighborhoods,
                "criterion": "ego_and_all_encoded_neighbors_on_goal",
                "forced_policy_action": "wait",
                "shielding_still_applied": True,
                "skipped_agent_decisions": skipped_agent_decisions,
                "fraction_of_agent_decisions": skipped_agent_fraction,
                "mean_agents_per_step": skipped_agent_decisions / steps,
                "max_agents_per_step": max(skipped_agents_raw, default=0),
                "per_step": skipped_agents_raw,
                "pibt_overrides_of_wait_proposals": skipped_pibt_changes,
                "pibt_override_fraction": (
                    skipped_pibt_changes / skipped_agent_decisions
                    if skipped_agent_decisions else 0.0
                ),
                "pibt_overrides_per_step": skipped_pibt_changes_raw,
            },
            "stochastic_tail": {
                "enabled": bool(
                    args.stochastic_policy or args.stochastic_tail_patience
                ),
                "mode": (
                    "always" if args.stochastic_policy else
                    "stagnation_tail" if args.stochastic_tail_patience else
                    "disabled"
                ),
                "criterion": (
                    "all_active_model_steps" if args.stochastic_policy else
                    "best_unresolved_stagnation"
                ),
                "patience_steps": args.stochastic_tail_patience,
                "max_unresolved": args.stochastic_tail_max_unresolved,
                "seed": args.stochastic_tail_seed,
                "rank_seed_stride": 1_000_003,
                "active_receivers_only": True,
                "settled_policy_action": "wait",
                "shielding_unchanged": True,
                "activation_step": stochastic_tail_activation_step,
                "stochastic_steps": stochastic_tail_steps,
                "per_step": stochastic_tail_raw,
            },
            "wall_seconds": wall_seconds,
            "step_seconds": step_seconds,
            "mean_step_seconds": sum(step_seconds) / steps,
            "prefill_step_seconds": step_seconds[0],
            "prefill_gpu_ms": stage_raw["gpu_total"][0],
            "prefill_observation_ms": stage_raw["observation"][0],
            "prefill_model_ms": stage_raw["model"][0],
            "timing": stage_summary,
            "timing_steady": steady_summary,
            "timing_raw_ms": stage_raw,
            "throughput_k_agent_steps_per_s": (
                args.agents / (1000.0 * (sum(step_seconds) / steps))
            ),
            "peak_allocated_gib_per_rank": float(peak[0].item() / 1024**3),
            "peak_reserved_gib_per_rank": float(peak[1].item() / 1024**3),
            "agent_chunk": args.agent_chunk, "bfs_chunk": args.bfs_chunk,
            "cache_radius": args.cache_radius,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256(args.checkpoint),
            "model_architecture": model_architecture,
            "master": args.scenario_master,
            "master_sha256": sha256(args.scenario_master),
            "git_branch": branch, "git_sha": commit,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(device)] * world,
            "shielding": "centralized_exact_cuda_pibt",
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
            trajectory_metadata = Path(str(trajectory_path) + ".json")
            metadata_temporary = Path(str(trajectory_metadata) + ".tmp")
            metadata_temporary.write_text(
                json.dumps(
                    {
                        **result["trajectory"],
                        "agents": args.agents,
                        "steps": steps,
                        "width": scenario.width,
                        "height": scenario.height,
                        "master": args.scenario_master,
                        "master_sha256": result["master_sha256"],
                        "git_sha": commit,
                        "checkpoint_sha256": result["checkpoint_sha256"],
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n"
            )
            metadata_temporary.replace(trajectory_metadata)
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
