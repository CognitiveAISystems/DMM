"""GPU state/observations/actions; only terminal flags are polled per step."""
from collections import defaultdict
import time

import torch

from ..simulator import CUDABatch
from ..tasks import _unique, content_hash
from .step_pipeline import StepPipeline, stage_seconds, validate_movement_backend


@torch.inference_mode()
def evaluate_cuda(tasks,policy,*,num_envs=8,progress=None,record_trace=False,
                  movement_backend="resolved", validate_moves=False, profile_steps=True):
    validate_movement_backend(movement_backend)
    started = time.perf_counter()
    tasks = list(tasks)
    _unique(tasks)
    if type(num_envs) is not int or num_envs<1:
        raise ValueError("num_envs must be a positive integer")
    buckets = defaultdict(list)
    for task in tasks:
        buckets[task.num_agents,task.obs_radius].append(task)
    records = {}
    totals = {"environment_startup":0.0,"observation":0.0,"inference_and_shield":0.0,
              "simulation":0.0,"terminal_export":0.0,"terminal_poll_host_wall":0.0}
    event_pairs = []
    movement_timings = dict.fromkeys(("inference", "movement_resolution", "commit"), 0.0)
    bfs_profiles = []
    shield_profiles = []
    agent_steps = 0
    for bucket in buckets.values():
        for offset in range(0,len(bucket),num_envs):
            cohort = bucket[offset:offset+num_envs]
            stage = time.perf_counter()
            batch = CUDABatch(cohort,device=policy.device)
            policy.reset(batch)
            if policy.shield is not None:
                policy.shield.profile = profile_steps
            pipeline = StepPipeline(batch, policy, movement_backend=movement_backend, validate=validate_moves)
            torch.cuda.synchronize(policy.device)
            totals["environment_startup"] += time.perf_counter()-stage
            active = list(range(len(cohort)))
            proposals, submitted, traces = [], [], []
            while active:
                events = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if profile_steps else None
                stream = torch.cuda.current_stream(policy.device)
                if events: events[0].record(stream)
                inputs = policy.prepare(active)
                if events: events[1].record(stream)
                output = pipeline.propose(inputs,active)
                if events: events[2].record(stream)
                raw, actions, move = pipeline.resolve(output,active)
                if events: events[3].record(stream)
                pipeline.commit(actions,move)
                if events:
                    events[4].record(stream)
                    event_pairs.append(events)
                # WUDLR is 0..4. Compact device tapes retain the exact exported
                # actions/proposals and hashes while using eight times less RAM.
                proposals.append(raw.to(torch.uint8))
                if policy.shield is not None:
                    submitted.append(actions.to(torch.uint8))
                if record_trace:
                    traces.append((batch.positions.clone(),batch.executed.clone(),batch.solved.clone(),batch.steps.clone()))
                agent_steps += len(active)*batch.n
                stage = time.perf_counter()
                # Deliberate B-byte synchronization boundary. No per-agent state,
                # token, observation dict or action crosses to CPU in this loop.
                finished = batch.finished.cpu().tolist()
                totals["terminal_poll_host_wall"] += time.perf_counter()-stage
                active = [i for i in active if not finished[i]]
            stage = time.perf_counter()
            terminal = batch.export()
            lengths = batch.steps.cpu().tolist()
            proposals = torch.stack(proposals).cpu().tolist()
            submitted = torch.stack(submitted).cpu().tolist() if submitted else proposals
            if record_trace:
                traces = [torch.stack([t[i] for t in traces]).cpu().tolist() for i in range(4)]
            for i, task in enumerate(cohort):
                length = lengths[i]
                raw = [rows[i] for rows in proposals[:length]]
                actions = [rows[i] for rows in submitted[:length]]
                record = {"task_id":task.task_id,"layout_hash":task.layout_hash,"task":task.to_dict(),
                          "profile":task.profile,"horizon":task.horizon,"status":"completed","policy_seed":task.policy_seed,
                          "policy":policy.metadata,"movement_backend":movement_backend,
                          "env_grid_search":{"map_name":task.task_id,"num_agents":task.num_agents},
                          "metrics":terminal[i]["metrics"],"steps":length,"actions_hash":content_hash(actions),
                          "actions":actions,"final_positions":terminal[i]["positions"]}
                if policy.shield is not None:
                    record["policy_proposals"] = raw
                    record["shield_overrides"] = sum(a!=p for row,prop in zip(actions,raw) for a,p in zip(row,prop))
                    if hasattr(policy.shield, "statistics"):
                        record["shield_statistics"] = policy.shield.statistics(i)
                if getattr(policy,"map_services",None) is not None:
                    record["observation_services"] = policy.map_services.metadata()
                if record_trace:
                    record["trace"] = [{"positions":[[p//batch.width-batch.radius,p%batch.width-batch.radius] for p in traces[0][s][i]],
                                        "executed_actions":traces[1][s][i],"rewards":[float(traces[2][s][i])]*batch.n,
                                        "terminated":[traces[2][s][i]]*batch.n,"truncated":[traces[3][s][i]>=task.horizon]*batch.n,
                                        "metrics":terminal[i]["metrics"] if s==length-1 else None} for s in range(length)]
                records[task.task_id] = record
                if progress:
                    progress(record,len(records),len(tasks))
            totals["terminal_export"] += time.perf_counter()-stage
            if getattr(policy.shield,"profile",False):
                shield_profiles.append(policy.shield.profiling_summary())
            if getattr(policy,"profile_bfs",False):
                bfs_profiles.append({"task_ids":[t.task_id for t in cohort],
                                     **policy.tokenizer.cache.profiling_summary()})
            del pipeline  # Do not retain the previous shield/cache across the next reset.
    torch.cuda.synchronize(policy.device)
    for events in event_pairs:
        stages = stage_seconds(events, shielded=policy.shield is not None)
        for key in ("observation","inference_and_shield","simulation"):
            totals[key] += stages[key]
        for key in movement_timings:
            movement_timings[key] += stages[key]
    result = {"records":[records[t.task_id] for t in tasks],"step_samples":[],"timings_seconds":totals,
            "evaluation_wall_seconds":time.perf_counter()-started,"active_agent_steps":agent_steps,"num_envs":num_envs,
            "instrumentation":"CUDA events for observation/inference/simulation (includes dispatch gaps); host wall for startup/export/poll; poll overlaps GPU stages, do not sum",
            "backend":"cuda-native-soft","terminal_poll_interval":1,
            "movement_backend":movement_backend,"movement_timings_seconds":movement_timings,
            "movement_timing_scope":"Breakdown of inference_and_shield + simulation, not additional time"}
    if bfs_profiles:
        result["bfs_profile"] = {"cohorts":bfs_profiles,
            "gpu_event_seconds":sum(p["gpu_event_seconds"] for p in bfs_profiles),
            "scope":"Nested in environment_startup/observation; do not add to parent timings"}
    if shield_profiles:
        result["shield_profile"] = {"cohorts":shield_profiles,
            "gpu_event_seconds":sum(p["gpu_event_seconds"] for p in shield_profiles),
            "scope":"Nested in inference_and_shield; includes dispatch gaps, not GPU utilization"}
    result["profile_steps"] = profile_steps
    result["action_tape_dtype"] = "uint8"
    if not profile_steps:
        # Missing means not measured, never report zero as a GPU timing.
        for key in ("observation", "inference_and_shield", "simulation"):
            totals.pop(key)
        result["movement_timings_seconds"] = {}
        result["movement_timing_scope"] = "not measured (profile_steps=False)"
        result["instrumentation"] = "host wall for startup/export/poll; per-step CUDA events disabled"
    return result
