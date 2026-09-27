"""Variable-N native DMM environments sharing one packed neural call per tick.

Each environment owns a native CUDA simulator/cache/shield context. Physics
and tokenization dispatch remain per-environment; neural inference is packed.
Only a vector of completion flags is read back on each tick. Completed episode
state is exported and released immediately; remaining contexts are not reset.
"""
import time
import torch
from ..simulator import CUDABatch
from ..tasks import _unique
from .recording import EpisodeRecorder
from .scheduling import AgentBudgetQueue
from .step_pipeline import StepPipeline, validate_movement_backend


def state_estimate(task, policy, record_trace):
    cells=(task.height+2*task.obs_radius)*(task.width+2*task.obs_radius)
    from ..observations.memory import window_cache_plan
    mode=policy.bfs_cache_options.get('cache_mode','bounded')
    if mode == 'vendor-window':
        from ..observations.memory import vendor_window_state_bytes
        bfs=vendor_window_state_bytes(task.num_agents,
            task.height+2*task.obs_radius,task.width+2*task.obs_radius,
            policy.bfs_chunk_size,policy.bfs_budget_bytes)
    elif max(task.height,task.width)+2*task.obs_radius <= 74:
        bfs=(task.num_agents+min(task.num_agents,policy.bfs_chunk_size))*cells*4
        if bfs>policy.bfs_budget_bytes: raise ValueError("task exceeds configured BFS budget")
    else:
        plan=window_cache_plan(task.num_agents,cells,policy.bfs_chunk_size,policy.bfs_budget_bytes)
        bfs=plan['cache_bytes']+plan['scratch_bytes']
    rse=(task.horizon+1)*(task.num_agents*4+16) if policy.shield_config.repeat_escape else 0
    if rse>policy.shield_config.rse_budget_bytes: raise ValueError("task exceeds configured RSE budget")
    # Conservative simulator/shield and tape allowance, separate activation reserve.
    return (bfs + rse
            + cells*128 + task.num_agents*4096
            + task.horizon*task.num_agents*(32 if record_trace else 2))


@torch.inference_mode()
def evaluate_ragged(tasks, policy, *, num_envs=128, max_batch_agents=8192,
                    max_state_bytes=8*1024**3, reserve_bytes=16*1024**3,
                    progress=None, record_trace=False, movement_backend="resolved",
                    validate_moves=False, task_ready=None, retain_records=True):
    validate_movement_backend(movement_backend)
    if not callable(getattr(policy,"propose_environments",None)):
        raise ValueError("ragged scheduler requires a native packed-environment policy")
    if max_batch_agents > policy.max_agents:
        raise ValueError("agent budget exceeds native AOTI package limit")
    if reserve_bytes < 0: raise ValueError("reserve_bytes must be nonnegative")
    if policy.bfs_cache_options.get('cache_mode','bounded') not in {'bounded','vendor-window'}:
        raise ValueError("ragged admission requires bounded or vendor-window BFS caches")
    started=time.perf_counter()
    tasks=list(tasks); _unique(tasks)
    free,_=torch.cuda.mem_get_info(policy.device)
    reusable=torch.cuda.memory_reserved(policy.device)-torch.cuda.memory_allocated(policy.device)
    available=min(max_state_bytes,max(0,free+reusable-reserve_bytes))
    queue=AgentBudgetQueue([t.num_agents for t in tasks],
        [state_estimate(t,policy,record_trace) for t in tasks],
        max_agents=max_batch_agents,max_state_bytes=available,max_envs=num_envs)
    contexts={}; records={}; samples=[]; admissions=[]
    totals=dict(environment_startup=0.0,terminal_export=0.0)
    agent_steps=0; tick=0; completed=0; admitted_at={}
    while queue.pending or queue.active:
        for i in queue.admit(None if task_ready is None else lambda i: task_ready(tasks[i])):
            stage=time.perf_counter()
            admitted_at[i]=stage
            batch=CUDABatch([tasks[i]],device=policy.device)
            ctx=policy.episode_context(batch)
            if ctx.shield is not None: ctx.shield.profile=False
            pipeline=StepPipeline(batch,ctx,movement_backend=movement_backend,validate=validate_moves)
            recorder=EpisodeRecorder([tasks[i]],batch,shielded=ctx.shield is not None,record_trace=record_trace)
            contexts[i]=(ctx,pipeline,recorder)
            admissions.append({'task_id':tasks[i].task_id,'tick':tick,'agents':tasks[i].num_agents})
            torch.cuda.synchronize(policy.device)
            totals['environment_startup']+=time.perf_counter()-stage
            del batch,ctx,pipeline,recorder
        ids=list(queue.active)
        if not ids:
            # External shared queues can temporarily have no assigned tasks.
            time.sleep(0.01)
            continue
        active=[contexts[i][0] for i in ids]
        inputs=[ctx.prepare([0]) for ctx in active]
        outputs=policy.propose_environments(active,inputs)
        count=sum(tasks[i].num_agents for i in ids)
        samples.append({'active_envs':len(ids),'agents':count})
        agent_steps+=count
        for i,output in zip(ids,outputs):
            ctx,pipeline,recorder=contexts[i]
            raw,actions,move=pipeline.resolve(output,[0]); pipeline.commit(actions,move)
            zero=ctx.batch.slots([0])
            recorder.step(zero,zero,ctx.batch.steps-1,raw,actions)
        finished=torch.cat([ctx.batch.finished for ctx in active]).cpu().tolist()
        done=[i for i,flag in zip(ids,finished) if flag]
        for i in done:
            stage=time.perf_counter()
            ctx,pipeline,recorder=contexts.pop(i)
            zero=ctx.batch.slots([0]); recorder.finish(zero,zero)
            record=recorder.export(policy.metadata)[0]
            record['movement_backend']=movement_backend
            if ctx.shield is not None: record['shield_statistics']=ctx.shield.statistics(0)
            record['scheduler']='ragged'
            record['environment_latency_seconds']=time.perf_counter()-admitted_at.pop(i)
            completed+=1
            if retain_records: records[i]=record
            if progress: progress(record,completed,len(tasks))
            totals['terminal_export']+=time.perf_counter()-stage
            del ctx,pipeline,recorder
        queue.finish(done)
        del active,inputs,outputs,output,raw,actions,move,zero
        tick+=1
    torch.cuda.synchronize(policy.device)
    return {'records':[records[i] for i in sorted(records)], 'scheduler':'ragged',
            'completed_tasks':completed,'retained_records':retain_records,
            'num_envs':num_envs,'max_batch_agents':max_batch_agents,
            'state_budget_bytes':available,'reserve_bytes':reserve_bytes,
            'admissions':admissions,'step_samples':samples,'active_agent_steps':agent_steps,
            'timings_seconds':totals,'evaluation_wall_seconds':time.perf_counter()-started,
            'profile_steps':False,'movement_backend':movement_backend,
            'instrumentation':'shared packed inference; host aggregate wall; no per-task runtime attribution'}
