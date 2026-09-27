"""Optional static-goal GPU BFS cache and DMM cost-to-go token view."""
from contextlib import nullcontext
import time

import torch


def validate_bfs_options(chunk_size, budget_bytes):
    """Reject invalid allocation settings before constructing device buffers."""
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("BFS chunk size must be a positive integer")
    if type(budget_bytes) is not int or budget_bytes < 1:
        raise ValueError("BFS budget in bytes must be a positive integer")


class CUDACostToGo:
    def __init__(self, batch, *, budget_bytes=256*1024**2, chunk_size=128, profile=False):
        validate_bfs_options(chunk_size, budget_bytes)
        self.batch = batch
        self.budget_bytes = budget_bytes
        self.profile_events = [] if profile else None
        self.profile_cache_hits = 0
        self.chunk_size = min(chunk_size,batch.b*batch.n)
        required = (batch.b*batch.n+self.chunk_size)*batch.cells*4
        if required>budget_bytes:
            raise ValueError(f"BFS cache and scratch require {required} bytes, exceeding budget {budget_bytes}")
        if batch.b*batch.n*batch.cells >= 2**31:
            raise ValueError("BFS cache exceeds the kernel's int32 indexing limit")
        self.distances = torch.empty((batch.b,batch.n,batch.cells), dtype=torch.int32, device=batch.device)
        self.queue = torch.empty((self.chunk_size,batch.cells), dtype=torch.int32, device=batch.device)
        self.generations = [-1]*batch.b
        self.refresh()

    def refresh(self):
        ids = [i for i,g in enumerate(self.batch.generations) if g!=self.generations[i]]
        if not ids:
            if self.profile_events is not None:
                self.profile_cache_hits += 1
            return ids
        slots = self.batch.slots(ids)
        total = len(ids)*self.batch.n
        profiling = self.profile_events is not None
        scope = torch.profiler.record_function("pogema_gpu::bfs_build") if profiling else nullcontext()
        with scope:
            if profiling:
                start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                stream = torch.cuda.current_stream(self.batch.device)
                start.record(stream)
                began = time.perf_counter()
            for offset in range(0,total,self.chunk_size):
                self.batch.kernel.bfs(self.batch.walls,self.batch.goals,slots,self.distances,self.queue,
                                      offset,min(self.chunk_size,total-offset))
            if profiling:
                enqueue_seconds = time.perf_counter()-began
                end.record(stream)
                self.profile_events.append((start,end,{"environments":len(ids),"goals":total,
                    "kernel_launches":(total+self.chunk_size-1)//self.chunk_size,
                    "kind":"initial" if all(g==-1 for g in self.generations) else "refresh",
                    "host_enqueue_seconds":enqueue_seconds}))
        for i in ids:
            self.generations[i] = self.batch.generations[i]
        return ids

    def profiling_summary(self):
        """Explicit reporting boundary: wait for recorded events, never in refresh()."""
        if self.profile_events is None:
            raise ValueError("BFS profiling was not enabled")
        builds = []
        for start,end,metadata in self.profile_events:
            end.synchronize()
            builds.append({**metadata,"gpu_event_seconds":start.elapsed_time(end)/1000})
        return {"builds":builds,"cache_hits":self.profile_cache_hits,
                "chunk_size":self.chunk_size,"budget_bytes":self.budget_bytes,
                "gpu_event_seconds":sum(b["gpu_event_seconds"] for b in builds),
                "host_enqueue_seconds":sum(b["host_enqueue_seconds"] for b in builds),
                "goals":sum(b["goals"] for b in builds),
                "kernel_launches":sum(b["kernel_launches"] for b in builds),
                "cache_bytes":self.distances.numel()*self.distances.element_size(),
                "scratch_bytes":self.queue.numel()*self.queue.element_size(),
                "scope":"BFS launch-loop CUDA events include dispatch gaps; excludes allocation/upload/compilation; overlaps parent startup/observation timing"}

    def at_positions(self):
        self.refresh()
        return self.distances.gather(-1,self.batch.positions.unsqueeze(-1)).squeeze(-1)


class CUDADMMTokenizer:
    def __init__(self, batch, *, services=None, **cache_options):
        # For padded size <=74 and padding >=5, every position's (pos-5)//64
        # window origin is zero. The 129-cell window covers the whole map,
        # so exact full-map BFS yields the same grid_step=64 tokens.
        if batch.radius<5:
            raise ValueError("DMM token parity requires padding>=5")
        self.batch = batch
        from .services import MapServices
        if services is not None and (services.batch is not batch or cache_options):
            raise ValueError("supply same-batch services or cache options, not both")
        self.services = services if services is not None else MapServices(batch, **cache_options)
        self.cache = self.services.cost_to_go()
        self.history = torch.full((batch.b,batch.n,5),44, dtype=torch.long, device=batch.device)
        self.generations = batch.generations.copy()

    def refresh(self):
        self.cache.refresh()
        changed = [i for i,g in enumerate(self.batch.generations) if g!=self.generations[i]]
        if changed:
            self.history[self.batch.slots(changed)] = 44
            self.generations = self.batch.generations.copy()

    def prepare(self, ids=None):
        self.refresh()
        options = {"origins": self.cache.origins} if hasattr(self.cache,"origins") else {}
        return self.batch.kernel.tokens(self.batch.positions,self.batch.goals,self.cache.distances,
                                        self.history,self.batch.slots(ids),self.batch.width,**options)

    def remember(self, proposals, ids=None, *, include_finished=False):
        """Call before stepping: DMM history tracks proposals, not collision repairs."""
        self.refresh()
        updated = torch.cat((self.history[...,1:],(proposals+45).unsqueeze(-1)),dim=-1)
        active = (torch.ones_like(self.batch.finished) if include_finished else
                  ~self.batch.finished)
        if ids is not None:
            selected = torch.zeros_like(active)
            selected[self.batch.slots(ids)] = True
            active = active & selected
        self.history = torch.where(active[:,None,None],updated,self.history)
