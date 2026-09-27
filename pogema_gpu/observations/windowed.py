"""Bounded grid-step=64 goal-distance windows for large maps.

Exact full-map BFS is streamed through scratch; only each agent's inclusive
129x129 window survives. This avoids a border-to-border matrix and the much
larger agents-by-full-map cache. Windows refresh when agents approach an edge.
"""
from contextlib import nullcontext
import time

import torch

from .cuda import CUDACostToGo
from .memory import WINDOW_CELLS, window_cache_plan


class CUDAWindowCostToGo(CUDACostToGo):
    def __init__(self, batch, *, budget_bytes=256*1024**2, chunk_size=128, profile=False,
                 device_dispatch=True):
        self.batch = batch
        self.plan = window_cache_plan(batch.b*batch.n, batch.cells, chunk_size, budget_bytes)
        self.budget_bytes = budget_bytes
        self.chunk_size = self.plan["chunk_size"]
        self.profile_events = [] if profile else None
        self.profile_cache_hits = 0
        self.distances = torch.empty((batch.b,batch.n,WINDOW_CELLS), dtype=torch.int32, device=batch.device)
        self.origins = torch.full((batch.b,batch.n,2), -1, dtype=torch.int32, device=batch.device)
        self.scratch = torch.empty((self.chunk_size,batch.cells), dtype=torch.int32, device=batch.device)
        self.queue = torch.empty_like(self.scratch)
        self.overflow = torch.zeros(self.chunk_size, dtype=torch.bool, device=batch.device)
        self.generations = [-1]*batch.b
        self._version = -1
        self.shapes = torch.empty((batch.b,2), dtype=torch.long, device=batch.device)
        self.device_dispatch = device_dispatch
        self.work = torch.empty(2, dtype=torch.int32, device=batch.device)
        self.refresh()

    def refresh(self):
        b = self.batch
        if self._version == b.state_version:
            if self.profile_events is not None:
                self.profile_cache_hits += 1
            return []
        changed = [i for i,g in enumerate(b.generations) if g != self.generations[i]]
        for i in changed:
            self.origins[i] = -1
            t = b.tasks[i]
            self.shapes[i] = torch.tensor([t.height+2*t.obs_radius, t.width+2*t.obs_radius], device=b.device)
        if self.device_dispatch:
            profiling = self.profile_events is not None
            if profiling:
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record(torch.cuda.current_stream(b.device))
                began = time.perf_counter()
            b.kernel.window_bfs_dispatch(b.walls,b.goals,b.positions,b.finished,self.shapes,
                self.distances,self.origins,self.scratch,self.queue,self.overflow,self.work)
            torch._assert_async(~self.overflow.any(), "goal distances exceed uint16 range")
            if profiling:
                count = self.work[1].clone()  # Resolve only at profiling_summary(), never here.
                end.record(torch.cuda.current_stream(b.device))
                self.profile_events.append((start,end,{"environments":b.b,"goals":count,
                    "kernel_launches":1,"kind":"device-window-dispatch",
                    "host_enqueue_seconds":time.perf_counter()-began}))
            self.generations = b.generations.copy()
            self._version = b.state_version
            return changed
        row, col = b.positions // b.width, b.positions % b.width
        top, left = self.origins[...,0], self.origins[...,1]
        bottom = torch.minimum(top+128, self.shapes[:,0,None]-1)
        right = torch.minimum(left+128, self.shapes[:,1,None]-1)
        stale = (top < 0) | ((~b.finished[:,None]) & (
            (row-5 < top) | (row+5 > bottom) | (col-5 < left) | (col+5 > right)))
        # One bounded host synchronization per state version to size the work
        # list, not per-agent transfers. No per-step BFS for retained windows.
        agents = stale.flatten().nonzero().flatten()
        count = agents.numel()
        if count:
            profiling = self.profile_events is not None
            scope = torch.profiler.record_function("pogema_gpu::bfs_build") if profiling else nullcontext()
            with scope:
                if profiling:
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    stream = torch.cuda.current_stream(b.device)
                    start.record(stream)
                    began = time.perf_counter()
                self.overflow.zero_()
                for offset in range(0, count, self.chunk_size):
                    b.kernel.window_bfs(b.walls, b.goals, b.positions, agents, self.distances,
                        self.origins, self.scratch, self.queue, self.overflow, offset,
                        min(self.chunk_size,count-offset))
                torch._assert_async(~self.overflow.any(), "goal distances exceed uint16 range")
                if profiling:
                    enqueue = time.perf_counter() - began
                    end.record(stream)
                    self.profile_events.append((start,end,{"environments":b.b,"goals":count,
                        "kernel_launches":(count+self.chunk_size-1)//self.chunk_size,
                        "kind":"initial" if all(g==-1 for g in self.generations) else "window-refresh",
                        "host_enqueue_seconds":enqueue}))
        elif self.profile_events is not None:
            self.profile_cache_hits += 1
        self.generations = b.generations.copy()
        self._version = b.state_version
        return changed

    def at_positions(self):
        self.refresh()
        b = self.batch
        local = (b.positions//b.width-self.origins[...,0])*129+b.positions%b.width-self.origins[...,1]
        return self.distances.gather(-1,local.unsqueeze(-1)).squeeze(-1)

    def profiling_summary(self):
        if self.profile_events is not None:
            for _,end,metadata in self.profile_events:
                if isinstance(metadata["goals"],torch.Tensor):
                    end.synchronize()
                    metadata["goals"] = int(metadata["goals"].item())
        result = super().profiling_summary()
        result.update(self.plan)
        result["implementation"] = ("streamed-exact-bfs-grid64-window129-device-dispatch-v2"
                                    if self.device_dispatch else "streamed-exact-bfs-grid64-window129-v1")
        return result
