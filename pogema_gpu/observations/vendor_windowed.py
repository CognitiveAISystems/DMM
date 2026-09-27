"""Grid64 window adapter for the shared CUDA BFS kernels."""
import time

import torch

from ..kernels.cuda_bfs import load_cuda_bfs
from .windowed import CUDAWindowCostToGo

class CUDAVendorWindowCostToGo(CUDAWindowCostToGo):
    def __init__(self, batch, **options):
        self.extension = load_cuda_bfs()
        self.vendor_walls = batch.walls.to(torch.int8).contiguous()
        super().__init__(batch, **options)

    def refresh(self):
        b = self.batch
        if self._version == b.state_version:
            if self.profile_events is not None:
                self.profile_cache_hits += 1
            return []
        changed = [i for i,g in enumerate(b.generations) if g != self.generations[i]]
        for i in changed:
            self.origins[i] = -1
            self.vendor_walls[i].copy_(b.walls[i])
            t = b.tasks[i]
            self.shapes[i] = torch.tensor([t.height+2*t.obs_radius,
                t.width+2*t.obs_radius], device=b.device)
        row, col = b.positions // b.width, b.positions % b.width
        top, left = self.origins[...,0], self.origins[...,1]
        bottom = torch.minimum(top+128, self.shapes[:,0,None]-1)
        right = torch.minimum(left+128, self.shapes[:,1,None]-1)
        stale = (top<0) | ((~b.finished[:,None]) &
            ((row-5<top)|(row+5>bottom)|(col-5<left)|(col+5>right)))
        profiling = self.profile_events is not None
        if profiling:
            start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record(torch.cuda.current_stream(b.device))
            began = time.perf_counter()
        count = launches = 0
        for e in range(b.b):
            agents = stale[e].nonzero().flatten()
            count += agents.numel()
            for offset in range(0,agents.numel(),self.chunk_size):
                ids = agents[offset:offset+self.chunk_size]
                origins = torch.stack(((row[e,ids]-5).clamp_min(0)//64*64,
                                       (col[e,ids]-5).clamp_min(0)//64*64),-1)
                centers = (origins+64).to(torch.int64).contiguous()
                goals = b.goals[e,ids]
                goals = torch.stack((goals//b.width,goals%b.width),-1).contiguous()
                with torch.cuda.device(b.device):
                    windows = self.extension.raw_bfs_cost2go(self.vendor_walls[e],
                        centers,goals,b.height,b.width,64)
                self.distances[e,ids] = windows.to(torch.int32)
                self.origins[e,ids] = origins.to(torch.int32)
                launches += 1
        if profiling:
            end.record(torch.cuda.current_stream(b.device))
            self.profile_events.append((start,end,{'environments':b.b,'goals':count,
                'kernel_launches':launches,'kind':'vendor-window-refresh',
                'host_enqueue_seconds':time.perf_counter()-began}))
        self.generations = b.generations.copy()
        self._version = b.state_version
        return changed

    def profiling_summary(self):
        result = super().profiling_summary()
        result['implementation'] = 'dmm-08m-vendor-parallel-bfs-grid64-window129-v1'
        return result
