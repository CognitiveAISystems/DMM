"""Optional DMM08M PIBT shield; native soft collision semantics stay separate."""
import torch

from ..kernels import extension
from ..observations.services import MapServices


class CUDAPIBT:
    def __init__(self, batch, *, resolver="sequential", cache=None, repeat_escape=False,
                 max_repeat_retries=16, rse_budget_bytes=512*1024**2, profile=False,
                 native_action_ties=False, device_rse=None):
        from .repeat_escape import RepeatStateEscape, validate_rse
        validate_rse(repeat_escape, max_repeat_retries, rse_budget_bytes)
        if resolver not in {"sequential","components"}:
            raise ValueError("PIBT resolver must be sequential or components")
        if cache is not None and cache.batch is not batch:
            raise ValueError("a PIBT distance cache must belong to the same environment batch")
        self.batch = batch
        self.resolve = getattr(extension("pibt"),resolver)
        self.cache = cache if cache is not None else MapServices(batch).cost_to_go()
        self.priorities = torch.empty((batch.b,batch.n), dtype=torch.float32, device=batch.device)
        self.generations = [-1]*batch.b
        self.committed_steps = torch.zeros_like(batch.steps)
        self.moves = torch.tensor([0,-batch.width,batch.width,-1,1],device=batch.device)
        self.num_cells = torch.full((batch.b,),batch.cells,device=batch.device,dtype=torch.long)
        self.rse = (RepeatStateEscape(batch.positions, max(t.horizon for t in batch.tasks),
                                     max_retries=max_repeat_retries, budget_bytes=rse_budget_bytes)
                    if repeat_escape else None)
        self.pending_positions = None
        self.profile = profile
        self.native_action_ties = native_action_ties
        self.device_rse = (native_action_ties and resolver == "sequential"
                           if device_rse is None else device_rse)
        if self.device_rse and resolver != "sequential":
            raise ValueError("device RSE requires the sequential resolver")
        self.graph_order = torch.tensor([3,4,2,1,0],device=batch.device)
        self.tie_state = (torch.empty((batch.b,625),device=batch.device,dtype=torch.long)
                          if native_action_ties else None)
        self.unused_tie_state = (torch.empty((batch.b,625),device=batch.device,dtype=torch.long)
                                if self.device_rse and self.tie_state is None else None)
        self.rse_workspace = None
        if self.rse is not None and self.device_rse:
            def scratch(shape, dtype=torch.long):
                return torch.empty(shape, device=batch.device, dtype=dtype)
            agents, actions = (batch.b,batch.n), (batch.b,batch.n,5)
            self.rse_workspace = [scratch((batch.b,batch.cells)), scratch((batch.b,batch.cells)),
                scratch(agents), scratch(agents), scratch(actions), scratch(actions,torch.bool),
                scratch(actions,torch.bool), scratch(actions,torch.float32), scratch(agents), scratch(agents)]
        self.event_pairs = []
        self.refresh()

    def refresh(self):
        changed = [i for i,g in enumerate(self.batch.generations) if g!=self.generations[i]]
        if changed:
            if self.tie_state is not None:
                extension("pibt").mt19937_reset(self.tie_state,
                    torch.tensor(changed,device=self.batch.device,dtype=torch.long),
                    torch.tensor([self.batch.tasks[i].policy_seed for i in changed],
                                 device=self.batch.device,dtype=torch.long))
            if self.rse is not None and any(self.batch.tasks[i].horizon >= self.rse.history.shape[1] for i in changed):
                raise ValueError("reset horizon exceeds RSE allocation; rebuild the shield")
            distances = self.cache.at_positions()
            for i in changed:
                cells = self.batch.tasks[i].height*self.batch.tasks[i].width
                # Match exact_cpu's unreachable sentinel and double division,
                # followed by one float32 rounding at the assignment boundary.
                if self.native_action_ties:
                    free_cells = sum(row.count(".") for row in self.batch.tasks[i].obstacles)
                    self.priorities[i] = extension("pibt").native_priority_init(
                        distances[i].to(torch.long).contiguous(), free_cells)
                else:
                    self.priorities[i] = torch.where(distances[i]<0,cells,distances[i]).double()/cells
                self.generations[i] = self.batch.generations[i]
                self.committed_steps[i] = 0
            if self.rse is not None:
                self.rse.reset_at(changed, self.batch.positions)

    def actions(self, scores, ids=None):
        events = None
        if self.profile:
            events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            events[0].record(torch.cuda.current_stream(self.batch.device))
        self.refresh()
        b = self.batch
        if scores.shape!=(b.b,b.n,5) or scores.device!=b.device or not scores.is_floating_point():
            raise ValueError("PIBT scores must be same-device floating [B,N,5]")
        candidates = b.positions[:,:,None]+self.moves
        valid = ~b.walls.flatten(1).gather(1,candidates.flatten(1)).reshape_as(candidates)
        order = (extension("pibt").native_priority_order(self.priorities)
                 if self.native_action_ties else
                 torch.argsort(self.priorities,dim=-1,descending=True,stable=True))
        active = ~b.finished
        if ids is not None:
            selected = torch.zeros_like(active)
            selected[ids] = True
            active &= selected
        def resolve(forbidden):
            allowed = valid if forbidden is None else valid & ~forbidden
            masked = torch.nan_to_num(scores.float(),nan=0.0).masked_fill(~allowed,-torch.inf)
            if self.tie_state is None:
                preferences = torch.argsort(masked,dim=-1,descending=True,stable=True)
            else:
                # Native draws BEFORE applying retry constraints, for every
                # wall-valid action, including non-ties. Never perturb scores.
                ties = extension("pibt").mt19937_ties(self.tie_state,valid,active)
                tie_order = self.graph_order[torch.argsort(ties[...,self.graph_order],dim=-1,stable=True)]
                primary = torch.argsort(masked.gather(-1,tie_order),dim=-1,descending=True,stable=True)
                preferences = tie_order.gather(-1,primary)
            return self.resolve(b.positions,candidates,preferences,allowed,order,self.num_cells,b.cells)
        if self.rse is not None and self.device_rse:
            actions, next_ids, stats = extension("pibt").rse_workspace(
                b.positions, b.goals, candidates, torch.nan_to_num(scores.float(),nan=0.0).contiguous(),
                valid, order, self.num_cells, active, self.rse.history, self.rse.hashes,
                self.rse.coefficients, self.rse.lengths,
                self.tie_state if self.tie_state is not None else self.unused_tie_state,
                b.cells, self.rse.max_retries, self.native_action_ties, self.rse_workspace)
            self.rse.pending_stats = stats
        else:
            actions, next_ids = (resolve(None) if self.rse is None else
                self.rse.actions(b.positions,b.goals,valid,order,active,resolve))
        self.pending_positions = torch.where(active[:,None],next_ids,b.positions)
        if events is not None:
            events[1].record(torch.cuda.current_stream(b.device))
            self.event_pairs.append(events)
        return torch.where(active[:,None],actions,0)

    def commit(self):
        """Advance priority only after the simulator commits the shielded action."""
        updated = torch.where(self.batch.positions==self.batch.goals,
                              self.priorities-torch.floor(self.priorities),self.priorities+1)
        # The terminal transition still commits. Repeated terminal steps do not.
        changed = self.batch.steps != self.committed_steps
        if self.rse is not None:
            torch._assert_async(((self.batch.positions == self.pending_positions) | ~changed[:,None]).all(),
                                "native simulator repaired an RSE-shielded transition")
            self.rse.commit(self.batch.positions, changed)
        self.priorities = torch.where(changed[:,None],updated,self.priorities)
        self.committed_steps = self.batch.steps.clone()

    def statistics(self, slot):
        return self.rse.statistics(slot) if self.rse is not None else {}

    def profiling_summary(self):
        torch.cuda.synchronize(self.batch.device)
        return {"gpu_event_seconds":sum(a.elapsed_time(b) for a,b in self.event_pairs)/1000,
                "calls":len(self.event_pairs),
                "scope":"PIBT and optional RSE, including host dispatch/synchronization gaps; nested in inference_and_shield"}
