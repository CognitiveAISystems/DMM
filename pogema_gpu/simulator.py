"""GPU-resident POGEMA state with task coordinates mapped to a padded grid."""
import torch

from .kernels import extension
from .tasks import _unique
from .movement import NativeSoftResolver, ResolvedMove, validate_resolved_move


def matrix(rows):
    return [[int(cell == "#") for cell in row] for row in rows]


class CUDABatch:
    def __init__(self, tasks, *, device="cuda:0", map_shape=None):
        self.tasks = list(tasks)
        _unique(self.tasks)
        if not self.tasks or len({(t.num_agents, t.obs_radius) for t in self.tasks}) != 1:
            raise ValueError("CUDA cohorts require matching agent counts and padding")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDABatch requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.b, self.n = len(self.tasks), self.tasks[0].num_agents
        self.radius = self.tasks[0].obs_radius
        minimum = (max(t.height for t in self.tasks), max(t.width for t in self.tasks))
        if map_shape is None:
            map_shape = minimum
        if (len(map_shape) != 2 or any(type(v) is not int or v < m for v, m in zip(map_shape, minimum))):
            raise ValueError("map_shape must contain two integer dimensions fitting every initial task")
        self.height, self.width = (v + 2*self.radius for v in map_shape)
        self.cells = self.height*self.width
        if self.n > 8192 or self.cells > 1_048_576:
            raise ValueError("the native CUDA kernel supports N<=8192 and <=1M padded cells")
        if self.b*(3*self.cells+4*self.n) >= 2**31:
            raise ValueError("batch exceeds the native kernel's int32 scratch indexing limit")
        self.kernel = extension()
        self.walls = torch.ones((self.b,self.height,self.width), dtype=torch.bool, device=self.device)
        self.positions = torch.empty((self.b,self.n), dtype=torch.long, device=self.device)
        self.goals = torch.empty_like(self.positions)
        self.executed = torch.zeros_like(self.positions)
        self.steps = torch.zeros(self.b, dtype=torch.long, device=self.device)
        self.horizons = torch.empty_like(self.steps)
        self.finished = torch.zeros(self.b, dtype=torch.bool, device=self.device)
        self.solved = torch.zeros_like(self.finished)
        self.repairs = torch.zeros_like(self.steps)
        self.solve_time = torch.full_like(self.positions, -1)
        self.native_resolver = NativeSoftResolver(self)  # scratch allocated only if used
        self.generations = [0]*self.b
        self.state_version = 0  # Host-side cache invalidation; no device readback.
        self.reset_at(range(self.b), self.tasks)

    def slots(self, ids=None):
        ids = list(range(self.b)) if ids is None else list(ids)
        if len(set(ids)) != len(ids) or any(type(i) is not int or not 0<=i<self.b for i in ids):
            raise ValueError("slot IDs must be unique valid Python integers")
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def reset_at(self, ids, tasks):
        ids, tasks = list(ids), list(tasks)
        slots = self.slots(ids)
        if len(ids)!=len(tasks):
            raise ValueError("reset IDs and tasks must match")
        proposed_tasks = self.tasks.copy()
        for i, task in zip(ids,tasks):
            if (task.num_agents,task.obs_radius)!=(self.n,self.radius) or task.height+2*self.radius>self.height or task.width+2*self.radius>self.width:
                raise ValueError("reset task does not fit the allocated cohort")
            proposed_tasks[i] = task
        _unique(proposed_tasks)
        for i, task in zip(ids,tasks):
            self.walls[i].fill_(True)
            saved = torch.tensor(matrix(task.observation_obstacles), dtype=torch.bool, device=self.device)
            self.walls[i,:saved.shape[0],:saved.shape[1]] = saved
            for tensor, points in ((self.positions,task.starts),(self.goals,task.goals)):
                tensor[i] = torch.tensor([(r+self.radius)*self.width+c+self.radius for r,c in points], device=self.device)
            self.generations[i] += 1
            self.horizons[i] = task.horizon
        self.tasks = proposed_tasks
        for tensor in (self.executed,self.steps,self.finished,self.solved,self.repairs):
            tensor[slots] = 0
        self.solve_time[slots] = -1
        self.state_version += 1

    def validate_actions(self, actions):
        if actions.shape != self.positions.shape or actions.dtype!=torch.long or actions.device!=self.device or not actions.is_contiguous():
            raise ValueError("actions must be contiguous same-device int64 [B,N]")
        # Device assertion instead of a per-step .item()/CPU synchronization.
        torch._assert_async(((actions>=0)&(actions<5)).all(), "actions must be in [0,4]")
    def step(self, actions):
        """Default standalone API: native soft resolution followed by commit."""
        return self.commit(self.native_resolver.resolve(actions))

    def commit(self, move, *, validate=False):
        """Apply one resolver's output; no collision resolution or BFS here."""
        if not isinstance(move, ResolvedMove) or move.batch is not self or move.state_version != self.state_version:
            raise ValueError("resolved move belongs to a different or stale batch state")
        for tensor in (move.positions, move.actions, move.submitted):
            if (tensor.shape != self.positions.shape or tensor.dtype != torch.long
                    or tensor.device != self.device or not tensor.is_contiguous()):
                raise ValueError("resolved tensors must be contiguous same-device int64 [B,N]")
        if validate:
            validate_resolved_move(move)
        self.kernel.commit(self.positions,self.goals,move.positions,move.submitted,move.actions,self.executed,
                           self.steps,self.horizons,self.finished,self.solved,self.repairs,self.solve_time)
        self.state_version += 1
        return self.executed

    def export(self):
        """Explicit synchronization boundary, for terminal records or diagnostics."""
        state = {name: getattr(self, name).cpu().tolist() for name in EXPORT_FIELDS}
        return export_transitions(self.tasks, self.width, self.radius, state)


EXPORT_FIELDS = ("positions", "goals", "steps", "finished", "solved", "repairs", "solve_time", "executed")


def export_transitions(tasks, width, radius, state):
    """Shared CPU formatting for live state and terminal snapshots before refill."""
    positions, goals = state["positions"], state["goals"]
    steps, finished, solved, repairs = (state[k] for k in ("steps", "finished", "solved", "repairs"))
    solve_times, executed = state["solve_time"], state["executed"]
    results = []
    for e, task in enumerate(tasks):
        n = task.num_agents
        metrics = None
        if finished[e]:
            metrics = {"ISR":sum(p==g for p,g in zip(positions[e],goals[e]))/n,
                       "CSR":float(solved[e]), "ep_length":steps[e],
                       "SoC":sum(solve_times[e])+n, "makespan":max(solve_times[e])+1,
                       "a_collisions":repairs[e], "o_collisions":0}
        results.append({"positions":[[p//width-radius,p%width-radius] for p in positions[e]],
                        "rewards":[float(solved[e])]*n, "terminated":[solved[e]]*n,
                        "truncated":[steps[e]>=task.horizon]*n, "executed_actions":executed[e], "metrics":metrics})
    return results
