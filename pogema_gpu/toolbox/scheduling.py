"""CPU-only admission of environments to a bounded GPU batch."""


class AgentBudgetQueue:
    """Deterministic first-fit queue bounded by agents, state bytes and slots."""
    def __init__(self, agents, state_bytes, *, max_agents, max_state_bytes, max_envs):
        if any(type(v) is not int or v < 1 for v in (max_agents, max_state_bytes, max_envs)):
            raise ValueError("budgets must be positive integers")
        if len(agents) != len(state_bytes):
            raise ValueError("unaligned task costs")
        if any(type(n) is not int or not 0 < n <= max_agents for n in agents):
            raise ValueError("one task exceeds agent budget or is invalid")
        if any(type(n) is not int or not 0 < n <= max_state_bytes for n in state_bytes):
            raise ValueError("one task exceeds state budget or is invalid")
        self.agents, self.state_bytes = list(agents), list(state_bytes)
        self.max_agents, self.max_state_bytes, self.max_envs = max_agents, max_state_bytes, max_envs
        self.pending, self.active = list(range(len(agents))), []

    def admit(self, ready=None):
        added=[]
        agents=sum(self.agents[i] for i in self.active)
        memory=sum(self.state_bytes[i] for i in self.active)
        for i in list(self.pending):
            if len(self.active) >= self.max_envs: break
            if ready is not None and not ready(i):
                continue
            if agents+self.agents[i] > self.max_agents or memory+self.state_bytes[i] > self.max_state_bytes:
                continue
            self.pending.remove(i); self.active.append(i); added.append(i)
            agents += self.agents[i]; memory += self.state_bytes[i]
        return added

    def finish(self, indices):
        indices=list(indices)
        if len(set(indices)) != len(indices) or any(i not in self.active for i in indices):
            raise ValueError("finished tasks must be unique and active")
        self.active=[i for i in self.active if i not in indices]
