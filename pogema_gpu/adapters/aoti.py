"""Persistent use of the unchanged, evaluated whole-policy AOTI package.

No model conversion, re-compilation, CPU Dirichlet, or per-round multinomial.
Independent environments are packed along the dynamic agent axis. Neighbor
indices are offset per environment; sampled votes draw per-episode noise, while
argmax votes do not consume the episode RNG.
"""
import torch
import copy

from .cuda_runtime import CUDAPolicyRuntime
from .runtime import check_native_runtime


def native_noise(agents, generator, device, round_mode, rounds=4):
    if round_mode == "argmax":
        return (torch.zeros((1, agents, 5), device=device, dtype=torch.float32),
                torch.zeros((rounds, 1, agents, 5), device=device, dtype=torch.float32))
    if round_mode != "sample":
        raise ValueError(f"unsupported communication round mode: {round_mode}")
    initial = torch.empty((1, agents, 5), device=device, dtype=torch.float32)
    initial.exponential_(1.0, generator=generator)
    votes = torch.empty((rounds, 1, agents, 5), device=device, dtype=torch.float32)
    votes.exponential_(1.0, generator=generator)
    return initial, votes.log().neg()


def packed_native_call(call, obs, chat, generators, max_agents, round_mode, rounds=4):
    """Pack equal-N environments, never split an environment's communication graph.

    Only nonnegative local neighbor IDs receive an offset. Masked (-1) slots
    stay masked. Separate generators preserve each episode's draw sequence.
    Chunking respects the unchanged AOTI package's dynamic-agent upper bound.
    """
    b, n = obs.shape[:2]
    if b < 1 or n < 1 or n > max_agents or len(generators) != b:
        raise ValueError("invalid native packed batch or package agent limit")
    if chat.shape[:2] != (b, n):
        raise ValueError("native observation/neighbor batch mismatch")
    torch._assert_async(((chat >= -1) & (chat < n)).all(), "invalid local neighbor ID")
    outputs = []
    for begin in range(0, b, max_agents // n):
        end = min(b, begin + max_agents // n)
        count = end - begin
        noise = [native_noise(n, generators[i], obs.device, round_mode, rounds)
                 for i in range(begin, end)]
        if count == 1:
            args = (obs[begin:end], chat[begin:end], *noise[0])
        else:
            local = chat[begin:end]
            offsets = torch.arange(count, device=chat.device, dtype=chat.dtype)[:, None, None] * n
            neighbors = torch.where(local >= 0, local + offsets, local).reshape(1, count*n, -1)
            args = (obs[begin:end].reshape(1, count*n, -1), neighbors,
                    torch.cat([x[0] for x in noise], dim=1),
                    torch.cat([x[1] for x in noise], dim=2))
        outputs.append(call(*args).reshape(count, n, 5))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


class CUDAAOTIPolicy(CUDAPolicyRuntime):
    always_post_shield_history = True
    rounds = 4

    def episode_context(self, batch):
        """Own episode state, share the immutable loaded AOTI executable."""
        context = copy.copy(self)
        context.reset(batch)
        return context

    def propose_environments(self, contexts, inputs):
        """One disconnected graph for variable-N environments, no padded agents."""
        check_native_runtime(torch)
        if not contexts or len(contexts) != len(inputs):
            raise ValueError("contexts and inputs must be nonempty and aligned")
        counts = [obs.shape[1] for obs, _ in inputs]
        if sum(counts) > self.max_agents:
            raise ValueError("packed environments exceed AOTI agent limit")
        observations, neighbors, initial, votes = [], [], [], []
        offset = 0
        for ctx, (obs, chat), n in zip(contexts, inputs, counts):
            if ctx.call is not self.call or obs.shape[0] != 1 or chat.shape[:2] != (1, n):
                raise ValueError("each context must share this model and own one environment")
            torch._assert_async(((chat >= -1) & (chat < n)).all(), "invalid local neighbor ID")
            z, v = native_noise(n, ctx.samplers[0], obs.device, ctx.round_mode, ctx.rounds)
            observations.append(obs)
            neighbors.append(torch.where(chat >= 0, chat + offset, chat))
            initial.append(z); votes.append(v)
            offset += n
        probabilities = self.call(torch.cat(observations, 1), torch.cat(neighbors, 1),
                                  torch.cat(initial, 1), torch.cat(votes, 2)).reshape(1, offset, 5)
        torch._assert_async(torch.isfinite(probabilities).all(), "non-finite native probabilities")
        return [(p.argmax(-1), p if ctx.shield is not None else None)
                for ctx, p in zip(contexts, probabilities.split(counts, dim=1))]

    def reset(self, batch):
        if batch.n > self.max_agents:
            raise ValueError(f"policy requires N<={self.max_agents} per environment")
        if any(task.policy_seed != 0 for task in batch.tasks):
            raise ValueError("evaluation requires policy seed 0")
        super().reset(batch)

    @torch.inference_mode()
    def propose(self, inputs, ids):
        check_native_runtime(torch)
        obs, chat = inputs
        if obs.shape[0] != len(ids) or len(set(ids)) != len(ids):
            raise ValueError("native active-slot IDs must uniquely match input rows")
        probabilities = packed_native_call(self.call, obs, chat,
            [self.samplers[i] for i in ids], self.max_agents, self.round_mode, self.rounds)
        torch._assert_async(torch.isfinite(probabilities).all(), "non-finite native probabilities")
        actions = probabilities.argmax(-1)
        return actions, probabilities if self.shield is not None else None
