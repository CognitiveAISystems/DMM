"""Shard-aware inference using the single DMM-08M model implementation."""

from types import SimpleNamespace

import torch
from torch import Tensor

from evaluation.models import load_checkpoint_model
from model.dmm_08m import DMM08M


class ShardedPolicy:
    def __init__(self, net: DMM08M, agent_chunk_size: int):
        self.net = net.eval()
        self.agent_chunk_size = agent_chunk_size

    def _encode_active(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        net = self.net
        count = observations.shape[1]
        keys = values = padding = None
        for start in range(0, count, self.agent_chunk_size):
            end = min(start + self.agent_chunk_size, count)
            tokens, current_padding = net.representation_encoder(
                observations[0, start:end]
            )
            key, value = net.communication.prepare_static_memory(tokens)
            if keys is None:
                keys = torch.empty(
                    (key.shape[0], count, *key.shape[2:]),
                    dtype=key.dtype, device=key.device,
                )
                values = torch.empty_like(keys)
                padding = torch.empty(
                    (count, current_padding.shape[1]),
                    dtype=torch.bool, device=current_padding.device,
                )
            keys[:, start:end] = key
            values[:, start:end] = value
            padding[start:end] = current_padding
        return keys, values, padding

    @staticmethod
    def _collect(global_sender: Tensor, connections: Tensor, empty: Tensor) -> Tensor:
        batch, _, width = global_sender.shape
        receivers, neighbors = connections.shape[1:]
        source = torch.cat(
            (empty.reshape(1, 1, width).expand(batch, 1, width), global_sender),
            dim=1,
        )
        indices = (connections.long() + 1).reshape(
            batch, receivers * neighbors, 1
        ).expand(-1, -1, width)
        return torch.gather(source, 1, indices).reshape(
            batch, receivers, neighbors, width
        )

    @torch.no_grad()
    def _act(
        self,
        observations: Tensor,
        connections: Tensor,
        gather_sender,
        *,
        active_mask: Tensor | None,
        stochastic: bool,
        generator: torch.Generator | None,
        rollout_tau: float = 1.0,
    ) -> SimpleNamespace:
        if observations.shape[0] != 1:
            raise ValueError("sharded inference requires a single environment")
        net = self.net
        count = observations.shape[1]
        if active_mask is None:
            active_mask = torch.ones(count, dtype=torch.bool, device=observations.device)
        active_mask = active_mask.to(device=observations.device, dtype=torch.bool)
        if active_mask.shape != (count,):
            raise ValueError("active_mask must have shape [local_agents]")
        active_ids = active_mask.nonzero(as_tuple=False).flatten()
        if active_ids.numel():
            keys, values, padding = self._encode_active(observations[:, active_ids])
            dtype = keys.dtype
        else:
            keys = values = padding = None
            dtype = (torch.get_autocast_dtype("cuda") if observations.is_cuda
                     and torch.is_autocast_enabled("cuda")
                     else net.communication.empty_message.dtype)
        z = torch.zeros(count, 5, dtype=dtype, device=observations.device)
        if stochastic:
            concentration = torch.full(
                (count, 5), 1.0, dtype=torch.float32, device=observations.device
            )
            gamma = torch._standard_gamma(concentration, generator=generator)
            probabilities = gamma / gamma.sum(dim=-1, keepdim=True)
            sampled = probabilities.clamp_min(1e-8).log()
            sampled -= sampled.mean(dim=-1, keepdim=True)
            z[active_ids] = sampled[active_ids].to(dtype)
        width = net.config.width
        h = net.communication.empty_message.reshape(1, 1, width).expand(
            1, count, width
        ).to(dtype)
        active_connections = connections[:, active_ids]
        for _ in range(net.config.n_comm_rounds):
            local_sender = h + net.z_proj(z).reshape(1, count, width).to(h.dtype)
            global_sender = gather_sender(local_sender.contiguous())
            logits = torch.empty(
                (active_ids.numel(), 5), dtype=h.dtype, device=h.device
            )
            h_new = h.clone()
            for start in range(0, active_ids.numel(), self.agent_chunk_size):
                end = min(start + self.agent_chunk_size, active_ids.numel())
                selected = active_connections[:, start:end]
                messages = self._collect(
                    global_sender, selected, net.communication.empty_message
                )
                feature = net.communication.forward_collected(
                    messages, selected, keys[:, start:end],
                    values[:, start:end], padding[start:end],
                )
                logits[start:end] = net.pi_head(feature)
                h_new[:, active_ids[start:end]] = net.msg_head(feature).reshape(
                    1, end - start, width
                ).to(h_new.dtype)
            h = h_new
            if stochastic:
                action_probabilities = torch.softmax(
                    logits.float() / rollout_tau, dim=-1
                )
                uniforms = torch.rand(
                    count, generator=generator, device=observations.device
                )[active_ids]
                vote = (
                    (uniforms[:, None] > action_probabilities.cumsum(dim=-1))
                    .sum(dim=-1).clamp_max(4)
                )
            else:
                vote = logits.nan_to_num(0.0).argmax(dim=-1)
            target = net._centred_log_ohe(vote, dtype)
            updated = z[active_ids] + net.config.dt * (target - z[active_ids])
            if updated.dtype != z.dtype:
                z = z.to(updated.dtype)
            z[active_ids] = updated
        return SimpleNamespace(logits=z.reshape(1, count, 5).float())

    def stochastic_act_sharded(
        self, observations, connections, gather_sender, generator,
        agent_chunk_size=None, active_mask=None, rollout_tau=1.0,
    ):
        if agent_chunk_size is not None:
            self.agent_chunk_size = int(agent_chunk_size)
        return self._act(
            observations, connections, gather_sender, active_mask=active_mask,
            stochastic=True, generator=generator, rollout_tau=rollout_tau,
        )

    def deterministic_zero_act_sharded(
        self, observations, connections, gather_sender,
        agent_chunk_size=None, active_mask=None,
    ):
        if agent_chunk_size is not None:
            self.agent_chunk_size = int(agent_chunk_size)
        return self._act(
            observations, connections, gather_sender, active_mask=active_mask,
            stochastic=False, generator=None,
        )


def deployment_policy_from_checkpoint(
    path: str, device: torch.device, n_comm_rounds: int = 4,
    agent_chunk_size: int = 32768,
) -> tuple[ShardedPolicy, str]:
    net, config, _ = load_checkpoint_model("DMM-MICPO-08M", path, device)
    if config.n_comm_rounds != n_comm_rounds:
        raise ValueError("million-agent communication rounds differ from the release model")
    return ShardedPolicy(net, agent_chunk_size), "dmm08m"
