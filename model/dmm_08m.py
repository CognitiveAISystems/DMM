"""DMM-08M with consensus/message passing.

The observation encoder emits 25 spatial tokens and 13 neighbor tokens. Each
communication round gathers per-neighbor dynamic messages, then a learned
action/message query attends to all 38 observation tokens and 13 messages.
The expensive static key/value projections are computed once per MAPF step
and reused by every round.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NUM_ACTIONS = 5


class FusionFriendlyResidualConvBlock(nn.Module):
    """ConvNeXt-like spatial block arranged for Inductor fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.norm = nn.RMSNorm(channels)
        self.expand = nn.Linear(channels, 2 * channels, bias=False)
        self.project = nn.Linear(channels, channels, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise(x).permute(0, 2, 3, 1)
        value, gate = self.expand(self.norm(x)).chunk(2, dim=-1)
        x = self.project(value * F.silu(gate)).permute(0, 3, 1, 2)
        return residual + x


class ExplicitSDPASelfAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, tokens: Tensor, padding_mask: Tensor) -> Tensor:
        batch, length, width = tokens.shape
        qkv = self.qkv(tokens).view(
            batch, length, 3, self.heads, self.head_width
        )
        query, key, value = qkv.unbind(dim=2)
        allowed = (~padding_mask).unsqueeze(1).unsqueeze(1)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=allowed,
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.output(attended)


class RelationalMixerBlock(nn.Module):
    """Mix compact spatial and neighbor tokens before communication."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(width)
        self.attention = ExplicitSDPASelfAttention(width, heads)
        self.ff_norm = nn.RMSNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 2 * width, bias=False),
            nn.SiLU(),
            nn.Linear(2 * width, width, bias=False),
        )

    def forward(self, tokens: Tensor, padding_mask: Tensor) -> Tensor:
        tokens = tokens + self.attention(
            self.attention_norm(tokens), padding_mask
        )
        tokens = tokens + self.ff(self.ff_norm(tokens))
        return tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)


@dataclass
class DMM08MConfig:
    block_size: int = 256
    vocab_size: int = 67
    field_of_view_size: int = 121
    agent_info_size: int = 10
    max_num_neighbors: int = 13
    empty_token_code: int = 66
    empty_connection_code: int = -1

    width: int = 96
    heads: int = 4
    conv_blocks: int = 3
    mixer_blocks: int = 2
    spatial_size: int = 5
    spatial_mode: str = "pooled"
    local_spatial_size: int = 7
    neighbor_tokenization: str = "flat"
    communication_query_blocks: int = 2
    communication_hidden_multiplier: int = 2
    dropout: float = 0.0
    bias: bool = False

    dt: float = 0.25
    tau: float = 1.0
    n_comm_rounds: int = 4

    # Supervised-training controls shared with DMM's training loop.
    dirichlet_tf_on: bool = True
    dirichlet_tf_beta: float = 1.0
    dirichlet_tf_beta_final: float = 0.8
    dirichlet_tf_anneal_steps: int = 100_000
    round_tf_on: bool = True
    round_tf_beta: float = 1.0
    round_tf_beta_final: float = 0.8
    round_tf_anneal_steps: int = 100_000


class StructuredEncoder08M(nn.Module):
    """Encode an observation as exactly 25 spatial + 13 neighbor tokens."""

    def __init__(self, config: DMM08MConfig) -> None:
        super().__init__()
        if config.block_size != 256:
            raise ValueError("StructuredEncoder08M requires 256 input tokens")
        if config.field_of_view_size != 121:
            raise ValueError("StructuredEncoder08M requires an 11x11 grid")
        if config.max_num_neighbors != 13 or config.agent_info_size != 10:
            raise ValueError("StructuredEncoder08M requires 13x10 neighbor data")
        if not 1 <= config.spatial_size <= 11:
            raise ValueError("spatial_size must lie in [1, 11]")
        if config.spatial_mode not in {"pooled", "multiscale"}:
            raise ValueError("spatial_mode must be 'pooled' or 'multiscale'")
        if not 1 <= config.local_spatial_size <= 11:
            raise ValueError("local_spatial_size must lie in [1, 11]")
        if config.local_spatial_size % 2 == 0:
            raise ValueError("local_spatial_size must be odd and centred")
        if config.neighbor_tokenization not in {"flat", "split"}:
            raise ValueError("neighbor_tokenization must be 'flat' or 'split'")

        width = config.width
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, width)
        self.grid_stem = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.grid_blocks = nn.Sequential(
            *(FusionFriendlyResidualConvBlock(width) for _ in range(config.conv_blocks))
        )
        if config.neighbor_tokenization == "flat":
            self.neighbor_mlp = nn.Sequential(
                nn.Linear(config.agent_info_size * width, 2 * width, bias=False),
                nn.SiLU(),
                nn.Linear(2 * width, width, bias=False),
            )
            self.neighbor_subtypes = None
        else:
            # [relative position, relative goal] and
            # [five previous actions, greedy-action mask] carry different
            # semantics and receive separate tokens instead of being flattened.
            self.neighbor_geometry_mlp = nn.Sequential(
                nn.Linear(4 * width, 2 * width, bias=False),
                nn.SiLU(),
                nn.Linear(2 * width, width, bias=False),
            )
            self.neighbor_motion_mlp = nn.Sequential(
                nn.Linear(6 * width, 2 * width, bias=False),
                nn.SiLU(),
                nn.Linear(2 * width, width, bias=False),
            )
            self.neighbor_subtypes = nn.Parameter(torch.randn(2, width) * 0.02)
        spatial_tokens = config.spatial_size ** 2
        if config.spatial_mode == "multiscale":
            spatial_tokens += config.local_spatial_size ** 2
        self.spatial_token_count = spatial_tokens
        self.neighbor_token_count = 13 * (
            2 if config.neighbor_tokenization == "split" else 1
        )
        self.spatial_positions = nn.Parameter(
            torch.randn(spatial_tokens, width) * 0.02
        )
        self.neighbor_slots = nn.Parameter(torch.randn(13, width) * 0.02)
        self.token_types = nn.Parameter(torch.randn(2, width) * 0.02)
        self.mixer = nn.ModuleList(
            RelationalMixerBlock(width, config.heads)
            for _ in range(config.mixer_blocks)
        )
        self.output_norm = nn.RMSNorm(width)

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        if observations.shape[-1] != 256:
            raise ValueError(f"Expected 256 tokens, got {observations.shape}")
        flat = observations.reshape(-1, 256)
        width = self.config.width

        grid = self.token_embedding(flat[:, :121])
        grid = grid.transpose(1, 2).reshape(-1, width, 11, 11)
        grid = self.grid_blocks(self.grid_stem(grid))
        coarse = F.adaptive_avg_pool2d(
            grid, (self.config.spatial_size, self.config.spatial_size)
        ).flatten(2).transpose(1, 2)
        if self.config.spatial_mode == "multiscale":
            local_size = self.config.local_spatial_size
            offset = (11 - local_size) // 2
            local = grid[
                :, :, offset : offset + local_size, offset : offset + local_size
            ].flatten(2).transpose(1, 2)
            grid = torch.cat((coarse, local), dim=1)
        else:
            grid = coarse
        grid = grid + self.spatial_positions.unsqueeze(0) + self.token_types[0]

        neighbor_ids = flat[:, 121:251].reshape(-1, 13, 10)
        neighbor_padding = neighbor_ids.eq(self.config.empty_token_code).all(-1)
        embedded_neighbors = self.token_embedding(neighbor_ids)
        if self.config.neighbor_tokenization == "flat":
            neighbors = self.neighbor_mlp(embedded_neighbors.flatten(2))
            neighbors = (
                neighbors
                + self.neighbor_slots.unsqueeze(0)
                + self.token_types[1]
            )
        else:
            geometry = self.neighbor_geometry_mlp(
                embedded_neighbors[:, :, :4].flatten(2)
            )
            motion = self.neighbor_motion_mlp(
                embedded_neighbors[:, :, 4:].flatten(2)
            )
            slots = self.neighbor_slots.unsqueeze(0)
            geometry = geometry + slots + self.token_types[1] + self.neighbor_subtypes[0]
            motion = motion + slots + self.token_types[1] + self.neighbor_subtypes[1]
            neighbors = torch.stack((geometry, motion), dim=2).flatten(1, 2)
            neighbor_padding = neighbor_padding.unsqueeze(-1).expand(
                -1, -1, 2
            ).flatten(1, 2)

        tokens = torch.cat((grid, neighbors), dim=1)
        spatial_padding = torch.zeros(
            (flat.shape[0], self.spatial_token_count),
            dtype=torch.bool,
            device=flat.device,
        )
        padding = torch.cat((spatial_padding, neighbor_padding), dim=1)
        for block in self.mixer:
            tokens = block(tokens, padding)
        tokens = self.output_norm(tokens)
        tokens = tokens.masked_fill(padding.unsqueeze(-1), 0.0)
        return tokens, padding


class QueryCrossAttentionBlock(nn.Module):
    """One query-only cross-attention/SwiGLU block with its own K/V space."""

    def __init__(self, config: DMM08MConfig) -> None:
        super().__init__()
        width = config.width
        hidden = config.communication_hidden_multiplier * width
        self.heads = config.heads
        self.head_width = width // config.heads
        self.memory_norm = nn.RMSNorm(width)
        self.static_key_value = nn.Linear(width, 2 * width, bias=False)
        self.message_key_value = nn.Linear(width, 2 * width, bias=False)
        self.query_norm = nn.RMSNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.attention_output_norm = nn.RMSNorm(width)
        self.ff_expand = nn.Linear(width, 2 * hidden, bias=False)
        self.ff_project = nn.Linear(hidden, width, bias=False)
        self.output_norm = nn.RMSNorm(width)

    def prepare_static_memory(
        self, observation_tokens: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch_agents, token_count, width = observation_tokens.shape
        key_value = self.static_key_value(
            self.memory_norm(observation_tokens)
        ).view(
            batch_agents,
            token_count,
            2,
            self.heads,
            self.head_width,
        )
        key, value = key_value.unbind(dim=2)
        return key.transpose(1, 2), value.transpose(1, 2)

    def forward(
        self,
        query_token: Tensor,
        messages: Tensor,
        static_key: Tensor,
        static_value: Tensor,
        allowed: Tensor,
    ) -> Tensor:
        batch_agents, message_count, width = messages.shape
        dynamic_key_value = self.message_key_value(
            self.memory_norm(messages)
        ).view(
            batch_agents,
            message_count,
            2,
            self.heads,
            self.head_width,
        )
        dynamic_key, dynamic_value = dynamic_key_value.unbind(dim=2)
        key = torch.cat((static_key, dynamic_key.transpose(1, 2)), dim=2)
        value = torch.cat((static_value, dynamic_value.transpose(1, 2)), dim=2)
        query = self.query(self.query_norm(query_token)).view(
            batch_agents, 1, self.heads, self.head_width
        ).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed, dropout_p=0.0
        )
        attended = attended.transpose(1, 2).reshape(batch_agents, 1, width)
        feature = query_token + self.output(attended)
        normalized = self.attention_output_norm(feature)
        ff_value, ff_gate = self.ff_expand(normalized).chunk(2, dim=-1)
        feature = feature + self.ff_project(ff_value * F.silu(ff_gate))
        return self.output_norm(feature)


class CachedCrossAttentionCommunication(nn.Module):
    """DMM communication decoder with cached observation key/value tensors.

    Each communication round follows:

    ``h + z -> gather neighbor slots -> shared action/message feature -> logits,h``.

    A small query stack uses fused SDPA over 38 static observation tokens and
    up to 13 fresh message tokens instead of repeatedly running full
    self-attention over their concatenation.
    """

    def __init__(self, config: DMM08MConfig) -> None:
        super().__init__()
        width = config.width
        if width % config.heads:
            raise ValueError("width must be divisible by heads")
        self.max_num_neighbors = config.max_num_neighbors
        self.empty_message = nn.Parameter(torch.randn(width) * 0.02)
        self.slot_embedding = nn.Parameter(
            torch.randn(config.max_num_neighbors, width) * 0.02
        )
        self.query_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        if config.communication_query_blocks < 1:
            raise ValueError("communication_query_blocks must be positive")
        self.blocks = nn.ModuleList(
            QueryCrossAttentionBlock(config)
            for _ in range(config.communication_query_blocks)
        )

    @staticmethod
    def collect(sender: Tensor, connections: Tensor, empty: Tensor) -> Tensor:
        """Gather sender states; connection -1 maps to a learned empty row."""
        batch, agents, width = sender.shape
        _, _, neighbors = connections.shape
        source = torch.cat(
            (empty.reshape(1, 1, width).expand(batch, 1, width), sender),
            dim=1,
        )
        indices = (connections.long() + 1).unsqueeze(-1).expand(
            batch, agents, neighbors, width
        )
        return torch.gather(
            source.unsqueeze(2).expand(batch, agents + 1, neighbors, width),
            dim=1,
            index=indices,
        )

    def prepare_static_memory(
        self, observation_tokens: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Project 38 static tokens once for every query block.

        The leading dimension indexes blocks and remains outside the four-round
        communication loop.
        """
        projected = [
            block.prepare_static_memory(observation_tokens)
            for block in self.blocks
        ]
        return (
            torch.stack([key for key, _ in projected]),
            torch.stack([value for _, value in projected]),
        )

    def forward(
        self,
        agent_to_message: Tensor,
        connections: Tensor,
        static_key: Tensor,
        static_value: Tensor,
        observation_padding: Tensor,
    ) -> Tensor:
        batch, agents, width = agent_to_message.shape
        batch_agents = batch * agents
        valid = connections.ge(0)
        messages = self.collect(
            agent_to_message, connections, self.empty_message
        )
        messages = messages + self.slot_embedding[: connections.shape[-1]]
        messages = messages.reshape(batch_agents, connections.shape[-1], width)
        query_token = self.query_token.expand(batch_agents, 1, width)
        padding = torch.cat(
            (observation_padding, ~valid.reshape(batch_agents, -1)), dim=1
        )
        allowed = (~padding).unsqueeze(1).unsqueeze(1)
        for block_index, block in enumerate(self.blocks):
            query_token = block(
                query_token,
                messages,
                static_key[block_index],
                static_value[block_index],
                allowed,
            )
        return query_token[:, 0]

    def forward_collected(
        self,
        messages: Tensor,
        connections: Tensor,
        static_key: Tensor,
        static_value: Tensor,
        observation_padding: Tensor,
    ) -> Tensor:
        """Decode receivers whose global sender messages were gathered by a shard."""
        batch, agents, neighbors, width = messages.shape
        count = batch * agents
        messages = messages + self.slot_embedding[:neighbors]
        messages = messages.reshape(count, neighbors, width)
        padding = torch.cat(
            (observation_padding, ~connections.ge(0).reshape(count, neighbors)),
            dim=1,
        )
        allowed = (~padding).unsqueeze(1).unsqueeze(1)
        query_token = self.query_token.expand(count, 1, width)
        for block_index, block in enumerate(self.blocks):
            query_token = block(
                query_token,
                messages,
                static_key[block_index],
                static_value[block_index],
                allowed,
            )
        return query_token[:, 0]


class DMM08M(nn.Module):
    """Scratch DMM with a structured encoder and lightweight GNN rounds."""

    def __init__(self, config: DMM08MConfig) -> None:
        super().__init__()
        self.config = config
        self.num_actions = NUM_ACTIONS
        self.representation_encoder = StructuredEncoder08M(config)
        self.z_proj = nn.Linear(NUM_ACTIONS, config.width, bias=False)
        self.communication = CachedCrossAttentionCommunication(config)
        self.pi_head = nn.Linear(config.width, NUM_ACTIONS, bias=False)
        self.msg_head = nn.Linear(config.width, config.width, bias=False)

    @staticmethod
    def _centred_log_ohe(actions: Tensor, dtype: torch.dtype) -> Tensor:
        one_hot = F.one_hot(actions, NUM_ACTIONS).to(dtype)
        target = F.log_softmax((one_hot + 1e-8).log(), dim=-1)
        return target - target.mean(-1, keepdim=True)

    @staticmethod
    def _dirichlet_z0(
        count: int,
        device: torch.device,
        dtype: torch.dtype,
        alpha: float,
        targets: Tensor | None = None,
        teacher_forcing_beta: float = 0.0,
    ) -> Tensor:
        concentration = torch.full((count, NUM_ACTIONS), alpha)
        if targets is not None and teacher_forcing_beta > 0.0:
            valid = targets.ne(-1).cpu()
            safe_targets = targets.clamp(min=0).cpu()
            use_teacher = torch.bernoulli(
                torch.full((count,), teacher_forcing_beta)
            ).bool() & valid
            target_concentration = F.one_hot(
                safe_targets, NUM_ACTIONS
            ).float()
            concentration[use_teacher] += target_concentration[use_teacher]
        probabilities = torch.distributions.Dirichlet(concentration).sample()
        z = probabilities.clamp(min=1e-8).log()
        z = z - z.mean(-1, keepdim=True)
        return z.to(device=device, dtype=dtype)

    @staticmethod
    def _prob_to_log_centered(probabilities: Tensor) -> Tensor:
        z = probabilities.float().clamp(min=1e-8).log()
        return (z - z.mean(-1, keepdim=True)).to(probabilities.dtype)

    def _encode(
        self, obs: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, agents, token_count = obs.shape
        tokens, padding = self.representation_encoder(
            obs.reshape(batch * agents, token_count)
        )
        static_key, static_value = self.communication.prepare_static_memory(tokens)
        return static_key, static_value, padding, tokens

    def _run_round(
        self,
        z: Tensor,
        h: Tensor,
        agent_chat_ids: Tensor,
        static_key: Tensor,
        static_value: Tensor,
        observation_padding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, agents, _ = agent_chat_ids.shape
        width = h.shape[-1]
        agent_to_message = h + self.z_proj(z).reshape(
            batch, agents, width
        ).to(h.dtype)
        feature = self.communication(
            agent_to_message,
            agent_chat_ids,
            static_key,
            static_value,
            observation_padding,
        )
        logits = self.pi_head(feature)
        h_new = self.msg_head(feature).reshape(batch, agents, width)
        return logits, h_new

    def forward(
        self,
        observations: Tensor,
        agent_chat_ids: Tensor,
        target_actions: Tensor,
        dirichlet_tf_beta: float | None = None,
        round_tf_beta: float | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Supervised forward contract.

        The trainer owns teacher-forcing schedules, DDP,
        optimisation, and checkpointing.  This method only implements the
        model-side per-round categorical objective expected by that trainer.
        """
        batch, agents, _ = observations.shape
        count = batch * agents
        targets = target_actions.reshape(count)
        static_key, static_value, observation_padding, _ = self._encode(
            observations
        )
        dtype = static_key.dtype

        dirichlet_beta = (
            self.config.dirichlet_tf_beta
            if dirichlet_tf_beta is None
            else dirichlet_tf_beta
        )
        if not self.config.dirichlet_tf_on:
            dirichlet_beta = 0.0
        round_beta = (
            self.config.round_tf_beta
            if round_tf_beta is None
            else round_tf_beta
        )
        if not self.config.round_tf_on:
            round_beta = 0.0

        z = self._dirichlet_z0(
            count,
            observations.device,
            dtype,
            1.0,
            targets=targets,
            teacher_forcing_beta=dirichlet_beta,
        )
        h = self.communication.empty_message.reshape(1, 1, -1).expand(
            batch, agents, -1
        ).to(dtype)
        per_round_losses: list[Tensor] = []

        for _ in range(self.config.n_comm_rounds):
            logits, h = self._run_round(
                z,
                h,
                agent_chat_ids,
                static_key,
                static_value,
                observation_padding,
            )
            per_round_losses.append(
                F.cross_entropy(logits, targets, ignore_index=-1)
            )
            with torch.no_grad():
                sampled = torch.distributions.Categorical(logits=logits).sample()
                if round_beta > 0.0:
                    use_teacher = torch.bernoulli(
                        torch.full(
                            (count,), round_beta, device=observations.device
                        )
                    ).bool() & targets.ne(-1)
                    votes = torch.where(
                        use_teacher, targets.clamp(min=0), sampled
                    )
                else:
                    votes = sampled
            target = self._centred_log_ohe(votes, dtype)
            z = z + self.config.dt * (target - z)

        return torch.stack(per_round_losses).mean(), per_round_losses

    @torch.no_grad()
    def rollout_act(
        self,
        obs: Tensor,
        agent_chat_ids: Tensor,
        rollout_tau: float = 1.0,
        z0_in: Tensor | None = None,
        dir_alpha: float = 1.0,
    ):
        batch, agents, _ = obs.shape
        count = batch * agents
        static_key, static_value, observation_padding, _ = self._encode(obs)
        dtype = static_key.dtype
        h = self.communication.empty_message.reshape(1, 1, -1).expand(
            batch, agents, -1
        ).to(dtype)
        if z0_in is None:
            z0 = self._dirichlet_z0(count, obs.device, dtype, dir_alpha)
        else:
            z0 = self._prob_to_log_centered(
                z0_in.to(device=obs.device, dtype=dtype)
            )
        z = z0.clone()
        votes_list = []
        log_pi_list = []
        for _ in range(self.config.n_comm_rounds):
            logits, h = self._run_round(
                z,
                h,
                agent_chat_ids,
                static_key,
                static_value,
                observation_padding,
            )
            logits = logits.nan_to_num(0.0)
            vote = torch.distributions.Categorical(
                logits=logits / rollout_tau
            ).sample()
            log_pi = F.log_softmax(logits, dim=-1).gather(
                -1, vote.unsqueeze(-1)
            ).squeeze(-1)
            votes_list.append(vote)
            log_pi_list.append(log_pi)
            target = self._centred_log_ohe(vote, dtype)
            z = z + self.config.dt * (target - z)
        return (
            z.argmax(-1),
            torch.stack(votes_list, dim=-1),
            torch.stack(log_pi_list, dim=-1),
            z0,
            z,
        )

    def forward_micpo(
        self,
        obs: Tensor,
        agent_chat_ids: Tensor,
        stored_votes: Tensor,
        stored_z0: Tensor,
    ):
        batch, agents, _ = obs.shape
        static_key, static_value, observation_padding, _ = self._encode(obs)
        dtype = static_key.dtype
        z = stored_z0.to(dtype)
        h = self.communication.empty_message.reshape(1, 1, -1).expand(
            batch, agents, -1
        ).to(dtype)
        log_pi_list = []
        logits_list = []
        for round_index in range(self.config.n_comm_rounds):
            logits, h = self._run_round(
                z,
                h,
                agent_chat_ids,
                static_key,
                static_value,
                observation_padding,
            )
            vote = stored_votes[:, round_index]
            log_pi = F.log_softmax(logits, dim=-1).gather(
                -1, vote.unsqueeze(-1)
            ).squeeze(-1)
            log_pi_list.append(log_pi)
            logits_list.append(logits)
            target = self._centred_log_ohe(vote.detach(), dtype)
            z = z + self.config.dt * (target - z)
        return torch.stack(log_pi_list, -1), torch.stack(logits_list, 1)

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = {"fused": True} if fused and device_type == "cuda" else {}
        return torch.optim.AdamW(
            groups, lr=learning_rate, betas=betas, **extra
        )
