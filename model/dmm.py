"""
DMM — checkpoint-compatible Consensus Voting Model for pretraining and MICPO.

z initialisation:
  Rollout / inference: z_0 ~ Dirichlet(1), log-centered (unbiased).
  Replay (forward_micpo): z_0 = stored_z0 from rollout buffer.

Rounds 0..K-1: full message exchange.
           z  ← z + dt * (e(g_t) − z)   (EMA, centred)
           h  ← msg_head(feat)

MICPO methods:
    rollout_act(obs, agent_chat_ids, rollout_tau)
        Stochastic forward for rollout collection. Initialises z_0 from
        Dirichlet(1) noise. Returns actions, per-round votes, per-round
        log-probs, the sampled z_0, and final z.

    forward_micpo(obs, agent_chat_ids, stored_votes, stored_z0)
        Deterministic replay given stored votes and z_0. Gradients flow
        through all round logits and through h across rounds. z is treated
        as a constant (computed from stored discrete votes).
        Returns per-round log-probs (for IS ratio) and per-round logits
        (for KL and entropy).

Pretraining uses forward(obs, chat_ids, target_actions, ...) with per-round
cross-entropy and optional teacher forcing; it shares all checkpoint-bearing
layers with the MICPO and inference paths.
"""

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LayerNorm(nn.Module):
    """ RMSNorm. """

    def __init__(self, ndim, bias=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, input):
        return F.rms_norm(input, self.weight.shape, self.weight.to(input.dtype), 1e-5)


class MultiHeadAttention(nn.Module):
    def __init__(self, config, q_dim: int, kv_dim: int = None, n_embd: int = None):
        """
        Multi-Head Attention with flexible input/output dimensions.

        Args:
            config : config object with attributes n_embd, n_head, dropout, bias
            q_dim  : input dimension for queries
            kv_dim : input dimension for keys/values (defaults to q_dim)
            n_embd : internal embedding dim (defaults to config.n_embd)
        """
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd if n_embd is None else n_embd
        self.dropout = config.dropout

        if kv_dim is None:
            kv_dim = q_dim

        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        self.head_size = config.n_embd // self.n_head

        self.q_proj = nn.Linear(q_dim, 2 * self.n_embd, bias=config.bias)
        self.kv_proj = nn.Linear(kv_dim, 3 * self.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(self.n_embd, q_dim, bias=config.bias)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.attn_dropout = nn.Dropout(self.dropout)

        self.q_norm_1 = LayerNorm(self.head_size)
        self.k_norm_1 = LayerNorm(self.head_size)
        self.q_norm_2 = LayerNorm(self.head_size)
        self.k_norm_2 = LayerNorm(self.head_size)
        self.gr_norm = nn.GroupNorm(self.n_head, self.n_embd)

        self.lmb = nn.Parameter(-4 * torch.ones(1, self.n_head, 1, 1))
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x_q, x_kv=None, attn_bias=None):
        if x_kv is None:
            x_kv = x_q

        B, T_q, C_q = x_q.size()
        _, T_kv, _ = x_kv.size()

        q1, q2 = self.q_proj(x_q).chunk(2, dim=2)
        q1 = q1.view(B, T_q, self.n_head, self.head_size).transpose(1, 2)
        q2 = q2.view(B, T_q, self.n_head, self.head_size).transpose(1, 2)

        k1, k2, v = self.kv_proj(x_kv).chunk(3, dim=2)
        k1 = k1.view(B, T_kv, self.n_head, self.head_size).transpose(1, 2)
        k2 = k2.view(B, T_kv, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T_kv, self.n_head, self.head_size).transpose(1, 2)

        q1, q2 = self.q_norm_1(q1), self.q_norm_2(q2)
        k1, k2 = self.k_norm_1(k1), self.k_norm_2(k2)

        if attn_bias is not None:
            attn_bias = attn_bias.to(q1.dtype)

        if self.flash:
            y1 = F.scaled_dot_product_attention(
                q1, k1, v, attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0, is_causal=False,
            )
            y2 = F.scaled_dot_product_attention(
                q2, k2, v, attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0, is_causal=False,
            )
            y = y1 - F.softplus(self.lmb) * y2
        else:
            att1 = (q1 @ k1.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
            att2 = (q2 @ k2.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
            if attn_bias is not None:
                att1, att2 = att1 + attn_bias, att2 + attn_bias
            att1 = self.attn_dropout(F.softmax(att1, dim=-1))
            att2 = self.attn_dropout(F.softmax(att2, dim=-1))
            y = att1 @ v - F.softplus(self.lmb) * (att2 @ v)

        y = y.transpose(1, 2).contiguous().view(B, T_q, self.n_embd)
        y = self.gr_norm(y.transpose(1, 2)).transpose(1, 2)
        y = self.resid_dropout(self.c_proj(y))
        return y


class SwiGLU_MLP(nn.Module):
    def __init__(self, config, n_embd=None):
        super().__init__()
        self.n_embd = config.n_embd if n_embd is None else n_embd
        hidden_dim = 4 * self.n_embd
        self.c_fc = nn.Linear(self.n_embd, hidden_dim * 2, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, self.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x_a, x_b = self.c_fc(x).chunk(2, dim=-1)
        x = x_a * F.silu(x_b)
        return self.dropout(self.c_proj(x))


class Block(nn.Module):

    def __init__(self, config, q_dim: int, kv_dim: int = None, n_embd: int = None):
        super().__init__()
        self.n_embd = config.n_embd if n_embd is None else n_embd
        if kv_dim is None:
            kv_dim = q_dim

        self.ln_1q = LayerNorm(q_dim, bias=config.bias)
        self.ln_1kv = LayerNorm(kv_dim, bias=config.bias)
        self.attn = MultiHeadAttention(config, q_dim, kv_dim, n_embd)
        self.ln_2 = LayerNorm(q_dim, bias=config.bias)
        self.mlp = SwiGLU_MLP(config, q_dim)
        self.ln_3 = LayerNorm(q_dim, bias=config.bias)
        self.ln_4 = LayerNorm(q_dim, bias=config.bias)

    def forward(self, x_q, x_kv=None, attn_bias=None):
        if x_kv is None:
            x_kv = x_q
        x = x_q + self.ln_3(self.attn(self.ln_1q(x_q), self.ln_1kv(x_kv), attn_bias=attn_bias))
        x = x + self.ln_4(self.mlp(self.ln_2(x)))
        return x


class RepresentationEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.n_head = config.n_head
        self.empty_token_code = config.empty_token_code
        self.latent_embd = config.latent_embd
        self.latent_tok_n = config.latent_tok_n
        self.max_num_neighbors = config.max_num_neighbors
        self.agent_info_size = config.agent_info_size
        self.field_of_view_size = config.field_of_view_size
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            wle=nn.Embedding(config.latent_tok_n, config.latent_embd),
            wne=nn.Embedding(config.max_num_neighbors, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, config.n_embd) for _ in range(config.n_encoder_layer)]),
            latent_encoder=Block(config, config.latent_embd, config.n_embd, config.n_embd),
            ln_f=LayerNorm(config.latent_embd, bias=config.bias),
        ))
        self.register_buffer(
            "position_ids", torch.arange(config.block_size, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "latent_ids", torch.arange(config.latent_tok_n, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "neighbor_slot_ids",
            torch.arange(config.max_num_neighbors, dtype=torch.long)
            .repeat_interleave(config.agent_info_size),
            persistent=False,
        )
        self.register_buffer(
            "neighbor_token_embeddings", None, persistent=False
        )

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_encoder_layer))

    def prepare_for_inference(self):
        nbrs = self.transformer.wne(self.neighbor_slot_ids)
        tail_size = (
            self.block_size
            - self.field_of_view_size
            - self.max_num_neighbors * self.agent_info_size
        )
        self.neighbor_token_embeddings = torch.cat(
            (
                torch.zeros(
                    self.field_of_view_size,
                    nbrs.shape[-1],
                    device=nbrs.device,
                    dtype=nbrs.dtype,
                ),
                nbrs,
                torch.zeros(
                    tail_size,
                    nbrs.shape[-1],
                    device=nbrs.device,
                    dtype=nbrs.dtype,
                ),
            ),
            dim=0,
        ).detach()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = self.position_ids[:t]
        latent_idx = self.latent_ids

        empty_mask_idx = idx == self.empty_token_code

        if self.neighbor_token_embeddings is None:
            nbrs = self.transformer.wne(self.neighbor_slot_ids)
            tail_size = (self.block_size
                         - self.field_of_view_size
                         - self.max_num_neighbors * self.agent_info_size)
            nbrs_embd = torch.cat([
                torch.zeros(self.field_of_view_size, nbrs.shape[-1], device=idx.device, dtype=nbrs.dtype),
                nbrs,
                torch.zeros(tail_size, nbrs.shape[-1], device=idx.device, dtype=nbrs.dtype),
            ], dim=0)
        else:
            nbrs_embd = self.neighbor_token_embeddings[:t]

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe.weight[:t]
        latent_emb = self.transformer.wle.weight[: self.latent_tok_n].unsqueeze(0).expand(b, -1, -1)

        # SDPA broadcasts a key mask over heads and query positions.  The old
        # [B,H,T,T] allocation reached ~192 MiB at N=256 despite every query
        # sharing the same padding mask.
        attn_bias = torch.zeros(
            (b, 1, 1, t), device=idx.device, dtype=tok_emb.dtype
        ).masked_fill(empty_mask_idx[:, None, None, :], float('-inf'))

        x = self.transformer.drop(tok_emb + pos_emb + nbrs_embd)
        latent = self.transformer.drop(latent_emb)

        for block in self.transformer.h:
            x = block(x, attn_bias=attn_bias)

        latent = self.transformer.latent_encoder(latent, x, attn_bias=attn_bias)
        latent = self.transformer.ln_f(latent)
        return latent


class MessageCoordinator(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.empty_connection_code == -1
        self.config = config
        self.empty_msg_emb = nn.Embedding(1, config.latent_embd)
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def collect(agent_to_msg, connections):
        # connections: -1 → 0 (points to prepended empty row)
        connections = connections.long() + 1
        connections = connections.unsqueeze(-1).expand(
            -1, -1, -1, agent_to_msg.size(-1)
        )
        out = torch.gather(
            agent_to_msg.unsqueeze(2).expand(
                -1, -1, connections.size(2), -1
            ),
            dim=1,
            index=connections,
        )
        return out  # [batch, agents, L, n_emb]

    def forward(self, agent_to_msg, connections):
        device = agent_to_msg.device
        b, c, _ = agent_to_msg.shape
        empty_msg = self.dropout(
            self.empty_msg_emb.weight[0].reshape(1, 1, -1).expand(b, -1, -1)
        )
        msg = torch.cat([empty_msg, agent_to_msg], dim=1)
        return self.collect(msg, connections)


class RepresentationDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.n_head = config.n_head
        self.empty_connection_code = config.empty_connection_code

        self.transformer = nn.ModuleDict(dict(
            drop=nn.Dropout(config.dropout),
            act_msg_embd=nn.Embedding(1, config.action_msg_feats),
            h=nn.ModuleList([Block(config, config.latent_embd) for _ in range(config.n_decoder_layer)]),
            ln_f=LayerNorm(config.action_msg_feats, bias=config.bias),
            out_block=Block(config, config.action_msg_feats, config.latent_embd, config.n_embd),
            msg_head=nn.Linear(config.action_msg_feats, config.latent_embd, bias=False),
        ))
        self.register_buffer(
            "action_token_id", torch.zeros(1, dtype=torch.long), persistent=False
        )

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_decoder_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, latent, messages, connections):
        b, t, _ = latent.size()
        _, k = connections.size()

        empty_mask_idx = (connections == self.empty_connection_code)
        empty_mask_idx = torch.cat([
            torch.zeros(b, t, dtype=torch.bool, device=latent.device),
            empty_mask_idx,
        ], dim=1)
        attn_bias = torch.zeros(
            (b, 1, 1, t + k), device=latent.device, dtype=latent.dtype
        ).masked_fill(empty_mask_idx[:, None, None, :], float('-inf'))

        act_msg_latent = self.transformer.drop(
            self.transformer.act_msg_embd.weight[0]
            .reshape(1, 1, -1)
            .expand(b, -1, -1)
        )

        x = torch.cat([latent, messages], dim=1)
        for block in self.transformer.h:
            x = block(x, attn_bias=attn_bias)

        act_msg_latent = self.transformer.out_block(act_msg_latent, x, attn_bias=attn_bias)
        act_msg_latent = self.transformer.ln_f(act_msg_latent)
        return act_msg_latent


NUM_ACTIONS = 5


@dataclass
class DMMConfig:
    # ── Observation / vocabulary ──────────────────────────────────────────
    block_size: int = 256
    vocab_size: int = 67
    field_of_view_size: int = 11 * 11
    agent_info_size: int = 10
    max_num_neighbors: int = 13
    empty_token_code: int = 66
    empty_connection_code: int = -1

    # ── Architecture ──────────────────────────────────────────────────────
    n_encoder_layer: int = 2
    n_decoder_layer: int = 2
    n_head: int = 2
    n_embd: int = 16
    latent_embd: int = 8
    latent_tok_n: int = 8
    action_msg_feats: int = 16
    dropout: float = 0.0
    bias: bool = False

    # ── Dynamics ──────────────────────────────────────────────────────────
    dt: float = 0.25        # EMA step size
    tau: float = 1.0        # sampling temperature (inference)
    n_comm_rounds: int = 2  # K communication rounds

    # ── Teacher forcing (IL only; unused in MICPO) ────────────────────────
    dirichlet_tf_on: bool = True
    dirichlet_tf_beta: float = 1.0
    dirichlet_tf_beta_final: float = 0.0
    dirichlet_tf_anneal_steps: int = 50_000
    round_tf_on: bool = True
    round_tf_beta: float = 1.0
    round_tf_beta_final: float = 0.0
    round_tf_anneal_steps: int = 50_000


class DMM(nn.Module):
    """
    Consensus Voting Model with K categorical rounds (no pre-round, no priority tag).

    Parameters vs DMM_FM:
        z_proj  Linear(5 → latent_embd)  — project z into message space
        pi_head Linear(action_msg_feats → 5) — categorical policy head
    """

    def __init__(self, config: DMMConfig):
        super().__init__()
        self.config = config
        self.num_actions = NUM_ACTIONS

        self.representation_encoder = RepresentationEncoder(config)
        self.coordinator = MessageCoordinator(config)
        self.representation_decoder = RepresentationDecoder(config)
        self.msg_nbrs_embedding = nn.Embedding(config.max_num_neighbors, config.latent_embd)

        self.z_proj = nn.Linear(NUM_ACTIONS, config.latent_embd, bias=False)
        self.pi_head = nn.Linear(config.action_msg_feats, NUM_ACTIONS, bias=False)

        self._init_new_weights()

    @staticmethod
    def _centred_log_ohe(g: Tensor, num_actions: int, dtype: torch.dtype) -> Tensor:
        """Centred log-softmax of one_hot(g) — the target point in z-space."""
        ohe = F.one_hot(g, num_actions).to(dtype)
        e = F.log_softmax((ohe + 1e-8).log(), dim=-1)
        return e - e.mean(-1, keepdim=True)

    @staticmethod
    def _dirichlet_z0(BN: int, num_actions: int, device, dtype, alpha: float = 1.0) -> Tensor:
        """Sample z_0 from Dirichlet(alpha) in log-centered space.
        alpha=1 → uniform (max variance), alpha>1 → concentrated near center (less variance)."""
        d = torch.distributions.Dirichlet(
            torch.full((BN, num_actions), alpha)
        ).sample()  # fp32 on CPU — log computed before cast to avoid fp16 underflow
        z = d.clamp(min=1e-8).log()
        z = z - z.mean(-1, keepdim=True)
        return z.to(device=device, dtype=dtype)

    @staticmethod
    def _prob_to_log_centered(d: Tensor) -> Tensor:
        """Convert raw Dirichlet probability sample to log-centered z space."""
        z = d.float().clamp(min=1e-8).log()
        z = z - z.mean(-1, keepdim=True)
        return z.to(dtype=d.dtype)

    @staticmethod
    def _dirichlet_z0_teacher(
        BN: int,
        num_actions: int,
        device,
        dtype,
        targets_flat: Tensor | None = None,
        tf_beta: float = 0.0,
        dir_alpha: float = 1.0,
    ) -> Tensor:
        """
        Sample z_0 in log-centered space from Dirichlet distribution.

        Training (targets provided, tf_beta > 0):
            Each agent independently: with prob tf_beta, alpha = ones + one_hot(target)
                                      with prob 1-tf_beta, alpha = dir_alpha (uniform)
        Inference / no teacher:
            alpha = dir_alpha  (uniform Dirichlet; >1 concentrates near uniform)
        Sampling and transforms run on the model device in fp32 before casting.
        """
        alpha = torch.full((BN, num_actions), dir_alpha, device=device, dtype=torch.float32)
        if targets_flat is not None and tf_beta > 0.0:
            valid = (targets_flat != -1).to(device)
            tgt_safe = targets_flat.clamp(min=0).to(device)
            ohe = F.one_hot(tgt_safe, num_actions).float()
            mask = torch.bernoulli(torch.full((BN,), tf_beta, device=device, dtype=torch.float32)).bool() & valid
            alpha[mask] = alpha[mask] + ohe[mask]
        d = torch.distributions.Dirichlet(alpha).sample()  # fp32 on the model device
        z = d.clamp(min=1e-8).log()
        z = z - z.mean(-1, keepdim=True)
        return z.to(device=device, dtype=dtype)

    def _init_new_weights(self):
        for m in [self.z_proj, self.pi_head]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.msg_nbrs_embedding.weight, mean=0.0, std=0.02)

    # ── Shared single-round logic ────────────────────────────────────────

    @staticmethod
    def _derangement(n: int, device) -> Tensor:
        """Permutation of range(n) with no fixed points, so no agent keeps its own message."""
        if n <= 1:
            return torch.arange(n, device=device)
        while True:
            perm = torch.randperm(n, device=device)
            if not bool((perm == torch.arange(n, device=device)).any()):
                return perm

    def _run_round(
        self,
        z: Tensor,
        h: Tensor,
        latent: Tensor,
        agent_chat_ids: Tensor,
        nbrs: Tensor,
        z_broadcast: Tensor | None = None,
        broadcast_mode: str = "full",
        shuffle_messages: bool = False,
    ):
        """
        One full round: form message → coordinator → position embeddings →
        decoder → pi_head (logits) + msg_head (new h).

        z:             [B*N, 5]
        h:             [B*N, latent_embd]
        latent:        [B*N, tok_n, latent_embd]
        agent_chat_ids:[B, N, L]
        nbrs:          [max_nbrs, latent_embd]

        The trailing arguments are inference-time broadcast ablations and leave the
        weights and the agent's own z update untouched: z_broadcast replaces the
        transmitted z, broadcast_mode drops h ("no_h") or z ("no_z") from the outgoing
        message, and shuffle_messages permutes assembled messages across agents. The
        defaults reproduce the unablated round.

        Returns: logits [B*N, 5], h_new [B*N, latent_embd]
        """
        BN = z.shape[0]
        B, N, L = agent_chat_ids.shape
        connections = agent_chat_ids.view(BN, L)

        z_for_msg = z if z_broadcast is None else z_broadcast
        if broadcast_mode == "full":
            agent_to_msg = h + self.z_proj(z_for_msg).to(h.dtype)    # [BN, latent_embd]
        elif broadcast_mode == "no_h":
            agent_to_msg = self.z_proj(z_for_msg).to(h.dtype)
        elif broadcast_mode == "no_z":
            agent_to_msg = h
        else:
            raise ValueError(f"unsupported broadcast mode: {broadcast_mode}")

        messages = self.coordinator(
            agent_to_msg.view(B, N, -1), agent_chat_ids
        ).view(B, N, L, -1)                              # [B, N, L, latent_embd]

        if shuffle_messages:
            messages = torch.stack([
                messages[b, self._derangement(N, messages.device)] for b in range(B)
            ])

        messages = messages.view(BN, L, -1) + nbrs[:L]

        feat = self.representation_decoder(latent, messages, connections)

        logits = self.pi_head(feat[:, 0, :])              # [BN, 5]
        h_new = self.representation_decoder.transformer.msg_head(feat)[:, 0, :]

        return logits, h_new

    # ── Training forward ─────────────────────────────────────────────────

    def forward(
        self,
        observations: Tensor,
        agent_chat_ids: Tensor,
        target_actions: Tensor,
        dirichlet_tf_beta: float | None = None,
        round_tf_beta: float | None = None,
    ):
        """
        observations:       [B, N, T]
        agent_chat_ids:     [B, N, L]
        target_actions:     [B, N]    (−1 = ignore index)
        dirichlet_tf_beta:  overrides config; prob of biased Dirichlet(1+ohe) for z_0
        round_tf_beta:      overrides config; prob of using target action for z EMA update

        Returns:
            loss             — mean CE over K comm rounds
            per_round_losses — list of K per-round CE tensors
        """
        B, N, T = observations.shape
        device = observations.device
        BN = B * N

        obs_flat = observations.view(BN, T)
        targets_flat = target_actions.view(BN)

        latent = self.representation_encoder(obs_flat)    # [BN, tok_n, latent_embd]
        dtype = latent.dtype

        nbrs = self.msg_nbrs_embedding(
            torch.arange(self.config.max_num_neighbors, device=device, dtype=torch.long)
        ).to(dtype)

        _dir_tf = (self.config.dirichlet_tf_beta if dirichlet_tf_beta is None else dirichlet_tf_beta)
        if not self.config.dirichlet_tf_on:
            _dir_tf = 0.0
        _rnd_tf = (self.config.round_tf_beta if round_tf_beta is None else round_tf_beta)
        if not self.config.round_tf_on:
            _rnd_tf = 0.0

        z = self._dirichlet_z0_teacher(BN, self.num_actions, device, dtype, targets_flat, _dir_tf)

        h = self.coordinator.empty_msg_emb(
            torch.zeros(BN, dtype=torch.long, device=device)
        ).to(dtype)

        per_round_losses = []

        for _ in range(self.config.n_comm_rounds):
            logits_t, h = self._run_round(z, h, latent, agent_chat_ids, nbrs)
            per_round_losses.append(
                F.cross_entropy(logits_t, targets_flat, ignore_index=-1)
            )

            with torch.no_grad():
                if _rnd_tf > 0.0:
                    use_tf = torch.bernoulli(
                        torch.full((BN,), _rnd_tf, device=device)
                    ).bool() & (targets_flat != -1)
                    g_sample = torch.distributions.Categorical(logits=logits_t).sample()
                    g_t = torch.where(use_tf, targets_flat.clamp(min=0), g_sample)
                else:
                    g_t = torch.distributions.Categorical(logits=logits_t).sample()

            e_g = self._centred_log_ohe(g_t, self.num_actions, dtype)
            z = z + self.config.dt * (e_g - z)

        loss = torch.stack(per_round_losses).mean()
        return loss, per_round_losses

    # ── Inference (IL) ───────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs: Tensor, agent_chat_ids: Tensor, do_sample: bool = True) -> Tensor:
        """Returns action indices [B*N]."""
        B, N, T = obs.shape
        device = obs.device
        BN = B * N

        latent = self.representation_encoder(obs.view(BN, T))
        dtype = latent.dtype

        nbrs = self.msg_nbrs_embedding(
            torch.arange(self.config.max_num_neighbors, device=device, dtype=torch.long)
        ).to(dtype)

        z = self._dirichlet_z0(BN, self.num_actions, device, dtype)
        h = self.coordinator.empty_msg_emb(
            torch.zeros(BN, dtype=torch.long, device=device)
        ).to(dtype)

        for _ in range(self.config.n_comm_rounds):
            logits_t, h = self._run_round(z, h, latent, agent_chat_ids, nbrs)
            if do_sample:
                g_t = torch.distributions.Categorical(
                    logits=logits_t / self.config.tau
                ).sample()
            else:
                g_t = logits_t.argmax(-1)
            e_g = self._centred_log_ohe(g_t, self.num_actions, dtype)
            z = z + self.config.dt * (e_g - z)

        return z.argmax(-1)

    # ── MICPO rollout ─────────────────────────────────────────────────────

    @torch.no_grad()
    def rollout_act(
        self,
        obs: Tensor,             # [B, N, T]
        agent_chat_ids: Tensor,  # [B, N, L]
        rollout_tau: float = 1.0,
        z0_in: Tensor | None = None,  # [B*N, 5] pre-sampled z0; if None, sample fresh
        dir_alpha: float = 1.0,  # Dirichlet concentration when sampling z0
    ):
        """
        Stochastic forward for MICPO rollout collection.

        z_0 is sampled from uniform Dirichlet(1) unless z0_in is provided
        (used to share z0 across group members for variance reduction).

        Returns:
            actions        [B*N]       — final action = argmax(z_K)
            votes          [B*N, K]    — g_0, …, g_{K-1}   int64
            log_pi_rounds  [B*N, K]    — log π_t(g_t) per round   float32
            z0             [B*N, 5]    — initial state used (store for replay)
            z              [B*N, 5]    — final EMA state
        """
        B, N, T = obs.shape
        device = obs.device
        BN = B * N

        latent = self.representation_encoder(obs.view(BN, T))
        dtype = latent.dtype

        nbrs = self.msg_nbrs_embedding(
            torch.arange(self.config.max_num_neighbors, device=device, dtype=torch.long)
        ).to(dtype)

        if z0_in is not None:
            # z0_in is raw Dirichlet probability-space; convert to log-centered
            z0 = self._prob_to_log_centered(z0_in.to(device=device, dtype=dtype))
        else:
            z0 = self._dirichlet_z0(BN, self.num_actions, device, dtype, alpha=dir_alpha)
        z = z0.clone()
        h = self.coordinator.empty_msg_emb(
            torch.zeros(BN, dtype=torch.long, device=device)
        ).to(dtype)

        votes_list: list[Tensor] = []
        log_pi_list: list[Tensor] = []

        for _ in range(self.config.n_comm_rounds):
            logits_t, h = self._run_round(z, h, latent, agent_chat_ids, nbrs)
            logits_t = logits_t.nan_to_num(0.0)
            g_t = torch.distributions.Categorical(logits=logits_t / rollout_tau).sample()
            log_pi_t = F.log_softmax(logits_t, dim=-1).gather(-1, g_t.unsqueeze(-1)).squeeze(-1)
            votes_list.append(g_t)
            log_pi_list.append(log_pi_t)
            e_g = self._centred_log_ohe(g_t, self.num_actions, dtype)
            z = z + self.config.dt * (e_g - z)

        votes = torch.stack(votes_list, dim=-1)           # [BN, K]
        log_pi_rounds = torch.stack(log_pi_list, dim=-1)  # [BN, K]
        actions = z.argmax(-1)                            # [BN]

        return actions, votes, log_pi_rounds, z0, z

    # ── MICPO training replay ─────────────────────────────────────────────

    def forward_micpo(
        self,
        obs: Tensor,            # [B, N, T]
        agent_chat_ids: Tensor, # [B, N, L]
        stored_votes: Tensor,   # [B*N, K]   int64
        stored_z0: Tensor,      # [B*N, 5]   float — z_0 from rollout
    ):
        """
        Deterministic replay given stored vote sequences and initial state z_0.

        Gradients flow through all round logits and through h across rounds.
        z is treated as a constant (computed from stored discrete votes).

        Returns:
            log_pi_rounds    [B*N, K]     — log π_θ_t(g_t) per round (grad-enabled)
            per_round_logits [B*N, K, 5]  — full distributions (for KL, entropy)
        """
        B, N, T = obs.shape
        device = obs.device
        BN = B * N

        latent = self.representation_encoder(obs.view(BN, T))
        dtype = latent.dtype

        nbrs = self.msg_nbrs_embedding(
            torch.arange(self.config.max_num_neighbors, device=device, dtype=torch.long)
        ).to(dtype)

        z = stored_z0.to(dtype)
        h = self.coordinator.empty_msg_emb(
            torch.zeros(BN, dtype=torch.long, device=device)
        ).to(dtype)

        log_pi_list: list[Tensor] = []
        logits_list: list[Tensor] = []

        for k in range(self.config.n_comm_rounds):
            logits_t, h = self._run_round(z, h, latent, agent_chat_ids, nbrs)
            g_t = stored_votes[:, k]
            log_pi_t = F.log_softmax(logits_t, dim=-1).gather(-1, g_t.unsqueeze(-1)).squeeze(-1)
            log_pi_list.append(log_pi_t)
            logits_list.append(logits_t)

            e_g = self._centred_log_ohe(g_t.detach(), self.num_actions, dtype)
            z = z + self.config.dt * (e_g - z)

        log_pi_rounds = torch.stack(log_pi_list, dim=-1)            # [BN, K]
        per_round_logits = torch.stack(logits_list, dim=1)          # [BN, K, 5]

        return log_pi_rounds, per_round_logits

    # ── Utilities ────────────────────────────────────────────────────────

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra)
        return optimizer
