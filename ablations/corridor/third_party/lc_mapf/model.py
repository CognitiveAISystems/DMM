import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from loguru import logger
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, input):
        return F.rms_norm(input, self.weight.shape, self.weight, 1e-5)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 config,
                 q_dim: int,
                 kv_dim: int = None,
                 n_embd: int = None,
                 ):
        super().__init__()
        self.n_head = config.n_head

        if n_embd is None:
            self.n_embd = config.n_embd
        else:
            self.n_embd = n_embd
        self.dropout = config.dropout

        if kv_dim is None:
            kv_dim = q_dim

        assert self.n_embd % self.n_head == 0, f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        self.head_size = config.n_embd // self.n_head

        self.q_proj = nn.Linear(q_dim, 2*self.n_embd, bias=config.bias)
        self.kv_proj = nn.Linear(kv_dim, 3*self.n_embd, bias=config.bias)

        self.c_proj = nn.Linear(self.n_embd, q_dim, bias=config.bias)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.attn_dropout = nn.Dropout(self.dropout)

        # per-head normalization
        self.q_norm_1 = RMSNorm(self.head_size)
        self.k_norm_1 = RMSNorm(self.head_size)
        self.q_norm_2 = RMSNorm(self.head_size)
        self.k_norm_2 = RMSNorm(self.head_size)
        self.gr_norm = nn.GroupNorm(self.n_head, self.n_embd)

        self.lmb = nn.Parameter(-4*torch.ones(1, self.n_head, 1, 1))

        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self,
                x_q,       # [batch, num_q_tokens, q_dim]
                x_kv=None, # [batch, num_kv_tokens, kv_dim]
                attn_bias=None, # [batch, n_head, num_q_tokens, num_kv_tokens]
               ):
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

        # differential attention
        if self.flash:
            y1 = F.scaled_dot_product_attention(
                q1, k1, v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False
            )
            y2 = F.scaled_dot_product_attention(
                q2, k2, v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False
            )
            y = y1 - F.softplus(self.lmb)*y2
        else:
            att1 = (q1 @ k1.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
            att2 = (q2 @ k2.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
            if attn_bias is not None:
                att1 = att1 + attn_bias
                att2 = att2 + attn_bias
            att1 = F.softmax(att1, dim=-1)
            att2 = F.softmax(att2, dim=-1)
            att1 = self.attn_dropout(att1)
            att2 = self.attn_dropout(att2)
            y = att1 @ v - F.softplus(self.lmb) * (att2 @ v)

        y = y.transpose(1, 2).contiguous().view(B, T_q, self.n_embd)
        y = self.gr_norm(y.transpose(1, 2)).transpose(1, 2)
        y = self.resid_dropout(self.c_proj(y))
        return y # [batch, num_q_tokens, n_embd]


class SwiGLUMLP(nn.Module):
    def __init__(self, config, n_embd=None):
        super().__init__()
        if n_embd is None:
            self.n_embd = config.n_embd
        else:
            self.n_embd = n_embd
        self.dropout = config.dropout

        hidden_dim = 4 * self.n_embd
        self.c_fc = nn.Linear(self.n_embd, hidden_dim * 2, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, self.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        x_fc = self.c_fc(x)
        x_a, x_b = x_fc.chunk(2, dim=-1)
        x = x_a * F.silu(x_b)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self,
                config,
                q_dim: int,
                kv_dim: int = None,
                n_embd: int = None,
                ):
        super().__init__()
        if n_embd is None:
            self.n_embd = config.n_embd
        else:
            self.n_embd = n_embd

        if kv_dim is None:
            kv_dim = q_dim

        self.ln_1q = RMSNorm(q_dim)
        self.ln_1kv = RMSNorm(kv_dim)
        self.attn = MultiHeadAttention(
            config, q_dim, kv_dim, n_embd
        )
        self.ln_2 = RMSNorm(q_dim)
        self.mlp = SwiGLUMLP(
            config, q_dim
        )
        self.ln_3 = RMSNorm(q_dim)
        self.ln_4 = RMSNorm(q_dim)

    def forward(self, x_q, x_kv=None, attn_bias=None):
        if x_kv is None:
            x_kv = x_q
        x = x_q + self.ln_3(self.attn(self.ln_1q(x_q),
                                    self.ln_1kv(x_kv),
                                    attn_bias=attn_bias))
        x = x + self.ln_4(self.mlp(self.ln_2(x)))
        return x


@dataclass
class Config:
    block_size: int = 256          # max sequence length
    vocab_size: int = 67
    field_of_view_size: int = 11*11  # flattened local obstacle grid
    agent_info_size: int = 10      # token features per neighbor
    max_num_neighbors: int = 13    # including self; must match dataset

    n_encoder_layer: int = 2
    n_decoder_layer: int = 2
    n_head: int = 2
    n_embd: int = 16
    latent_embd: int = 8           # latent token dimension
    latent_tok_n: int = 8          # latent tokens per agent
    dropout: float = 0.0
    bias: bool = False
    empty_token_code: int = 66     # token ID for masked/absent positions
    action_msg_feats: int = 16     # action-query feature size (shared by action and message heads)
    empty_connection_code: int = -1  # sentinel for absent neighbor slots
    n_comm_rounds: int = 2         # message-passing rounds


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
            wne=nn.Embedding(config.max_num_neighbors, config.n_embd), # neighbor embeddings
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList(
                [Block(config, config.n_embd) for _ in range(config.n_encoder_layer)]
            ),
            latent_encoder=Block(config,
                                 config.latent_embd,
                                 config.n_embd,
                                 config.n_embd
                                 ),
            ln_f=RMSNorm(config.latent_embd),
        ))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_encoder_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        latent = torch.arange(0, self.latent_tok_n, dtype=torch.long, device=device)

        # mask empty tokens
        empty_mask_idx = (idx == self.empty_token_code)
        attn_bias = torch.zeros(b, self.n_head, t, t, device=device)
        attn_bias = attn_bias.masked_fill(empty_mask_idx[:, None, None, :], float('-inf'))

        # neighbor embeddings (includes self)
        nbr_idx = torch.arange(self.max_num_neighbors, device=device, dtype=torch.long)
        nbrs = self.transformer.wne(nbr_idx.repeat_interleave(self.agent_info_size))
        tail_size = self.block_size - self.field_of_view_size - self.max_num_neighbors*self.agent_info_size
        nbrs_embd = torch.cat(
            [
                torch.zeros(self.field_of_view_size, nbrs.shape[-1], device=device, dtype=nbrs.dtype),
                nbrs,
                torch.zeros(tail_size, nbrs.shape[-1], device=device, dtype=nbrs.dtype)
            ],
            dim=0
        )

        tok_emb = self.transformer.wte(idx)  # [b, t, n_embd]
        pos_emb = self.transformer.wpe(pos)  # [t, n_embd]
        latent_emb = self.transformer.wle(latent) # [latent_tok_n, latent_tok_embd]
        latent_emb = latent_emb.unsqueeze(0).repeat(b, 1, 1)

        x = self.transformer.drop(tok_emb + pos_emb + nbrs_embd)
        latent = self.transformer.drop(latent_emb)

        for block in self.transformer.h:
            x = block(x, attn_bias=attn_bias)

        # latent cross-attention mask
        empty_mask_latent = empty_mask_idx[:, None, :].repeat(1, self.latent_tok_n, 1) # [b, t] -> [b, latent_tok_n, t]
        attn_bias = torch.zeros_like(empty_mask_latent, dtype=torch.float, device=device)
        attn_bias = attn_bias.masked_fill(empty_mask_latent, float('-inf'))
        attn_bias = attn_bias[:, None, :, :].repeat(1, self.n_head, 1, 1)

        latent = self.transformer.latent_encoder(latent, x, attn_bias=attn_bias)
        latent = self.transformer.ln_f(latent)
        return latent


class MessageCoordinator(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        assert config.empty_connection_code == -1
        self.empty_msg_emb = nn.Embedding(1, config.latent_embd)
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def collect(
        agent_msgs,  # [batch, total_agents, n_emb]
        connections, # [batch, total_agents, local_agent_num]
        ):
        # shift by 1 to account for the prepended empty-message slot (index -1 → 0)
        connections = connections.long() + 1
        connections = connections.unsqueeze(-1).expand(-1, -1, -1, agent_msgs.size(-1))  # [batch, block, local_agent_num, n_emb]
        # gather along dim=1 (number of agents)
        out = torch.gather(
            agent_msgs.unsqueeze(2).expand(-1, -1, connections.size(2), -1),
            dim=1,
            index=connections,
        )
        return out # [batch, total_agents, local_agent_num, n_emb]

    def forward(self, agent_msgs, connections):
        device = agent_msgs.device
        b, c, _ = agent_msgs.shape
        empty_msg = torch.zeros(b, 1, dtype=torch.long, device=device)
        empty_msg = self.empty_msg_emb(empty_msg)
        empty_msg = self.dropout(empty_msg)

        # slot 0: empty message for absent neighbors (-1 connections)
        msg = torch.cat([empty_msg, agent_msgs], dim=1)
        msg = self.collect(msg, connections)
        return msg


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
            ln_f=RMSNorm(config.action_msg_feats),
            out_block=Block(config,
                            config.action_msg_feats,
                            config.latent_embd,
                            config.n_embd
                            ),
            msg_head=nn.Linear(config.action_msg_feats, config.latent_embd, bias=False)
        ))

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
        device = latent.device
        b, t, h = latent.size()
        _, k = connections.size()

        # mask empty tokens
        empty_mask_idx = (connections == self.empty_connection_code)
        empty_mask_idx = torch.cat(
            [torch.zeros(b, t, dtype=torch.bool, device=device),
             empty_mask_idx
             ],
             dim=1
        )
        attn_bias = torch.zeros(b, self.n_head, t+k, t+k, device=device)
        attn_bias = attn_bias.masked_fill(empty_mask_idx[:, None, None, :], float('-inf'))

        action_query = torch.arange(0, 1, dtype=torch.long, device=device)
        action_query = self.transformer.act_msg_embd(action_query)[None, :, :].repeat(b, 1, 1)
        action_query = self.transformer.drop(action_query)

        x = torch.cat([latent, messages], dim=1)

        for block in self.transformer.h:
            x = block(x, attn_bias=attn_bias)

        # output cross-attention mask
        empty_mask_latent = empty_mask_idx[:, None, :]
        attn_bias = torch.zeros_like(empty_mask_latent, dtype=torch.float, device=device)
        attn_bias = attn_bias.masked_fill(empty_mask_latent, float('-inf'))
        attn_bias = attn_bias[:, None, :, :].repeat(1, self.n_head, 1, 1)

        action_query = self.transformer.out_block(action_query, x, attn_bias=attn_bias)
        action_query = self.transformer.ln_f(action_query)

        return action_query


class LCMAPF(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.n_comm_rounds = config.n_comm_rounds

        self.max_num_neighbors = config.max_num_neighbors
        self.agent_info_size = config.agent_info_size
        self.field_of_view_size = config.field_of_view_size
        self.block_size = config.block_size

        self.representation_encoder = RepresentationEncoder(self.config)
        self.representation_decoder = RepresentationDecoder(self.config)
        self.coordinator = MessageCoordinator(self.config)

        self.msg_nbrs_embedding = nn.Embedding(self.max_num_neighbors, config.latent_embd)
        self.action_head = nn.Linear(config.action_msg_feats, 5, bias=False)

    def forward(self, observations, agent_chat_ids, target_actions=None):
        B, C, T = observations.shape
        device = observations.device
        _, _, L = agent_chat_ids.shape
        observations = observations.view(B * C, T)
        if target_actions is not None:
            target_actions = target_actions.view(-1)
        connections = agent_chat_ids.view(B * C, L)

        latent = self.representation_encoder(observations)
        agent_msgs = torch.zeros(B*C, dtype=torch.long, device=device)
        agent_msgs = self.coordinator.empty_msg_emb(agent_msgs)
        agent_msgs = self.coordinator.dropout(agent_msgs)

        # neighbor position embeddings for messages
        nbr_idx = torch.arange(self.max_num_neighbors, device=device, dtype=torch.long)
        nbrs = self.msg_nbrs_embedding(nbr_idx)

        losses = []

        for _ in range(self.n_comm_rounds):
            agent_msgs = agent_msgs.view(B, C, -1)
            messages = self.coordinator(agent_msgs, agent_chat_ids)
            messages = messages.view(B * C, L, -1) + nbrs[:L]
            action_query = self.representation_decoder(
                latent, messages, connections
            )
            action_logits = self.action_head(action_query.squeeze(1))
            agent_msgs = self.representation_decoder.transformer.msg_head(action_query)
            if target_actions is not None:
                loss = F.cross_entropy(action_logits, target_actions, ignore_index=-1)
                losses.append(loss)

        if target_actions is not None:
            loss = torch.stack(losses).mean()
        else:
            loss = None

        return action_logits, loss

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # 2D+ params (matmuls, embeddings) are weight-decayed; biases and norms are not
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.debug(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logger.debug(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        logger.debug(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def act(self, obs, agent_chat_ids, do_sample=True):
        logits, _ = self(obs, agent_chat_ids)
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        return idx_next.squeeze()

    def encode(self, observations):
        B, C, T = observations.shape
        device = observations.device
        observations = observations.view(B * C, T)

        latent = self.representation_encoder(observations)
        _, T_l, N_l = latent.shape
        return latent.view(B, C, T_l, N_l)
