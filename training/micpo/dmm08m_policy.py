"""
DMM08MPolicy — policy adapter for the DMM08M MICPO pipeline.

Three usage modes:

1. Rollout collection:
       actions, votes, log_pi_rounds, z0, z = policy.rollout_act(obs, chat_ids)
   Returns per-round votes/log-pis and z0 for deterministic replay.

2. Training epoch forward:
       log_pi_rounds, per_round_logits = policy.forward_micpo(obs, chat_ids, votes, z0)
   Deterministic replay with stored votes and z0; gradients flow through logits and h.

3. Validation / animation (policy(obs, chat_ids).logits):
       policy_out = policy(obs, chat_ids)   → PolicyOutput(logits=z.view(B,N,5))
   z is the final EMA state — argmax(z) gives the greedy action.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class PolicyOutput:
    logits: Tensor   # [B, N, 5]  — z (EMA state) used as logits


class DMM08MPolicy(nn.Module):
    """
    Thin adapter over model.dmm_08m.DMM08M.

    Exposes the interface expected by the training loop and validation code.
    """

    N_ACTIONS = 5

    def __init__(self, net, dir_alpha: float = 1.0):
        """
        Args:
            net: an already-constructed DMM08M instance (with weights loaded).
            dir_alpha: Dirichlet concentration for z0 sampling (1=uniform, >1=tighter).
        """
        super().__init__()
        self.net = net
        self.dir_alpha = dir_alpha

    def forward(self, obs: Tensor, agent_chat_ids: Tensor) -> PolicyOutput:
        """
        For validation, animation, and reference-baseline calls.
        Runs all K rounds stochastically (tau-scaled) and returns z as logits.

        obs:            [B, N, context_size]   long
        agent_chat_ids: [B, N, max_neighbors]  long
        Returns PolicyOutput(logits=[B, N, 5]).
        """
        B, N, _ = obs.shape
        _, _, _, _, z = self.net.rollout_act(obs, agent_chat_ids, dir_alpha=self.dir_alpha)
        logits = z.view(B, N, self.N_ACTIONS).float()
        return PolicyOutput(logits=logits)

    def rollout_act(self, obs: Tensor, agent_chat_ids: Tensor, rollout_tau: float = 1.0,
                    z0_in=None):
        """Delegate to net.rollout_act. Returns (actions, votes, log_pi_rounds, z0, z)."""
        return self.net.rollout_act(obs, agent_chat_ids, rollout_tau=rollout_tau,
                                    z0_in=z0_in, dir_alpha=self.dir_alpha)

    @torch.no_grad()
    def deterministic_zero_act(
        self, obs: Tensor, agent_chat_ids: Tensor
    ) -> PolicyOutput:
        """Deployment path: z0=0 and argmax vote on every communication round."""
        batch, agents, _ = obs.shape
        count = batch * agents
        net = self.net
        static_key, static_value, padding, _ = net._encode(obs)
        dtype = static_key.dtype
        z = torch.zeros(count, self.N_ACTIONS, device=obs.device, dtype=dtype)
        h = net.communication.empty_message.reshape(1, 1, -1).expand(
            batch, agents, -1
        ).to(dtype)
        for _ in range(net.config.n_comm_rounds):
            logits, h = net._run_round(
                z, h, agent_chat_ids, static_key, static_value, padding
            )
            vote = logits.nan_to_num(0.0).argmax(dim=-1)
            target = net._centred_log_ohe(vote, dtype)
            z = z + net.config.dt * (target - z)
        return PolicyOutput(logits=z.view(batch, agents, self.N_ACTIONS).float())

    def forward_micpo(
        self,
        obs: Tensor,
        agent_chat_ids: Tensor,
        stored_votes: Tensor,
        stored_z0: Tensor,
    ):
        """
        Deterministic replay for MICPO training epoch.
        Returns (log_pi_rounds [BN, K], per_round_logits [BN, K, 5]).
        """
        return self.net.forward_micpo(obs, agent_chat_ids, stored_votes, stored_z0)

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: torch.device,
        n_comm_rounds: int = 4,
        use_fp16: bool = False,
        use_compile: bool = False,
    ) -> "DMM08MPolicy":
        import logging
        from model.dmm_08m import DMM08MConfig, DMM08M

        log = logging.getLogger(__name__)

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        raw_sd = checkpoint["model"]

        # Strip _orig_mod. prefix (saved from a torch.compile'd model)
        state_dict = {}
        for key, value in raw_sd.items():
            # Supervised checkpoints store DMM08M directly, whereas MICPO
            # checkpoints store the DMM08MPolicy wrapper ("net.").  Accept both
            # forms so a second MICPO stage really continues the learned model.
            while key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]
            if key.startswith("net."):
                key = key[len("net."):]
            state_dict[key] = value

        config_dict = dict(checkpoint["model_args"])
        config_dict["n_comm_rounds"] = n_comm_rounds

        known = {f.name for f in DMM08MConfig.__dataclass_fields__.values()}
        config = DMM08MConfig(**{k: v for k, v in config_dict.items() if k in known})
        net = DMM08M(config)

        # Always load in fp32; use autocast during training for mixed precision.
        # Never call .half() here — that causes fp16 NaN in rollout/validation.
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "DMM-08M checkpoint does not match the model architecture: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        log.info("from_checkpoint: all %d keys matched", len(state_dict))

        net.to(device)

        if use_compile and device.type == "cuda":
            net = torch.compile(net, mode="reduce-overhead")

        return cls(net)

    @classmethod
    def from_scratch(cls, model_args: dict, device: torch.device) -> "DMM08MPolicy":
        from model.dmm_08m import DMM08MConfig, DMM08M
        known = {f.name for f in DMM08MConfig.__dataclass_fields__.values()}
        config = DMM08MConfig(**{k: v for k, v in model_args.items() if k in known})
        net = DMM08M(config)
        net.to(device)
        return cls(net)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        return self.net.configure_optimizers(weight_decay, lr, betas, device_type)

    @property
    def representation_encoder(self):
        return self.net.representation_encoder


def build_policy(config, device: torch.device) -> DMM08MPolicy:
    if str(config.policy_class).lower() != "dmm08m":
        raise ValueError("DMM-08M MICPO supports only policy_class='dmm08m'")
    if config.init_from == "scratch":
        return DMM08MPolicy.from_scratch(_model_args_from_config(config), device)
    if config.path_to_weights is None:
        raise ValueError("path_to_weights must be set when init_from != 'scratch'")
    return DMM08MPolicy.from_checkpoint(
        path=config.path_to_weights,
        device=device,
        n_comm_rounds=config.n_comm_rounds,
        use_fp16=config.use_fp16,
        use_compile=False,
    )


def _model_args_from_config(config) -> dict:
    return dict(
        block_size=config.context_size,
        vocab_size=67,
        field_of_view_size=121,
        agent_info_size=10,
        max_num_neighbors=config.max_num_neighbors,
        n_encoder_layer=config.n_encoder_layer,
        n_decoder_layer=config.n_decoder_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        latent_embd=config.latent_embd,
        latent_tok_n=config.latent_tok_n,
        dropout=config.dropout,
        bias=config.bias,
        empty_token_code=66,
        action_msg_feats=config.action_msg_feats,
        empty_connection_code=-1,
        n_comm_rounds=config.n_comm_rounds,
        dt=config.dt,
        tau=config.tau,
    )
