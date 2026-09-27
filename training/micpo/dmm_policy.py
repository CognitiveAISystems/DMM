"""
DMMPolicy — policy adapter for DMM in the MICPO pipeline.

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


def _normalize_model_state_dict(state_dict: dict) -> dict:
    """Strip nested DDP, torch.compile, and policy-wrapper prefixes."""
    prefixes = ("module.", "_orig_mod.", "net.")
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        normalized[new_key] = value
    return normalized


@dataclass
class PolicyOutput:
    logits: Tensor   # [B, N, 5]  — z (EMA state) used as logits


class DMMPolicy(nn.Module):
    """
    Thin adapter over model.dmm.DMM.

    Exposes the interface expected by the training loop and validation code.
    """

    N_ACTIONS = 5

    def __init__(self, net, dir_alpha: float = 1.0):
        """
        Args:
            net: an already-constructed DMM instance (with weights loaded).
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
    ) -> "DMMPolicy":
        import logging
        from model.dmm import DMMConfig, DMM

        log = logging.getLogger(__name__)

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        raw_sd = checkpoint["model"]

        state_dict = _normalize_model_state_dict(raw_sd)

        config_dict = dict(checkpoint["model_args"])
        config_dict["n_comm_rounds"] = n_comm_rounds

        known = {f.name for f in DMMConfig.__dataclass_fields__.values()}
        config = DMMConfig(**{k: v for k, v in config_dict.items() if k in known})
        net = DMM(config)

        # Always load in fp32; use autocast during training for mixed precision.
        # Never call .half() here — that causes fp16 NaN in rollout/validation.
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint did not match DMM: "
                f"{len(missing)} missing and {len(unexpected)} unexpected keys"
            )
        log.info(f"from_checkpoint: all {len(state_dict)} keys matched")

        net.to(device)

        if use_compile and device.type == "cuda":
            net = torch.compile(net, mode="reduce-overhead")

        return cls(net)

    @classmethod
    def from_scratch(cls, model_args: dict, device: torch.device) -> "DMMPolicy":
        architecture = model_args.get("_architecture", "dmm")
        if architecture != "dmm":
            raise ValueError(f"Unknown scratch architecture {architecture!r}")
        from model.dmm import DMMConfig, DMM
        model_args = dict(model_args)
        model_args.pop("_architecture", None)
        config = DMMConfig(**model_args)
        net = DMM(config)
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


def build_policy(config, device: torch.device) -> DMMPolicy:
    if config.policy_class != "dmm":
        raise ValueError(f"3M MICPO requires policy_class='dmm', got {config.policy_class!r}")
    if config.init_from == "scratch":
        model_args = _model_args_from_config(config)
        return DMMPolicy.from_scratch(model_args, device)
    else:
        path = config.path_to_weights
        if path is None:
            raise ValueError("path_to_weights must be set when init_from != 'scratch'")
        return DMMPolicy.from_checkpoint(
            path=path,
            device=device,
            n_comm_rounds=config.n_comm_rounds,
            use_fp16=config.use_fp16,
            use_compile=False,
        )


def _model_args_from_config(config) -> dict:
    if config.policy_class != "dmm":
        raise ValueError(f"Unknown policy_class {config.policy_class!r}")
    return dict(
        _architecture="dmm",
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
