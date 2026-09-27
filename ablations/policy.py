"""Eager DMM policy with inference-time refinement-depth and broadcast ablations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from evaluation.compile_aoti import NUM_ACTIONS, WholePolicy
from pogema_gpu.adapters.aoti import CUDAAOTIPolicy
from pogema_gpu.adapters.cuda_runtime import CUDAPolicyRuntime
from pogema_gpu.adapters.runtime import configure_native_runtime

MODES = ("full", "no-h", "no-z", "z0-only", "shuffled")
BROADCAST = {"no-h": "no_h", "no-z": "no_z"}


class AblationWholePolicy(WholePolicy):
    """The compiled POGEMA policy, executed eagerly with one broadcast intervention."""

    def __init__(self, model, precision: str, mode: str):
        if mode not in MODES:
            raise ValueError(f"unsupported communication ablation: {mode}")
        super().__init__(model, "canonical-dmm", precision, "sample")
        self.mode = mode

    def forward(self, obs, chat, exponential, gumbels):
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.precision == "bf16"):
            batch, agents, tokens = obs.shape
            count = batch * agents
            rounds = self.model.config.n_comm_rounds
            latent = self.model.representation_encoder(obs.reshape(count, tokens))
            dtype = latent.dtype
            neighbors = self.model.msg_nbrs_embedding(
                torch.arange(self.model.config.max_num_neighbors,
                             device=obs.device, dtype=torch.long)
            ).to(dtype)
            hidden = self.model.coordinator.empty_msg_emb(
                torch.zeros(count, device=obs.device, dtype=torch.long)
            ).to(dtype)
            probabilities = exponential.reshape(count, NUM_ACTIONS)
            probabilities = probabilities / probabilities.sum(-1, keepdim=True)
            consensus = probabilities.clamp(min=1e-8).log()
            consensus = (consensus - consensus.mean(-1, keepdim=True)).to(dtype)
            noise = gumbels.reshape(rounds, count, NUM_ACTIONS)
            broadcast = BROADCAST.get(self.mode, "full")
            frozen = consensus if self.mode == "z0-only" else None
            for round_index in range(rounds):
                logits, hidden = self.model._run_round(
                    consensus, hidden, latent, chat, neighbors,
                    z_broadcast=frozen, broadcast_mode=broadcast,
                    shuffle_messages=self.mode == "shuffled",
                )
                vote = (logits / self.model.config.tau
                        + noise[round_index].to(dtype)).argmax(-1)
                target = self.model._centred_log_ohe(vote, NUM_ACTIONS, dtype)
                consensus = consensus + self.model.config.dt * (target - consensus)
            return F.softmax(consensus / self.model.config.tau, dim=-1)


class AblationPolicy(CUDAAOTIPolicy):
    """Eager counterpart of DMMRaggedPolicy: no package, no contract, depth and mode free."""

    def __init__(self, model, *, model_name: str, precision: str, rounds: int,
                 mode: str, max_num_agents: int):
        self.device = torch.device("cuda:0")
        runtime_settings = configure_native_runtime(torch)
        CUDAPolicyRuntime.__init__(
            self,
            rng_factory=lambda seed, device: torch.Generator(device=device).manual_seed(seed),
            pibt="none",
            repeat_escape=False,
            max_repeat_retries=16,
            bfs_budget_bytes=1024**3,
            native_action_ties=True,
        )
        self.call = AblationWholePolicy(model, precision, mode)
        self.round_mode = "sample"
        self.rounds = rounds
        self.max_agents = max_num_agents
        self.metadata = {
            "name": model_name,
            "precision": precision,
            "initial_state": "dirichlet",
            "communication_rounds": rounds,
            "round_mode": "sample",
            "broadcast_mode": mode,
            "final_action": "argmax-probabilities",
            "rng_protocol": "per-task-cuda-exponential-gumbel-eager-v1",
            "history": "executed-actions",
            "cost_to_go_backend": "vendor-window",
            "repeat_escape": False,
            "runtime_numerics": runtime_settings,
            "max_agents_per_call": max_num_agents,
        }
        self.configure_cost_to_go(cache_mode="vendor-window")
