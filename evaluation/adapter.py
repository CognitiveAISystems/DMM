"""AOTI policy adapter shared by the two DMM architectures."""

import json
from pathlib import Path

import torch

from pogema_gpu.adapters.cuda_runtime import CUDAPolicyRuntime
from pogema_gpu.adapters.runtime import configure_native_runtime
from pogema_gpu.adapters.aoti import CUDAAOTIPolicy

class DMMRaggedPolicy(CUDAAOTIPolicy):
    def __init__(
        self,
        package,
        *,
        model_name,
        architecture,
        precision,
        max_num_agents,
        benchmark,
        round_mode,
        shielded=True,
    ):
        if torch.__version__.split(".")[:2] != ["2", "13"]:
            raise RuntimeError(f"{model_name} AOTI requires PyTorch 2.13")
        package = Path(package)
        contract = json.loads(Path(str(package) + ".json").read_text())
        expected = {
            "model_name": model_name, "benchmark": benchmark,
            "precision": precision,
            "input_contract": "explicit-random-v1",
            "initial_z": "dirichlet" if round_mode == "sample" else "zero",
            "round_mode": round_mode,
            "n_comm_rounds": 4, "max_num_agents": max_num_agents,
            "torch_version": str(torch.__version__),
        }
        if contract.get("architecture") is not None:
            expected["architecture"] = architecture
        for key, value in expected.items():
            if contract.get(key) != value:
                raise ValueError(f"{model_name} package contract mismatch: {key}")
        if contract.get("compile_probe", {}).get("passed") is not True:
            raise ValueError(f"{model_name} package failed compilation checks")
        if ("smoke_episode" in contract
                and contract["smoke_episode"].get("passed") is not True):
            raise ValueError(f"{model_name} package failed evaluator smoke test")
        if round_mode not in {"sample", "argmax"}:
            raise ValueError(f"unsupported communication round mode: {round_mode}")
        self.round_mode = round_mode
        self.device = torch.device("cuda:0")
        runtime_settings = configure_native_runtime(torch)
        CUDAPolicyRuntime.__init__(
            self,
            rng_factory=lambda seed, device: torch.Generator(device=device).manual_seed(seed),
            pibt="sequential" if shielded else "none",
            repeat_escape=shielded,
            max_repeat_retries=16,
            bfs_budget_bytes=1024**3,
            native_action_ties=True,
        )
        with torch.cuda.device(self.device):
            self.call = torch._inductor.aoti_load_package(str(package))
        self.max_agents = max_num_agents
        self.metadata = {
            "name": model_name,
            "checkpoint_step": contract.get("checkpoint_step"),
            "torch_version": str(torch.__version__),
            "precision": precision,
            "initial_state": "dirichlet" if round_mode == "sample" else "zero",
            "communication_rounds": 4,
            "round_mode": round_mode,
            "final_action": "argmax-probabilities",
            "rng_protocol": (
                "per-task-cuda-exponential-gumbel-explicit-random-v1"
                if round_mode == "sample"
                else "deterministic-zero-z0-argmax-v1"
            ),
            "history": "post-shield-actions" if shielded else "executed-actions",
            "cost_to_go_backend": "vendor-window",
            "repeat_escape": shielded,
            "runtime_numerics": runtime_settings,
            "max_agents_per_aoti_call": max_num_agents,
        }
        self.shield_config.annotate(
            self.metadata, scores="softmax-final-z", history="post-shield-actions"
        )

    def reset(self, batch):
        if batch.n > self.max_agents:
            raise ValueError("task exceeds the DMM AOTI agent limit")
        CUDAPolicyRuntime.reset(self, batch)
