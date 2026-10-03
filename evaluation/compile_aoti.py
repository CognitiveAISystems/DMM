"""Compile a DMM checkpoint to a whole-policy CUDA AOTI package.

The package consumes tokenized observations and neighbor indices. POGEMA uses
Dirichlet z0 and four sampled communication rounds; MovingAI uses zero z0 and
argmax communication rounds. The evaluator applies final argmax and, for
MovingAI, its external CS-PIBT/RSE shield.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from evaluation.models import (MODELS, load_checkpoint_model, policy_for,
                               precision_for, round_mode_for)


NUM_ACTIONS = 5
ROOT = Path(__file__).resolve().parents[1]


class WholePolicy(nn.Module):
    def __init__(self, model: nn.Module, architecture: str, precision: str,
                 round_mode: str):
        super().__init__()
        self.model = model
        self.architecture = architecture
        self.precision = precision
        self.round_mode = round_mode

    def forward(self, obs: Tensor, chat: Tensor, exponential: Tensor,
                gumbels: Tensor) -> Tensor:
        batch, agents, tokens = obs.shape
        count = batch * agents
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.precision == "bf16-fp32"):
            if self.architecture == "canonical-dmm":
                latent = self.model.representation_encoder(obs.reshape(count, tokens))
                dtype = latent.dtype
                neighbors = self.model.msg_nbrs_embedding(
                    torch.arange(self.model.config.max_num_neighbors,
                                 device=obs.device, dtype=torch.long)
                ).to(dtype)
                hidden = self.model.coordinator.empty_msg_emb(
                    torch.zeros(count, device=obs.device, dtype=torch.long)
                ).to(dtype)
            else:
                static_key, static_value, padding, _ = self.model._encode(obs)
                dtype = static_key.dtype
                hidden = self.model.communication.empty_message.reshape(1, 1, -1).expand(
                    batch, agents, -1
                ).to(dtype)
        # Keep sampling and consensus updates in FP32 to avoid accumulating
        # low-precision error across communication rounds.
        if self.round_mode == "sample":
            probabilities = exponential.reshape(count, NUM_ACTIONS).float()
            probabilities = probabilities / probabilities.sum(-1, keepdim=True)
            consensus = probabilities.clamp(min=1e-8).log()
            consensus = consensus - consensus.mean(-1, keepdim=True)
            noise = gumbels.reshape(self.model.config.n_comm_rounds, count, NUM_ACTIONS).float()
        else:
            consensus = torch.zeros_like(exponential.reshape(count, NUM_ACTIONS),
                                         dtype=torch.float32)
        for round_index in range(self.model.config.n_comm_rounds):
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=self.precision == "bf16-fp32"):
                if self.architecture == "canonical-dmm":
                    logits, hidden = self.model._run_round(
                        consensus.to(dtype), hidden, latent, chat, neighbors
                    )
                else:
                    logits, hidden = self.model._run_round(
                        consensus.to(dtype), hidden, chat, static_key, static_value, padding
                    )
            if self.precision == "fp16-fp32":
                logits = logits.nan_to_num(0.0)
            if self.round_mode == "sample":
                vote = (logits.float() / self.model.config.tau
                        + noise[round_index]).argmax(-1)
            else:
                vote = logits.float().argmax(-1)
            target = (self.model._centred_log_ohe(vote, NUM_ACTIONS, torch.float32)
                      if self.architecture == "canonical-dmm"
                      else self.model._centred_log_ohe(vote, torch.float32))
            consensus = consensus + self.model.config.dt * (target - consensus)
        return F.softmax(consensus / self.model.config.tau, dim=-1)


def probe_inputs(count: int, device: torch.device) -> tuple[Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(20260926 + count)
    obs = torch.randint(0, 67, (1, count, 256), device=device,
                        generator=generator, dtype=torch.long)
    chat = torch.full((1, count, 13), -1, device=device, dtype=torch.long)
    positions = torch.arange(count, device=device)
    chat[0, :, 0] = positions
    chat[0, :, 1] = (positions + 1) % count
    exponential = torch.rand((1, count, NUM_ACTIONS), device=device,
                             generator=generator).clamp_min(1e-6)
    uniform = torch.rand((4, 1, count, NUM_ACTIONS), device=device,
                         generator=generator).clamp(1e-6, 1 - 1e-6)
    gumbels = -torch.log(-torch.log(uniform))
    return obs, chat, exponential, gumbels


def smoke_episode(model_name: str, benchmark: str, package: Path,
                  max_agents: int) -> dict:
    """Exercise the compiled package through the actual POGEMA-GPU evaluator."""
    from pogema_gpu.tasks import Task
    from pogema_gpu.toolbox.cuda_evaluator import evaluate_cuda

    task = Task(
        "aoti-smoke", ("............",) * 12,
        ((2, 2), (9, 9)), ((9, 9), (2, 2)),
        policy_seed=0, horizon=8,
    )
    shielded = benchmark == "movingai"
    policy = policy_for(
        model_name, package, benchmark=benchmark, shielded=shielded,
        max_num_agents=max_agents,
    )
    result = evaluate_cuda(
        [task], policy, num_envs=1, record_trace=False,
        movement_backend="resolved", profile_steps=False,
    )
    if len(result["records"]) != 1 or result["records"][0]["task_id"] != task.task_id:
        raise RuntimeError("POGEMA-GPU did not finish the AOTI smoke episode")
    return {"passed": True, "metrics": result["records"][0]["metrics"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--benchmark", choices=("pogema", "movingai"), required=True)
    parser.add_argument("--precision", choices=("fp32", "hybrid"),
                        help="compilation mode; omit to use the model default")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-agents", type=int,
                        help="test-only smaller dynamic limit; omit for the model default")
    parser.add_argument("--max-autotune", action="store_true")
    args = parser.parse_args()
    spec = MODELS[args.model]
    if args.benchmark not in spec["benchmarks"]:
        parser.error(f"{args.model} is not a verified {args.benchmark} model")
    precision = precision_for(args.model, args.precision)
    round_mode = round_mode_for(args.model, args.benchmark)
    max_agents = args.max_agents or spec["max_num_agents"]
    if not 1 <= max_agents <= spec["max_num_agents"]:
        parser.error("max-agents must be within the model's verified profile")
    if args.output.suffix != ".pt2":
        parser.error("output must have a .pt2 suffix")
    if args.output.exists() or Path(str(args.output) + ".json").exists():
        parser.error("refusing to replace an existing package or sidecar")
    if str(torch.__version__) != "2.13.0+cu126" or torch.cuda.device_count() != 1:
        raise RuntimeError("requires PyTorch 2.13.0+cu126 and exactly one visible CUDA GPU")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, config, checkpoint_step = load_checkpoint_model(
        args.model, args.checkpoint_dir / f"{args.model}.pt", torch.device("cuda"))
    network_dtype = {
        "fp32": torch.float32,
        "fp16-fp32": torch.float16,
        "bf16-fp32": torch.bfloat16,
    }[precision]
    model = model.to(dtype=network_dtype)
    wrapper = WholePolicy(model, spec["architecture"], precision,
                          round_mode).cuda().eval()
    device = torch.device("cuda:0")
    example = probe_inputs(min(8, max_agents), device)
    agents = torch.export.Dim("agents", min=1, max=max_agents)
    dynamic_shapes = ({1: agents}, {1: agents}, {1: agents}, {2: agents})
    options = ({"max_autotune": True, "max_autotune_gemm": True}
               if args.max_autotune else {})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        exported = torch.export.export(wrapper, example, dynamic_shapes=dynamic_shapes)
        torch._inductor.aoti_compile_and_package(
            exported, package_path=str(args.output), inductor_configs=options
        )
        compiled = torch._inductor.aoti_load_package(str(args.output))
        tests = []
        for count in sorted({1, min(8, max_agents), min(32, max_agents),
                             min(256, max_agents), max_agents}):
            values = probe_inputs(count, device)
            eager = wrapper(*values).float()
            actual = compiled(*values).float()
            delta = (eager - actual).abs()
            disagreements = int((eager.argmax(-1) != actual.argmax(-1)).sum().item())
            valid = (actual.shape == eager.shape
                     and torch.isfinite(actual).all().item()
                     and torch.allclose(actual.sum(-1), torch.ones(count, device=device), atol=1e-3))
            passed = (valid and (delta.max().item() <= 1e-3 if network_dtype == torch.float32
                                 else disagreements <= max(1, math.ceil(count * 0.02))))
            tests.append({"agents": count, "max_abs": float(delta.max().item()),
                          "action_disagreements": disagreements,
                          "passed": bool(passed)})
    disagreements_total = sum(row["action_disagreements"] for row in tests)
    decisions_total = sum(row["agents"] for row in tests)
    rate = disagreements_total / decisions_total
    probe_passed = all(row["passed"] for row in tests)
    if network_dtype != torch.float32:
        probe_passed = probe_passed and rate <= 0.01
    sidecar = {
        "model_name": args.model,
        "benchmark": args.benchmark,
        "architecture": spec["architecture"],
        "format": "aoti",
        "precision": precision,
        "precision_mode": "fp32" if precision == "fp32" else "hybrid",
        "network_precision": str(network_dtype).split(".")[-1],
        "sampling_precision": "fp32",
        "dynamics_precision": "fp32",
        "input_contract": "explicit-random-v1",
        "output_contract": "N,5 probabilities",
        "initial_z": "dirichlet" if round_mode == "sample" else "zero",
        "round_mode": round_mode,
        "final_action": "argmax",
        "n_comm_rounds": 4,
        "max_num_agents": max_agents,
        "torch_version": str(torch.__version__),
        "checkpoint_step": checkpoint_step,
        "config": asdict(config),
        "compile_options": options,
        "compile_probe": {"passed": probe_passed,
                          "exact_action_parity": disagreements_total == 0,
                          "action_disagreements": disagreements_total,
                          "decisions": decisions_total,
                          "action_disagreement_rate": rate,
                          "cases": tests},
    }
    sidecar_path = Path(str(args.output) + ".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    if not sidecar["compile_probe"]["passed"]:
        raise RuntimeError("compiled AOTI package failed eager parity probes")

    # Check the compiled package in the evaluator after eager/AOTI parity.
    del model, wrapper, exported, compiled, example, values, eager, actual, delta
    gc.collect()
    torch.cuda.empty_cache()
    try:
        sidecar["smoke_episode"] = smoke_episode(
            args.model, args.benchmark, args.output, max_agents
        )
    except Exception:
        sidecar["smoke_episode"] = {"passed": False}
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        raise
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"model": args.model, "package": str(args.output),
                      "compile_probe": sidecar["compile_probe"],
                      "smoke_episode": sidecar["smoke_episode"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
