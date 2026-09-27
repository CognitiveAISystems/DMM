"""
Checkpointing utilities — DMM-compatible format.

Checkpoint fields:
  "model"        — raw_model.state_dict()
  "model_args"   — DMM architecture arguments
  "iter_num"     — completed optimizer steps
  "completed_iterations" — completed outer training iterations
  "best_val_loss"— best validation ISR
  "config"       — OneshotMICPOConfig.to_dict()

Extra MICPO fields (ignored by DMM inference, needed for training resume):
  "optimizer"    — optimizer.state_dict()
  "scaler"       — GradScaler state (empty dict if CPU/bfloat16)
  "ref_model"    — pi_ref.state_dict()
  "rng_states"   — per-rank torch, CUDA, and iteration-generator states
  "run_dir"      — str path

Two checkpoint types:
  ckpt_latest.pt      — overwritten every `latest_ckpt_freq` iters (crash recovery)
  ckpt_step_{N}.pt    — permanent, saved every `ckpt_freq` iters
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def resolve_run_dir(runs_dir: str, run_name: str, resume: bool = False) -> Path:
    """
    Return the run directory.

    resume=False: create runs_dir/run_name (appending _0, _1, … on collision).
    resume=True:  return runs_dir/run_name as-is (must exist).
    """
    base = Path(runs_dir)

    if resume:
        run_dir = base / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Cannot resume: run directory not found: {run_dir}")
        return run_dir

    candidate = base / run_name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    idx = 0
    while True:
        candidate = base / f"{run_name}_{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        idx += 1


def save_checkpoint(
    run_dir: Path,
    step: int,
    completed_iterations: int,
    raw_model,
    optimizer,
    scaler,
    config,
    model_args: dict,
    best_val_isr: float,
    val_step: int,
    periodic: bool,
    rng_states: list[dict],
    pi_ref=None,
) -> None:
    """
    Save checkpoint.

    Always overwrites ckpt_latest.pt.
    If periodic=True, also writes ckpt_step_{step}.pt.
    """
    payload = {
        # DMM-compatible keys
        "model":          raw_model.state_dict(),
        "model_args":     model_args,
        "iter_num":       step,
        "completed_iterations": completed_iterations,
        "best_val_loss":  best_val_isr,   # name kept for DMM compat
        "config":         config.to_dict(),
        # MICPO-specific (ignored by DMM)
        "optimizer":      optimizer.state_dict(),
        "scaler":         scaler.state_dict() if scaler is not None else {},
        "ref_model":      pi_ref.state_dict() if pi_ref is not None else None,
        "rng_states":     rng_states,
        "run_dir":        str(run_dir),
        "val_step":       val_step,
    }

    latest_path = run_dir / "ckpt_latest.pt"
    torch.save(payload, latest_path)

    if periodic:
        torch.save(payload, run_dir / f"ckpt_step_{step}.pt")


def capture_rng_state(iter_rng: np.random.Generator, device: torch.device) -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "iteration": iter_rng.bit_generator.state,
    }


def gather_rng_states(local_state: dict, world_size: int) -> list[dict]:
    if world_size == 1:
        return [local_state]
    states = [None] * world_size
    dist.all_gather_object(states, local_state)
    return states


def restore_rng_state(state: dict, iter_rng: np.random.Generator, device: torch.device) -> None:
    torch.set_rng_state(state["torch"].cpu().byte())
    if device.type == "cuda":
        if state["cuda"] is None:
            raise ValueError("Checkpoint is missing the CUDA RNG state")
        torch.cuda.set_rng_state(state["cuda"].cpu().byte(), device)
    iter_rng.bit_generator.state = state["iteration"]


def uncompiled_state_dict(model) -> dict:
    """Weights for a plain reference policy, without torch.compile's prefix."""
    return getattr(model, "_orig_mod", model).state_dict()


def resume_position(ckpt: dict, n_iters: int, world_size: int) -> tuple[int, int, list[dict]]:
    if "completed_iterations" not in ckpt or "rng_states" not in ckpt:
        raise ValueError(
            "This checkpoint predates exact MICPO resume metadata "
            "(completed_iterations and per-rank rng_states); "
            "its training position cannot be recovered safely."
        )
    step = ckpt["iter_num"]
    completed_iterations = ckpt["completed_iterations"]
    rng_states = ckpt["rng_states"]
    if not 0 <= completed_iterations <= n_iters:
        raise ValueError(
            f"Checkpoint completed_iterations={completed_iterations} is outside "
            f"the configured n_iters={n_iters}"
        )
    if len(rng_states) != world_size:
        raise ValueError(
            f"Checkpoint has RNG states for {len(rng_states)} ranks, "
            f"but this run has {world_size} ranks"
        )
    return step, completed_iterations, rng_states


def load_checkpoint(run_dir: str | Path, device: torch.device) -> dict:
    path = Path(run_dir) / "ckpt_latest.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")
    return torch.load(path, map_location=device, weights_only=False)


def should_save_latest(step: int, freq: int) -> bool:
    return freq > 0 and step % freq == 0


def should_save_periodic(step: int, freq: int) -> bool:
    return freq > 0 and step % freq == 0
