"""
Oneshot validation utilities.

Greedy rollout on fixed scenarios; reports ISR, CSR, makespan, blocked rate.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

log = logging.getLogger(__name__)

import numpy as np
import torch
from training.micpo.instances import build_env_instance_from_seeds, EnvInstance
from training.micpo.collision import count_blocked
from training.micpo.losses import policy_entropy as compute_entropy


def generate_val_scenarios(config, run_dir: Path) -> list[EnvInstance]:
    """
    Generate n_val_scenarios instances at a fixed seed.
    Saved to run_dir/val_scenarios.pt; reloaded on resume.
    """
    save_path = run_dir / config.val_save_path

    if save_path.exists():
        return _load_val_scenarios(save_path)

    rng      = np.random.default_rng(config.val_seed)
    env_mode = config.env_mode
    scenarios: list[EnvInstance] = []

    if env_mode in ("fixed_map_scenario", "custom_map_scenario"):
        scenarios = [build_env_instance_from_seeds(
            config=config,
            map_seed=config.map_seed if env_mode == "fixed_map_scenario" else None,
            scenario_seed=config.scenario_seed,
            custom_map_path=config.custom_map_path if env_mode == "custom_map_scenario" else None,
        )]

    elif env_mode == "fixed_map":
        for _ in range(config.n_val_scenarios):
            scenarios.append(build_env_instance_from_seeds(
                config=config,
                map_seed=config.map_seed,
                scenario_seed=int(rng.integers(0, 2**31)),
            ))

    elif env_mode == "custom_map":
        for _ in range(config.n_val_scenarios):
            scenarios.append(build_env_instance_from_seeds(
                config=config,
                map_seed=None,
                scenario_seed=int(rng.integers(0, 2**31)),
                custom_map_path=config.custom_map_path,
            ))

    else:  # "random"
        val_num_agents = getattr(config, "val_num_agents", None) or None
        h_min = getattr(config, "map_height_min", None)
        h_max = getattr(config, "map_height_max", None)
        w_min = getattr(config, "map_width_min", None)
        w_max = getattr(config, "map_width_max", None)
        for _ in range(config.n_val_scenarios):
            for _attempt in range(10):
                h = int(rng.integers(h_min, h_max + 1)) if h_min is not None and h_max is not None else None
                w = int(rng.integers(w_min, w_max + 1)) if w_min is not None and w_max is not None else None
                try:
                    scenarios.append(build_env_instance_from_seeds(
                        config=config,
                        map_seed=int(rng.integers(0, 2**31)),
                        scenario_seed=int(rng.integers(0, 2**31)),
                        height=h,
                        width=w,
                        num_agents=val_num_agents,
                    ))
                    break
                except RuntimeError as _exc:
                    if _attempt == 9:
                        raise
                    log.warning("val build_env_instance_from_seeds failed (%s); retrying…", _exc)

    _save_val_scenarios(scenarios, save_path)
    return scenarios


def _save_val_scenarios(scenarios: list[EnvInstance], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "grid":          s.grid,
            "positions":     s.positions,
            "goals":         s.goals,
            "height":        s.height,
            "width":         s.width,
            "map_seed":      s.map_seed,
            "scenario_seed": s.scenario_seed,
        }
        for s in scenarios
    ]
    torch.save(data, path)


def _load_val_scenarios(path: Path) -> list[EnvInstance]:
    data = torch.load(path, weights_only=False)
    return [
        EnvInstance(
            grid=d["grid"],
            positions=d["positions"],
            goals=d["goals"],
            height=d["height"],
            width=d["width"],
            map_seed=d["map_seed"],
            scenario_seed=d["scenario_seed"],
        )
        for d in data
    ]


# ------------------------------------------------------------------ #
# Validation rollout
# ------------------------------------------------------------------ #

def _is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cudamalloc failed" in msg


def _seed_validation_step(config, scenario_idx: int, step: int) -> None:
    """Make stochastic inference reproducible without changing its distribution."""
    if not getattr(config, "val_deterministic_inference", False):
        return
    seed = (
        int(getattr(config, "val_inference_seed", 0))
        + 1_000_003 * scenario_idx
        + step
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _round_metrics(round_counts: list[int]) -> dict[str, float]:
    if not round_counts:
        return {}
    ordered = sorted(round_counts)
    p95_idx = max(0, int(np.ceil(0.95 * len(ordered))) - 1)
    return {
        "val/mean_comm_rounds": float(np.mean(round_counts)),
        "val/p95_comm_rounds": float(ordered[p95_idx]),
        "val/max_comm_rounds": float(ordered[-1]),
        "val/total_comm_rounds": float(sum(round_counts)),
    }


def _policy_logits(policy, obs, chat_ids, config, device, *, scenario_idx=None, step=None):
    """Select the validation inference protocol by policy class."""
    if config.policy_class == "dmm":
        if scenario_idx is not None:
            _seed_validation_step(config, scenario_idx, step)
        return policy(obs, chat_ids).logits
    if config.policy_class == "dmm08m":
        use_bf16 = bool(getattr(config, "use_fp16", False)) and device.type == "cuda"
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            if getattr(config, "val_deterministic_rounds", False):
                return policy.deterministic_zero_act(obs, chat_ids).logits
            return policy(obs, chat_ids).logits
    raise ValueError(f"Unsupported policy_class: {config.policy_class!r}")


@torch.no_grad()
def _run_val_sequential(
    policy,
    val_scenarios: list[EnvInstance],
    config,
    device: torch.device,
) -> dict[str, float]:
    from training.micpo.environment import MICPOEnvironment
    policy.eval()

    isr_list, csr_list, makespan_list, blocked_list, entropy_list = [], [], [], [], []
    sum_cost_list, mean_cost_list = [], []
    round_counts: list[int] = []

    for scenario_idx, inst in enumerate(val_scenarios):
        env = MICPOEnvironment(
            grid=inst.grid,
            positions=inst.positions,
            goals=inst.goals,
            max_episode_steps=config.max_horizon,
            device=str(device),
            on_target="nothing",
            observation_config=config,
        )

        goals_t        = env.goals.clone()   # [N, 2]
        ep_blocked     = 0
        ep_entropy     = 0.0
        ep_makespan    = config.max_horizon
        done           = False
        ep_cost        = torch.zeros(config.num_agents, device=device)   # steps off goal per agent

        for step in range(config.max_horizon):
            obs_t, chat_ids = env.observe()

            logits = _policy_logits(policy,
                obs_t.unsqueeze(0),
                chat_ids.unsqueeze(0),
                config, device, scenario_idx=scenario_idx, step=step)
            logits = logits.float().nan_to_num(0.0).squeeze(0)
            if getattr(config, "val_greedy", True):
                actions = logits.argmax(dim=-1)
            else:
                actions = torch.distributions.Categorical(
                    logits=logits
                ).sample()
            if config.policy_class == "dmm":
                round_counts.append(int(config.n_comm_rounds))

            pos_before = env.pos.clone()
            env.step(actions)
            pos_after  = env.pos

            ep_blocked += count_blocked(pos_before, pos_after, actions)
            ep_entropy += compute_entropy(logits).item()
            # Count steps where each agent is not at its goal
            off_goal = ~(pos_after == goals_t).all(dim=-1)   # [N] bool
            ep_cost += off_goal.float()

            if not done:
                all_on_goal = (pos_after == goals_t).all(dim=-1).all()
                if all_on_goal:
                    ep_makespan = step + 1
                    done = True
                    break

        on_goal_final = (env.pos == goals_t).all(dim=-1)   # [N]
        isr_list.append(float(on_goal_final.float().mean().item()))
        csr_list.append(1.0 if on_goal_final.all().item() else 0.0)
        makespan_list.append(ep_makespan)
        blocked_list.append(ep_blocked)
        entropy_list.append(ep_entropy / max(1, ep_makespan))
        sum_cost_list.append(float(ep_cost.sum().item()))
        mean_cost_list.append(float(ep_cost.mean().item()))

    policy.train()

    metrics = {
        "val/mean_isr":      float(np.mean(isr_list)),
        "val/mean_csr":      float(np.mean(csr_list)),
        "val/mean_makespan": float(np.mean(makespan_list)),
        "val/mean_blocked":  float(np.mean(blocked_list)),
        "val/mean_entropy":  float(np.mean(entropy_list)),
        "val/mean_sum_cost": float(np.mean(sum_cost_list)),
        "val/mean_cost":     float(np.mean(mean_cost_list)),
    }
    if config.policy_class == "dmm":
        metrics.update(_round_metrics(round_counts))
    return metrics


@torch.no_grad()
def _run_val_batched(
    policy,
    val_scenarios: list[EnvInstance],
    config,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, float]:
    from training.micpo.environment import MICPOEnvironment

    batch_size = max(1, int(batch_size))
    policy.eval()

    isr_list, csr_list, makespan_list, blocked_list, entropy_list = [], [], [], [], []
    sum_cost_list, mean_cost_list = [], []

    for batch_start in range(0, len(val_scenarios), batch_size):
        batch = val_scenarios[batch_start: batch_start + batch_size]
        envs, goals_list = [], []
        stats = []

        for inst in batch:
            env = MICPOEnvironment(
                grid=inst.grid,
                positions=inst.positions,
                goals=inst.goals,
                max_episode_steps=config.max_horizon,
                device=str(device),
                on_target="nothing",
                observation_config=config,
            )
            envs.append(env)
            goals_list.append(env.goals.clone())
            stats.append({
                "blocked": 0, "entropy": 0.0, "makespan": config.max_horizon, "done": False,
                "cost": torch.zeros(config.num_agents, device=device),
            })

        for step in range(config.max_horizon):
            obs_list, chat_list = [], []
            for env in envs:
                observations, neighbors = env.observe()
                obs_list.append(observations)
                chat_list.append(neighbors)

            obs_b    = torch.stack(obs_list)   # [B, N, 256]
            cid_b    = torch.stack(chat_list)  # [B, N, 13]
            logits_b = _policy_logits(
                policy, obs_b, cid_b, config, device
            ).float().nan_to_num(0.0)   # [B, N, 5]
            if getattr(config, "val_greedy", True):
                actions_b = logits_b.argmax(dim=-1)                          # [B, N]
            else:
                actions_b = torch.distributions.Categorical(logits=logits_b).sample()  # [B, N]

            for idx, (env, s, goals_t) in enumerate(zip(envs, stats, goals_list)):
                actions = actions_b[idx]
                pos_before = env.pos.clone()
                env.step(actions)
                pos_after  = env.pos

                s["blocked"] += count_blocked(pos_before, pos_after, actions)
                s["entropy"] += compute_entropy(logits_b[idx: idx + 1]).item()
                s["cost"]    += (~(pos_after == goals_t).all(dim=-1)).float()
                if not s["done"] and (pos_after == goals_t).all(dim=-1).all():
                    s["makespan"] = step + 1
                    s["done"]     = True

            del obs_list, chat_list, obs_b, cid_b, logits_b, actions_b

            if all(s["done"] for s in stats):
                break

        for env, s, goals_t in zip(envs, stats, goals_list):
            on_goal_final = (env.pos == goals_t).all(dim=-1)
            isr_list.append(float(on_goal_final.float().mean().item()))
            csr_list.append(1.0 if on_goal_final.all().item() else 0.0)
            makespan_list.append(s["makespan"])
            blocked_list.append(s["blocked"])
            entropy_list.append(s["entropy"] / max(1, s["makespan"]))
            sum_cost_list.append(float(s["cost"].sum().item()))
            mean_cost_list.append(float(s["cost"].mean().item()))

        del envs, goals_list, stats
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gc.collect()
    policy.train()

    metrics = {
        "val/mean_isr":      float(np.mean(isr_list)),
        "val/mean_csr":      float(np.mean(csr_list)),
        "val/mean_makespan": float(np.mean(makespan_list)),
        "val/mean_blocked":  float(np.mean(blocked_list)),
        "val/mean_entropy":  float(np.mean(entropy_list)),
        "val/mean_sum_cost": float(np.mean(sum_cost_list)),
        "val/mean_cost":     float(np.mean(mean_cost_list)),
    }
    if config.policy_class == "dmm":
        fixed_rounds = float(config.n_comm_rounds)
        metrics.update({
            "val/mean_comm_rounds": fixed_rounds,
            "val/p95_comm_rounds": fixed_rounds,
            "val/max_comm_rounds": fixed_rounds,
            "val/total_comm_rounds": float(
                config.n_comm_rounds * sum(makespan_list)
            ),
        })
    return metrics


def run_validation(
    policy,
    val_scenarios: list[EnvInstance],
    config,
    device: torch.device,
) -> dict[str, float]:
    """
    Greedy rollout on all val_scenarios.
    Dispatches to batched inference when val_batch_size > 1.
    On CUDA OOM, halves batch_size and retries.
    """
    requested  = int(getattr(config, "val_batch_size", 1))
    batch_size = max(1, len(val_scenarios)) if requested <= 0 else max(1, requested)
    auto_oom   = bool(getattr(config, "val_auto_batch_on_oom", True))
    while True:
        try:
            if batch_size <= 1:
                return _run_val_sequential(policy, val_scenarios, config, device)
            return _run_val_batched(policy, val_scenarios, config, device, batch_size=batch_size)
        except RuntimeError as exc:
            if not auto_oom or batch_size <= 1 or not _is_cuda_oom(exc):
                raise
            batch_size = max(1, batch_size // 2)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
