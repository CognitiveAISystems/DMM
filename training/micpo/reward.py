"""
Trajectory-wise return computation for DMM MICPO.

Full reward formula (outcome supervision — single scalar per agent per episode):

  C[n]   — per-agent cost, controlled by config.cost_mode:
              "goal_reached": steps until first arrival at goal (= H if never)
              "on_goal":      total steps where pos[n] != goal[n]

  B[n]   — total blocked steps (intended to move, did not move)

  SR_ind[n]  = Σ_h  log_π(a[h,n]) · mask[h,n]
  SR_team    = Σ_h  min_n( log_π(a[h,n]) · mask[h,n] )   ← bottleneck agent
  mask[h,n] follows cost_mode:
    "goal_reached": 1 if h < first_arrival[n]
    "on_goal":      1 if pos[h,n] != goal[n]

  R[n] = − w_makespan  ·  T_ep
         − w_cost      · [ (1−α_cost)    · C[n]      + α_cost    · mean_n(C[n]) ]
         − w_blocked   · [ (1−α_blocked) · B[n]      + α_blocked · mean_n(B[n]) ]
         + w_sr_makespan · SR_team
         + w_sr_cost   · [ (1−α_sr)      · SR_ind[n] + α_sr      · mean_n(SR_ind) ]

  T_ep = episode_length[g]  (makespan or max_horizon if unsolved)

All quantities are trajectory-level scalars; no per-step decomposition.
Returns shape [G, N].
"""

from __future__ import annotations

import torch
from torch import Tensor

from training.micpo.advantage import compute_advantage_grouped


def _apply_transform(x: Tensor, transform: str) -> Tensor:
    """Apply a monotone transform to a non-negative reward magnitude tensor."""
    if transform == "none":
        return x
    if transform == "log":
        return (x + 1.0).log()   # +1 to avoid log(0) when x=0
    if transform == "sqrt":
        return x.sqrt()
    raise ValueError(f"Unknown transform: {transform!r}")


def _blend(individual: Tensor, alpha: float) -> Tensor:
    """
    Convex blend: (1-alpha)*individual + alpha*mean_across_agents.

    individual: [G, N]
    Returns:    [G, N]
    """
    if alpha == 0.0:
        return individual
    mean = individual.mean(dim=-1, keepdim=True)   # [G, 1]
    if alpha == 1.0:
        return mean.expand_as(individual)
    return (1.0 - alpha) * individual + alpha * mean


def compute_cost(
    pos_after:      Tensor,   # [G, H, N, 2]
    goals:          Tensor,   # [G, N, 2]
    first_arrival:  Tensor,   # [G, N]  int64 — step of first arrival (H if never)
    episode_length: Tensor,   # [G]     int64
    cost_mode:      str,
) -> Tensor:
    """
    Compute per-agent cost C[n].

    "goal_reached" → first_arrival[g,n]
    "on_goal"      → number of steps in [0, T_ep) where pos[n] != goal[n]

    Returns [G, N] float32.
    """
    G, H, N, _ = pos_after.shape

    if cost_mode == "goal_reached":
        return first_arrival.float()   # [G, N]

    if cost_mode == "on_goal":
        # on_goal[g,h,n] = True when pos == goal at step h
        on_goal = (pos_after == goals.unsqueeze(1)).all(dim=-1)   # [G, H, N] bool

        # Mask out steps beyond episode_length[g] — only count valid steps.
        step_idx = torch.arange(H, device=pos_after.device).view(1, H, 1)  # [1, H, 1]
        ep_len   = episode_length.view(G, 1, 1)                             # [G, 1, 1]
        valid    = step_idx < ep_len                                         # [G, H, N] broadcast

        off_goal = (~on_goal) & valid                                        # [G, H, N]
        return off_goal.sum(dim=1).float()                                   # [G, N]

    raise ValueError(f"Unknown cost_mode: {cost_mode!r}")


def compute_blocked_per_agent(
    pos_before: Tensor,   # [G, H, N, 2]
    pos_after:  Tensor,   # [G, H, N, 2]
    actions:    Tensor,   # [G, H, N]
    episode_length: Tensor,  # [G]
) -> Tensor:
    """
    Total blocked steps per agent, counting only steps within episode_length.
    Returns [G, N] float32.
    """
    G, H, N, _ = pos_before.shape

    intended_to_move = (actions != 0)                                         # [G, H, N]
    stayed           = (pos_after == pos_before).all(dim=-1)                  # [G, H, N]
    blocked          = intended_to_move & stayed                               # [G, H, N]

    step_idx = torch.arange(H, device=pos_before.device).view(1, H, 1)
    ep_len   = episode_length.view(G, 1, 1)
    valid    = step_idx < ep_len

    return (blocked & valid).sum(dim=1).float()   # [G, N]


def compute_selfreward(
    log_pi:         Tensor,   # [G, H, N]
    pos_after:      Tensor,   # [G, H, N, 2]
    goals:          Tensor,   # [G, N, 2]
    first_arrival:  Tensor,   # [G, N]
    episode_length: Tensor,   # [G]
    cost_mode:      str,
    sr_team_mode:   str = "mean",   # "mean" | "min"
    sr_complement:  bool = False,   # True → use log(1-p) instead of log(p)
) -> tuple[Tensor, Tensor]:
    """
    Compute SR_ind[n] and SR_team (scalar per env).

    mask follows cost_mode:
      "goal_reached" — mask[h,n] = 1 if h < first_arrival[n]
      "on_goal"      — mask[h,n] = 1 if pos[h,n] != goal[n]

    Also masks out steps beyond episode_length[g].

    SR_team: at each step h, aggregate over agents that are still active
    (mask[h,n]=1) using sr_team_mode ("mean" or "min").
    Agents already on goal are excluded so a fast agent does not pull
    the team signal toward zero.
    At steps where NO agent is active the contribution is 0.

    sr_complement=True: replace log_pi with log(1 - p) = log1p(-exp(log_pi)).
    Less negative when the policy is uncertain; use with positive w_sr_* to
    promote entropy without sign gymnastics.

    Returns:
        SR_ind  [G, N]  — individual SR signal sum under mask
        SR_team [G]     — Σ_h agg_{n: mask[h,n]=1}(sr_signal[h,n])
    """
    G, H, N = log_pi.shape

    step_idx = torch.arange(H, device=log_pi.device).view(1, H, 1)   # [1, H, 1]
    ep_len   = episode_length.view(G, 1, 1)                           # [G, 1, 1]
    valid    = step_idx < ep_len                                       # [G, H, N]

    if cost_mode == "goal_reached":
        fa   = first_arrival.view(G, 1, N)                            # [G, 1, N]
        mask = (step_idx < fa) & valid                                 # [G, H, N]
    elif cost_mode == "on_goal":
        on_goal = (pos_after == goals.unsqueeze(1)).all(dim=-1)        # [G, H, N]
        mask    = (~on_goal) & valid                                   # [G, H, N]
    else:
        raise ValueError(f"Unknown cost_mode: {cost_mode!r}")

    if sr_complement:
        sr_signal = torch.log1p(-log_pi.exp().clamp(max=1.0 - 1e-6))  # [G, H, N]
    else:
        sr_signal = log_pi

    SR_ind = (sr_signal * mask.float()).sum(dim=1)                     # [G, N]

    # SR_team: aggregate over active agents only per step.
    any_active = mask.any(dim=-1)                                        # [G, H]

    if sr_team_mode == "min":
        # Fill inactive slots with +INF so they never win the min.
        INF      = torch.finfo(sr_signal.dtype).max
        sig_act  = torch.where(mask, sr_signal, torch.full_like(sr_signal, INF))
        step_agg = sig_act.min(dim=-1).values                           # [G, H]
    elif sr_team_mode == "mean":
        # Sum active sr_signal then divide by number of active agents.
        n_active = mask.float().sum(dim=-1).clamp(min=1.0)              # [G, H]
        step_agg = (sr_signal * mask.float()).sum(dim=-1) / n_active    # [G, H]
    else:
        raise ValueError(f"Unknown sr_team_mode: {sr_team_mode!r}")

    # Zero out steps where no agent was active.
    step_agg = torch.where(any_active, step_agg, torch.zeros_like(step_agg))
    SR_team  = step_agg.sum(dim=1)                                      # [G]

    return SR_ind, SR_team


def compute_cpr_advantage(
    log_pi_old: Tensor,   # [G, H, N] — stored in buffer (from pi_old at rollout time)
    log_pi_ref: Tensor,   # [G, H, N] — from _compute_ref_logprobs
    buffer,               # OneshotBuffer
    config,               # OneshotMICPOConfig
    G_group: int,         # group size for within-group normalisation (= config.G)
) -> Tensor:
    """
    Cumulative Process Reward (CPR) advantage, adapted from SPRO (arXiv 2507.01551).

    For each agent n at each step h:
        log_ratio[g,h,n] = (log_pi_old[g,h,n] - log_pi_ref[g,h,n]) * mask[g,h,n]
        cpr_ind[g,h,n]   = Σ_{j=0}^{h} log_ratio[g,j,n]   (prefix cumsum)

    mask follows config.cost_mode + episode_length (same convention as SR).

    If alpha_cpr > 0, a team-aggregated CPR is computed per step via sr_team_mode
    (same as SR team mode), cumsummed, and blended:
        cpr = (1 - alpha_cpr) * cpr_ind + alpha_cpr * cpr_team

    The result is normalised within each group of G_group trajectories:
        reshape [G, H, N] → [G, H*N], apply compute_advantage_grouped(..., G_group),
        reshape back → [G, H, N].

    Returns [G, H, N] float32 — zero-mean, unit-variance within each G_group.
    """
    G, H, N = log_pi_old.shape
    device = log_pi_old.device

    # ------------------------------------------------------------------ #
    # Mask (same as SR)
    # ------------------------------------------------------------------ #
    step_idx = torch.arange(H, device=device).view(1, H, 1)   # [1, H, 1]
    ep_len   = buffer.episode_length.view(G, 1, 1)             # [G, 1, 1]
    valid    = step_idx < ep_len                                # [G, H, N]

    if config.cost_mode == "goal_reached":
        fa   = buffer.first_arrival.view(G, 1, N)
        mask = (step_idx < fa) & valid
    elif config.cost_mode == "on_goal":
        on_goal = (buffer.pos_after == buffer.goals.unsqueeze(1)).all(dim=-1)  # [G, H, N]
        mask    = (~on_goal) & valid
    else:
        raise ValueError(f"Unknown cost_mode: {config.cost_mode!r}")

    # ------------------------------------------------------------------ #
    # Log-ratio, masked
    # ------------------------------------------------------------------ #
    log_ratio = (log_pi_old - log_pi_ref) * mask.float()   # [G, H, N]

    # ------------------------------------------------------------------ #
    # Individual CPR — prefix cumsum
    # ------------------------------------------------------------------ #
    cpr_ind = log_ratio.cumsum(dim=1)   # [G, H, N]

    # ------------------------------------------------------------------ #
    # Team CPR (optional blend)
    # ------------------------------------------------------------------ #
    alpha_cpr    = getattr(config, "alpha_cpr", 0.0)
    sr_team_mode = getattr(config, "sr_team_mode", "mean")

    if alpha_cpr > 0.0:
        any_active = mask.any(dim=-1)   # [G, H]

        if sr_team_mode == "min":
            INF      = torch.finfo(log_ratio.dtype).max
            lr_act   = torch.where(mask, log_ratio, torch.full_like(log_ratio, INF))
            step_agg = lr_act.min(dim=-1).values                          # [G, H]
        elif sr_team_mode == "mean":
            n_active = mask.float().sum(dim=-1).clamp(min=1.0)            # [G, H]
            step_agg = log_ratio.sum(dim=-1) / n_active                   # [G, H]
        else:
            raise ValueError(f"Unknown sr_team_mode: {sr_team_mode!r}")

        step_agg = torch.where(any_active, step_agg, torch.zeros_like(step_agg))
        cpr_team_1d = step_agg.cumsum(dim=1)                              # [G, H]
        cpr_team    = cpr_team_1d.unsqueeze(-1).expand_as(cpr_ind)        # [G, H, N]

        cpr = (1.0 - alpha_cpr) * cpr_ind + alpha_cpr * cpr_team
    else:
        cpr = cpr_ind

    # ------------------------------------------------------------------ #
    # Normalise within each group of G_group trajectories.
    # Reshape [G, H, N] → [G, H*N] for group-wise operations.
    #
    # cpr_normalize_std=True  (default): full z-score like outcome advantage —
    #     w_sr_cpr is a clean relative-scale knob.
    # cpr_normalize_std=False (paper-style): subtract group mean only —
    #     CPR retains its natural scale; small when policy ≈ ref.
    # ------------------------------------------------------------------ #
    cpr_flat = cpr.reshape(G, H * N)   # [G, H*N]

    if getattr(config, "cpr_normalize_std", True):
        cpr_norm_flat = compute_advantage_grouped(cpr_flat, G_group)
    else:
        B_     = G // G_group
        grouped = cpr_flat.view(B_, G_group, H * N)
        mean    = grouped.mean(dim=1, keepdim=True)
        cpr_norm_flat = (grouped - mean).view(G, H * N)

    return cpr_norm_flat.reshape(G, H, N)   # [G, H, N]


def compute_oneshot_return(
    buffer,        # OneshotBuffer
    log_pi_self: "Tensor | None",   # [G, H, N] — from buffer or pi_ref; None if unused
    config,        # OneshotMICPOConfig
) -> Tensor:
    """
    Compute trajectory-wise returns R[g, n] for all G envs and N agents.

    Returns [G, N] float32.
    """
    G, H, N, _ = buffer.pos_after.shape
    device = buffer.pos_after.device

    T_ep = buffer.episode_length.float()       # [G] — raw makespan, always logged as-is
    T_ep_reward = _apply_transform(T_ep, getattr(config, "makespan_transform", "none"))

    # ------------------------------------------------------------------ #
    # Cost C[n]
    # ------------------------------------------------------------------ #
    C = compute_cost(
        pos_after=buffer.pos_after,
        goals=buffer.goals,
        first_arrival=buffer.first_arrival,
        episode_length=buffer.episode_length,
        cost_mode=config.cost_mode,
    )   # [G, N]

    # ------------------------------------------------------------------ #
    # Blocked B[n]
    # ------------------------------------------------------------------ #
    B_per = compute_blocked_per_agent(
        pos_before=buffer.pos_before,
        pos_after=buffer.pos_after,
        actions=buffer.actions,
        episode_length=buffer.episode_length,
    )   # [G, N]

    # ------------------------------------------------------------------ #
    # Self-reward
    # ------------------------------------------------------------------ #
    sr_cost_term   = torch.zeros(G, N, device=device)
    sr_mk_term     = torch.zeros(G, N, device=device)

    if (config.w_sr_cost != 0.0 or config.w_sr_makespan != 0.0) and log_pi_self is not None:
        SR_ind, SR_team = compute_selfreward(
            log_pi=log_pi_self,
            pos_after=buffer.pos_after,
            goals=buffer.goals,
            first_arrival=buffer.first_arrival,
            episode_length=buffer.episode_length,
            cost_mode=config.cost_mode,
            sr_team_mode=getattr(config, "sr_team_mode", "mean"),
            sr_complement=getattr(config, "sr_complement", False),
        )   # [G, N], [G]

        if config.w_sr_cost != 0.0:
            sr_cost_term = config.w_sr_cost * _blend(SR_ind, config.alpha_sr)

        if config.w_sr_makespan != 0.0:
            # SR_team is [G] — broadcast to [G, N]
            sr_mk_term = config.w_sr_makespan * SR_team.unsqueeze(-1).expand(G, N)

    # ------------------------------------------------------------------ #
    # Assemble total return
    # ------------------------------------------------------------------ #
    R = torch.zeros(G, N, device=device)

    # Makespan: T_ep_reward broadcast to all agents (T_ep kept raw for logging)
    if config.w_makespan != 0.0:
        R -= config.w_makespan * T_ep_reward.unsqueeze(-1).expand(G, N)

    # Cost blend
    if config.w_cost != 0.0:
        C_reward = _apply_transform(C, getattr(config, "cost_transform", "none"))
        R -= config.w_cost * _blend(C_reward, config.alpha_cost)

    # Blocked blend
    if config.w_blocked != 0.0:
        R -= config.w_blocked * _blend(B_per, config.alpha_blocked)

    # Move penalty: penalise any action != 0 within episode_length
    w_move = float(getattr(config, "w_move", 0.0))
    if w_move != 0.0:
        step_idx  = torch.arange(H, device=device).view(1, H, 1)
        ep_len    = buffer.episode_length.view(G, 1, 1)
        valid     = step_idx < ep_len                               # [G, H, N]
        moved     = (buffer.actions != 0) & valid                   # [G, H, N]
        M_per     = moved.sum(dim=1).float()                        # [G, N]
        R -= w_move * _blend(M_per, getattr(config, "alpha_move", 0.0))

    # Self-reward
    R += sr_cost_term + sr_mk_term

    return R


def compute_oneshot_components(
    buffer,
    log_pi_self: "Tensor | None",
    config,
) -> "dict[str, tuple[Tensor, float, float]]":
    """
    Compute raw (un-weighted, un-blended) reward components for the post-norm path.

    Returns each active component as (ind [G,N], alpha, weight) where:
      ind   — raw per-agent values, no alpha blending applied yet
      alpha — blend coefficient (0=pure individual, 1=pure team-mean)
              team signals (makespan, sr_mk) use alpha=0 since ind is already broadcast
      weight — w_* scalar

    The caller normalises ind and mean_n(ind) independently within groups,
    blends with alpha, then scales by weight.
    """
    G, H, N, _ = buffer.pos_after.shape
    device = buffer.pos_after.device
    T_ep   = buffer.episode_length.float()   # [G]

    T_ep_reward = _apply_transform(T_ep, getattr(config, "makespan_transform", "none"))

    out: dict = {}

    if config.w_makespan != 0.0:
        # Team signal: same value for all agents — alpha blending is a no-op.
        out["ms"] = (-T_ep_reward.unsqueeze(-1).expand(G, N).clone(), 0.0, config.w_makespan)

    if config.w_cost != 0.0:
        C = compute_cost(
            pos_after=buffer.pos_after,
            goals=buffer.goals,
            first_arrival=buffer.first_arrival,
            episode_length=buffer.episode_length,
            cost_mode=config.cost_mode,
        )
        C_reward = _apply_transform(C, getattr(config, "cost_transform", "none"))
        out["cost"] = (-C_reward, config.alpha_cost, config.w_cost)

    if config.w_blocked != 0.0:
        B_per = compute_blocked_per_agent(
            pos_before=buffer.pos_before,
            pos_after=buffer.pos_after,
            actions=buffer.actions,
            episode_length=buffer.episode_length,
        )
        out["blocked"] = (-B_per, config.alpha_blocked, config.w_blocked)

    if (config.w_sr_cost != 0.0 or config.w_sr_makespan != 0.0) and log_pi_self is not None:
        SR_ind, SR_team = compute_selfreward(
            log_pi=log_pi_self,
            pos_after=buffer.pos_after,
            goals=buffer.goals,
            first_arrival=buffer.first_arrival,
            episode_length=buffer.episode_length,
            cost_mode=config.cost_mode,
            sr_team_mode=getattr(config, "sr_team_mode", "mean"),
            sr_complement=getattr(config, "sr_complement", False),
        )
        if config.w_sr_cost != 0.0:
            out["sr_cost"] = (SR_ind, config.alpha_sr, config.w_sr_cost)
        if config.w_sr_makespan != 0.0:
            # Team signal: already aggregated across agents.
            out["sr_mk"] = (SR_team.unsqueeze(-1).expand(G, N).clone(), 0.0, config.w_sr_makespan)

    w_move = float(getattr(config, "w_move", 0.0))
    if w_move != 0.0:
        step_idx = torch.arange(H, device=device).view(1, H, 1)
        ep_len   = buffer.episode_length.view(G, 1, 1)
        valid    = step_idx < ep_len
        moved    = (buffer.actions != 0) & valid
        M_per    = moved.sum(dim=1).float()   # [G, N]
        out["move"] = (-M_per, getattr(config, "alpha_move", 0.0), w_move)

    return out


# ------------------------------------------------------------------ #
# Metric helpers (used for logging, not for training)
# ------------------------------------------------------------------ #

def compute_isr(buffer) -> float:
    """
    Individual Success Rate: fraction of agents at their goal at T_ep.
    """
    G, H, N, _ = buffer.pos_after.shape
    # pos_after at the last valid step for each env
    ep_idx = (buffer.episode_length - 1).clamp(0, H - 1)   # [G]
    last_pos = torch.stack([
        buffer.pos_after[g, ep_idx[g]]
        for g in range(G)
    ])   # [G, N, 2]
    on_goal = (last_pos == buffer.goals).all(dim=-1)   # [G, N]
    return float(on_goal.float().mean().item())


def compute_csr(buffer) -> float:
    """
    Complete Solution Rate: fraction of envs where all N agents are at goal at T_ep.
    """
    G, H, N, _ = buffer.pos_after.shape
    ep_idx = (buffer.episode_length - 1).clamp(0, H - 1)   # [G]
    last_pos = torch.stack([
        buffer.pos_after[g, ep_idx[g]]
        for g in range(G)
    ])   # [G, N, 2]
    on_goal = (last_pos == buffer.goals).all(dim=-1)   # [G, N]
    csr     = on_goal.all(dim=-1).float()               # [G]
    return float(csr.mean().item())


def compute_mean_makespan(buffer) -> float:
    """Mean episode length across all G envs."""
    return float(buffer.episode_length.float().mean().item())
