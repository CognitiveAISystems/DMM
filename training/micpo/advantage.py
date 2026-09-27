"""
MICPO advantage computation.

Episode-level advantage — normalised within each group of G trajectories
that share the same initial map-scenario.  Trajectory differences within a
group are due to policy stochasticity (Categorical action sampling) only.

B  = number of distinct scenarios per iteration  (config.B)
G  = group size: trajectories per scenario        (config.G)
Total envs per iteration = B * G.

Rewards are always [B*G, H, N].  The mixing strategy is controlled by
alpha_individual and reward_mix_post_norm in config:

  Pre-norm  (reward_mix_post_norm=False, default):
      Rewards are mixed before storage: r_mix = (1-a)*mean_n(r_ind) + a*r_ind
      One normalisation pass over the mixed rewards.

  Post-norm (reward_mix_post_norm=True):
      Raw individual rewards stored; mixing happens after normalisation:
          A_ind  = normalise(Q_ind)
          A_team = normalise(Q_team),  Q_team = Q(mean_n(r_ind))
          A_mix  = (1-a)*A_team + a*A_ind
      Individual and team advantages each have unit variance before blending.

Special case B=1:
    compute_advantage_grouped reduces to global normalisation over all G
    trajectories — identical to the B=1 MICPO formulation.
"""

from __future__ import annotations

import torch
from torch import Tensor


_EPS = 1e-8
# Threshold below which std is treated as zero (all returns in group identical).
_STD_ZERO_THRESHOLD = 1e-6


# ------------------------------------------------------------------ #
# Public — pre-norm path (default)
# ------------------------------------------------------------------ #

def compute_advantage_grouped(
    returns: Tensor,
    G: int,
) -> Tensor:
    """
    Normalise returns within each group of G trajectories.

    Args:
        returns: [B*G, N] float32 — per-agent episode returns
        G:       group size (trajectories per scenario)

    Returns:
        [B*G, N] normalised advantages.
    """
    n_envs, N = returns.shape
    B = n_envs // G
    grouped = returns.view(B, G, N)                      # [B, G, N]
    mean = grouped.mean(dim=1, keepdim=True)             # [B, 1, N]
    std  = _safe_std(grouped, dim=1, keepdim=True)       # [B, 1, N]
    centered = grouped - mean
    normed = torch.where(std < _STD_ZERO_THRESHOLD, torch.zeros_like(centered),
                         centered / std)                 # [B, G, N]
    return normed.view(n_envs, N)                        # [B*G, N]


def compute_reward_to_go_advantage_grouped(
    rewards: Tensor,
    G: int,
) -> Tensor:
    """
    Per-timestep reward-to-go advantages, normalised within each group.

    Q[g, t] = Σ_{t'≥t} rewards[g, t']   (γ = 1, no discounting)

    Args:
        rewards: [B*G, H, N] float32
        G:       group size

    Returns:
        [B*G, H, N] normalised per-timestep per-agent advantages.
    """
    Q = rewards.flip(1).cumsum(1).flip(1)                # [B*G, H, N]
    n_envs, H, N = Q.shape
    B = n_envs // G
    grouped  = Q.view(B, G, H, N)
    mean     = grouped.mean(dim=1, keepdim=True)
    std      = _safe_std(grouped, dim=1, keepdim=True)
    centered = grouped - mean
    normed   = torch.where(std < _STD_ZERO_THRESHOLD, torch.zeros_like(centered),
                           centered / std)
    return normed.view(n_envs, H, N)                     # [B*G, H, N]


# ------------------------------------------------------------------ #
# Public — post-norm path
# ------------------------------------------------------------------ #

def compute_post_norm_advantage_grouped(
    rewards: Tensor,
    G: int,
    alpha: float,
    process_supervision: bool,
) -> Tensor:
    """
    Post-normalisation blend of individual and team advantages.

    Individual and team advantages are each normalised to unit variance
    independently before blending with alpha_individual.  This preserves
    the scale of each signal regardless of alpha.

    Pre-condition: rewards contains raw individual rewards [B*G, H, N].

    Args:
        rewards:            [B*G, H, N] individual rewards
        G:                  group size
        alpha:              alpha_individual ∈ [0, 1]
        process_supervision: True → reward-to-go; False → episode return

    Returns:
        outcome supervision: [B*G, N]
        process supervision: [B*G, H, N]
    """
    if process_supervision:
        A_ind  = compute_reward_to_go_advantage_grouped(rewards, G)   # [B*G, H, N]
        A_team = _rtg_advantage_grouped_team(rewards.mean(dim=-1), G) # [B*G, H]
        A_team = A_team.unsqueeze(-1).expand_as(A_ind)                # [B*G, H, N]
    else:
        returns_ind  = rewards.sum(dim=1)            # [B*G, N]
        returns_team = rewards.mean(dim=-1).sum(dim=1)  # [B*G]
        A_ind  = compute_advantage_grouped(returns_ind, G)            # [B*G, N]
        A_team = _advantage_grouped_team(returns_team, G)             # [B*G]
        A_team = A_team.unsqueeze(-1).expand_as(A_ind)                # [B*G, N]

    if alpha == 0.0:
        return A_team
    if alpha == 1.0:
        return A_ind
    return (1.0 - alpha) * A_team + alpha * A_ind


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def compute_advantage(returns: Tensor) -> Tensor:
    """Normalise over all trajectories (B=1 case). Kept for test compatibility."""
    G = returns.shape[0]
    return compute_advantage_grouped(returns, G)


def broadcast_advantage(advantage: Tensor, n_envs: int, H_or_k: int, N: int) -> Tensor:
    """Expand episode-level advantage [n_envs, N] to [n_envs, H_or_k, N]."""
    return advantage.unsqueeze(1).expand(n_envs, H_or_k, N).contiguous()


def select_top_bottom_k_indices(
    returns: Tensor,
    B: int,
    G: int,
    k: int,
) -> Tensor:
    """
    For each of the B scenario groups (each of size G), select the k
    highest-return and k lowest-return trajectory indices.

    Args:
        returns: [B*G, N] — reduced to mean over N for ranking
        B:       number of scenarios
        G:       group size
        k:       number of top / bottom trajectories per group

    Returns:
        LongTensor [B * 2k] — flat indices into the [B*G] buffer axis.
    """
    scalar = returns.mean(dim=-1) if returns.dim() > 1 else returns   # [B*G]
    grouped = scalar.view(B, G)

    selected = []
    for b in range(B):
        order  = grouped[b].argsort()
        bottom = order[:k]
        top    = order[G - k:]
        selected.append(b * G + bottom)
        selected.append(b * G + top)

    return torch.cat(selected)                                         # [B * 2k]


# ------------------------------------------------------------------ #
# Internal
# ------------------------------------------------------------------ #

def _safe_std(x: Tensor, dim: int, keepdim: bool) -> Tensor:
    """
    std that returns 0 instead of NaN for singleton groups (G=1, unbiased=True).
    """
    return torch.nan_to_num(x.std(dim=dim, keepdim=keepdim), nan=0.0)


def _advantage_grouped_team(returns: Tensor, G: int) -> Tensor:
    """
    Normalise 1-D team returns [B*G] within each group.  Returns [B*G].
    Used by the post-norm path to build the team branch.
    """
    n_envs = returns.shape[0]
    B = n_envs // G
    grouped  = returns.view(B, G)
    mean     = grouped.mean(dim=1, keepdim=True)
    std      = _safe_std(grouped, dim=1, keepdim=True)
    centered = grouped - mean
    normed   = torch.where(std < _STD_ZERO_THRESHOLD, torch.zeros_like(centered),
                           centered / std)
    return normed.view(n_envs)                                         # [B*G]


def _rtg_advantage_grouped_team(rewards: Tensor, G: int) -> Tensor:
    """
    Reward-to-go advantages for 1-D team rewards [B*G, H].  Returns [B*G, H].
    Used by the post-norm path to build the team branch.
    """
    Q = rewards.flip(1).cumsum(1).flip(1)                              # [B*G, H]
    n_envs, H = Q.shape
    B = n_envs // G
    grouped  = Q.view(B, G, H)
    mean     = grouped.mean(dim=1, keepdim=True)
    std      = _safe_std(grouped, dim=1, keepdim=True)
    centered = grouped - mean
    normed   = torch.where(std < _STD_ZERO_THRESHOLD, torch.zeros_like(centered),
                           centered / std)
    return normed.view(n_envs, H)                                      # [B*G, H]


def _normalise_1d(x: Tensor) -> Tensor:
    """Zero-mean, unit-variance normalisation over a 1-D tensor."""
    return (x - x.mean()) / (x.std() + _EPS)
