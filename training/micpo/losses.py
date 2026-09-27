"""
MICPO loss functions for DMM (per-round IS ratios).

DMM produces K+1 categorical votes per MAPF step. Each round gets its own
IS ratio and clipped PG objective. The advantage A is the same for all K+1
rounds of the same MAPF step (trajectory-level signal broadcast).

L_clip — mean over (batch, N, K+1) of clipped PG objective
L_kl   — sum over rounds of exact categorical KL(pi_theta_t || pi_ref_t)
L_ent  — mean over rounds of entropy bonus
L      = L_clip + L_kl + L_ent

Why per-round IS ratios (not summed):
  Summing log-pis gives a valid compound IS ratio but it accumulates
  multiplicatively over K+1 rounds, producing very large/small values.
  Per-round ratios stay near 1.0 and are individually clipped — more stable.

Why exact KL (not MICPO sample estimator):
  5 actions — summing over the full action space is essentially free.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_importance_ratio(
    log_pi_rounds_theta: Tensor,  # [*, N, K+1]
    log_pi_old_r:        Tensor,  # [*, N, K+1]
) -> Tensor:
    """
    Per-round IS ratio: r_t = exp(log_pi_theta_t - log_pi_old_t).
    Returns [*, N, K+1].
    """
    return (log_pi_rounds_theta - log_pi_old_r).exp()


def clip_pg_loss(
    log_pi_rounds_theta: Tensor,  # [batch, N, K+1]
    log_pi_old_r:        Tensor,  # [batch, N, K+1]
    advantage:           Tensor,  # [batch, N]   — trajectory-level, broadcast to K+1 rounds
    eps_clip:            float,
) -> Tensor:
    """
    Clipped policy gradient loss over all rounds.

    L_clip = −mean_{batch, N, K+1} [ min(r_t * A, clip(r_t, 1−ε, 1+ε) * A) ]

    Advantage A is the same for all K+1 rounds of the same MAPF step.
    """
    ratio         = compute_importance_ratio(log_pi_rounds_theta, log_pi_old_r)  # [batch, N, K+1]
    ratio_clipped = ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip)

    adv = advantage.unsqueeze(-1)  # [batch, N, 1] → broadcast to [batch, N, K+1]
    obj  = torch.min(ratio * adv, ratio_clipped * adv)
    return -obj.mean()


def kl_categorical(
    per_round_logits_theta: Tensor,  # [batch, N, K+1, 5]
    per_round_logits_ref:   Tensor,  # [batch, N, K+1, 5]
    alpha_kl: float,
) -> Tensor:
    """
    Exact categorical KL summed over rounds, meaned over batch and N.

    L_kl = alpha_kl * mean_{batch,N} sum_t KL(pi_theta_t || pi_ref_t)
    """
    # Flatten (batch, N) and round dimensions for torch.distributions
    batch, N, R, A = per_round_logits_theta.shape
    flat_theta = per_round_logits_theta.reshape(batch * N * R, A)
    flat_ref   = per_round_logits_ref.reshape(batch * N * R, A)

    pi_theta = torch.distributions.Categorical(logits=flat_theta)
    pi_ref   = torch.distributions.Categorical(logits=flat_ref)
    kl_flat  = torch.distributions.kl_divergence(pi_theta, pi_ref)  # [batch*N*R]

    # Reshape to [batch, N, R] and sum over rounds, mean over (batch, N)
    kl = kl_flat.reshape(batch, N, R).sum(-1).mean()
    return alpha_kl * kl


def entropy_bonus(
    per_round_logits_theta: Tensor,  # [batch, N, K+1, 5]
    alpha_entropy: float,
) -> Tensor:
    """
    Entropy regularisation: -alpha_entropy * mean_t H(pi_theta_t).

    Averaged over rounds, batch, and N.
    """
    batch, N, R, A = per_round_logits_theta.shape
    flat = per_round_logits_theta.reshape(batch * N * R, A)
    dist = torch.distributions.Categorical(logits=flat)
    return -alpha_entropy * dist.entropy().mean()


def total_loss(L_clip: Tensor, L_kl: Tensor, L_ent: Tensor) -> Tensor:
    return L_clip + L_kl + L_ent


def policy_entropy(logits: Tensor) -> Tensor:
    """
    Mean policy entropy over batch and N. Accepts either:
      - [*, N, 5]      (single-round policy logits)
      - [*, N, K+1, 5] (per-round logits)
    Used for logging only.
    """
    if logits.dim() == 4:
        B, N, R, A = logits.shape
        flat = logits.reshape(B * N * R, A)
    else:
        flat = logits.reshape(-1, logits.shape[-1])
    dist = torch.distributions.Categorical(logits=flat)
    return dist.entropy().mean()
