"""Training-loop helpers shared by both MICPO model sizes."""

from __future__ import annotations

from contextlib import nullcontext
import math

import torch

from training.micpo.advantage import broadcast_advantage


def log_scalars(loggers: dict, scalars: dict, step: int) -> None:
    tb = loggers.get("tb")
    if tb is not None:
        for k, v in scalars.items():
            tb.add_scalar(k, v, step)
    w = loggers.get("wandb")
    if w is not None:
        w.log(scalars, step=step)


# ------------------------------------------------------------------ #
# LR schedule
# ------------------------------------------------------------------ #

def get_lr(step: int, config) -> float:
    if not config.decay_lr:
        return config.lr
    if step < config.warmup_iters:
        return config.lr * step / max(1, config.warmup_iters)
    if step > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (step - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.lr - config.min_lr)


# ------------------------------------------------------------------ #
# Minibatch iterator
# ------------------------------------------------------------------ #

def iter_minibatches(sub_buffer, advantage_expanded, minibatch_size: int,
                     logits_ref_rounds_flat=None):
    """
    Yield (obs, chat_ids, votes, z0, log_pi_old_r, adv, logits_ref_rounds) minibatches.

    sub_buffer:              SubsampledBuffer [G, k, N, ...]
    advantage_expanded:      [G, k, N]
    minibatch_size:          0 → one full batch
    logits_ref_rounds_flat:  [G*k, N, K, 5] cached ref logits or None
    """
    G, k, N = sub_buffer.actions.shape
    total    = G * k

    obs      = sub_buffer.obs.reshape(total, N, -1)
    chat_ids = sub_buffer.agent_chat_ids.reshape(total, N, -1)
    votes    = sub_buffer.votes.reshape(total, N, -1)          # [total, N, K]
    z0       = sub_buffer.z0.reshape(total, N, 5)              # [total, N, 5]
    log_pi_r = sub_buffer.log_pi_old_r.reshape(total, N, -1)  # [total, N, K]
    adv      = advantage_expanded.reshape(total, N)

    if minibatch_size <= 0 or minibatch_size >= total:
        ref_mb = logits_ref_rounds_flat if logits_ref_rounds_flat is not None else None
        yield obs, chat_ids, votes, z0, log_pi_r, adv, ref_mb
        return

    perm = torch.randperm(total, device=obs.device)
    for start in range(0, total, minibatch_size):
        idx    = perm[start: start + minibatch_size]
        ref_mb = logits_ref_rounds_flat[idx] if logits_ref_rounds_flat is not None else None
        yield obs[idx], chat_ids[idx], votes[idx], z0[idx], log_pi_r[idx], adv[idx], ref_mb


# ------------------------------------------------------------------ #
# Advantage → expanded [n_envs, k, N]
# ------------------------------------------------------------------ #

def build_adv_expanded(
    advantages,
    sub_buffer,
    n_envs: int,
    N: int,
    process_supervision: bool,
) -> "torch.Tensor":
    k = sub_buffer.k
    if not process_supervision:
        return broadcast_advantage(advantages, n_envs, k, N)
    indices = sub_buffer.timestep_indices
    idx_exp = indices.unsqueeze(-1).expand(n_envs, k, N)
    return advantages.gather(1, idx_exp).contiguous()


# ------------------------------------------------------------------ #
# Reference model logprobs (for SR and KL cache)
# ------------------------------------------------------------------ #

def _compute_ref_logprobs(pi_ref, buffer, device, ctx=None):
    """
    Run pi_ref.forward_micpo over the full buffer with stored votes and z0.

    Returns:
        log_pi  [G, H, N]       — sum of per-round log-probs (for SR/CPR)
        logits  [G, H, N, K, 5] — per-round logits (cached for KL in epoch loop)
    """
    if ctx is None:
        ctx = nullcontext()

    G, H, N, C  = buffer.obs.shape
    L            = buffer.agent_chat_ids.shape[-1]
    R            = buffer.votes.shape[-1]

    obs_flat   = buffer.obs.view(G * H, N, C)
    cid_flat   = buffer.agent_chat_ids.view(G * H, N, L)
    votes_flat = buffer.votes.view(G * H, N, R).reshape(G * H * N, R)
    z0_flat    = buffer.z0.view(G * H, N, 5)

    chunk = G * H
    while True:
        lp_chunks, lg_chunks = [], []
        try:
            with torch.no_grad(), ctx:
                for start in range(0, G * H, chunk):
                    end         = min(start + chunk, G * H)
                    chunk_size  = end - start
                    v_chunk     = votes_flat[start * N: end * N].reshape(chunk_size * N, R)
                    z0_chunk    = z0_flat[start:end].reshape(chunk_size * N, 5)
                    lp_r, lg_r = pi_ref.forward_micpo(
                        obs_flat[start:end].nan_to_num(0.0),
                        cid_flat[start:end],
                        v_chunk,
                        z0_chunk,
                    )
                    # lp_r: [chunk*N, R]   lg_r: [chunk*N, R, 5]
                    lp_chunks.append(lp_r.reshape(chunk_size, N, R))
                    lg_chunks.append(lg_r.reshape(chunk_size, N, R, 5))
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or chunk <= G:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            chunk = max(G, chunk // 2)

    log_pi_r = torch.cat(lp_chunks, dim=0).view(G, H, N, R)      # [G, H, N, R]
    logits   = torch.cat(lg_chunks, dim=0).view(G, H, N, R, 5)   # [G, H, N, R, 5]

    log_pi_sum = log_pi_r.sum(-1)  # [G, H, N] — sum over rounds for SR
    return log_pi_sum, logits


def _subsample_ref_logits(logits_full, sub_buf):
    """
    Index [G, H, N, R, 5] → [G*k, N, R, 5] using sub_buf.timestep_indices.
    """
    if logits_full is None:
        return None
    G_, k_, N_ = sub_buf.timestep_indices.shape[0], sub_buf.k, logits_full.shape[2]
    R           = logits_full.shape[3]
    A           = logits_full.shape[4]
    idx = sub_buf.timestep_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(G_, k_, N_, R, A)
    return logits_full.gather(1, idx).reshape(G_ * k_, N_, R, A)


def _apply_cpr(adv_exp, cpr_full, sub_buf, w):
    if cpr_full is None or w == 0.0:
        return adv_exp
    G_, k_ = sub_buf.timestep_indices.shape
    N_     = cpr_full.shape[2]
    idx    = sub_buf.timestep_indices.unsqueeze(-1).expand(G_, k_, N_)
    cpr_sub = cpr_full.gather(1, idx)
    return adv_exp + w * cpr_sub
