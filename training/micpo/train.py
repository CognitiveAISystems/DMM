"""
MICPO training entry point for both DMM architectures.

Each MAPF step the model runs K communication rounds.
Each round produces an IS ratio; advantages are trajectory-level and broadcast
to all K rounds of the same step.
Subsampling is on the MAPF-step axis — all rounds are kept together.

Usage:
    # Single GPU:
    python -m training.micpo --config training/micpo/dmm_08m.yaml --run_name my_run

    # CLI overrides:
    python -m training.micpo --config training/micpo/dmm_08m.yaml --G=16 --lr=3e-4

    # Multi-GPU (torchrun):
    torchrun --nproc_per_node=4 -m training.micpo --config training/micpo/dmm_08m.yaml

    # Resume:
    python -m training.micpo --config training/micpo/dmm_08m.yaml \\
        --init_from=resume --resume_run_dir=runs/my_run
"""

from __future__ import annotations

import argparse
import copy
from importlib import import_module
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from training.micpo.config import OneshotMICPOConfig
from training.micpo.transition import subsample_buffer
from training.micpo.rollout import build_rollout
from training.micpo.reward import (
    compute_oneshot_return,
    compute_oneshot_components,
    compute_cpr_advantage,
    compute_cost,
    compute_isr,
    compute_csr,
    compute_mean_makespan,
)
from training.micpo.advantage import (
    compute_advantage_grouped,
    compute_reward_to_go_advantage_grouped,
    select_top_bottom_k_indices,
)
from training.micpo.losses import (
    clip_pg_loss, kl_categorical, entropy_bonus, total_loss,
    policy_entropy, compute_importance_ratio,
)
from training.micpo.ddp_utils import init_ddp, cleanup_ddp, barrier
from training.micpo.checkpointing import (
    resolve_run_dir, save_checkpoint, load_checkpoint,
    should_save_latest, should_save_periodic,
    capture_rng_state, gather_rng_states, restore_rng_state, resume_position,
    uncompiled_state_dict,
)
from training.micpo.driver_utils import (
    log_scalars, get_lr, iter_minibatches, build_adv_expanded,
    _compute_ref_logprobs, _subsample_ref_logits, _apply_cpr,
)


# ------------------------------------------------------------------ #
# Logging backends
# ------------------------------------------------------------------ #

def _build_logger(run_dir: Path, policy_class: str) -> dict:
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(log_dir=str(run_dir / "tb"))

    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", f"micpo-oneshot-{policy_class}"),
                name=run_dir.name,
                dir=str(run_dir),
                resume="allow",
            )
        except Exception as e:
            print(f"[WARNING] wandb init failed: {e}")

    return {"tb": tb, "wandb": wandb_run}


def main():
    parser = argparse.ArgumentParser(description="MICPO oneshot DMM training")
    parser.add_argument("--config", type=str, required=True)
    args, overrides = parser.parse_known_args()

    # ---------------------------------------------------------------- #
    # Config
    # ---------------------------------------------------------------- #
    config_path = Path(args.config)
    if not config_path.is_file():
        parser.error(f"Config not found: {config_path}")
    config = OneshotMICPOConfig.from_yaml(str(config_path))

    cli_overrides = [a for a in overrides if a.startswith("--") and "=" in a]
    if cli_overrides:
        config = config.apply_overrides(cli_overrides)

    # One optimizer/replay/checkpoint loop with model-specific policy and
    # scenario generation.
    model_kind = config.policy_class
    if model_kind not in {"dmm08m", "dmm"}:
        raise ValueError(f"Unsupported policy_class: {config.policy_class!r}")
    policy_module = import_module(f"training.micpo.{model_kind}_policy")
    from training.micpo.instances import build_env_instance
    from training.micpo.validation import generate_val_scenarios, run_validation
    policy_type = getattr(
        policy_module,
        {"dmm08m": "DMM08MPolicy", "dmm": "DMMPolicy"}[config.policy_class],
    )
    build_policy = policy_module.build_policy
    _model_args_from_config = policy_module._model_args_from_config

    # ---------------------------------------------------------------- #
    # DDP init
    # ---------------------------------------------------------------- #
    rank, local_rank, world_size, is_master = init_ddp(config.backend)

    device_str = (
        f"cuda:{local_rank}"
        if "cuda" in config.device and torch.cuda.is_available()
        else "cpu"
    )
    device = torch.device(device_str)

    if "cuda" in config.device and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU")
        config = config.apply_overrides(["--device=cpu", "--use_fp16=False"])

    torch.manual_seed(42 + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed(42 + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---------------------------------------------------------------- #
    # Run directory
    # ---------------------------------------------------------------- #
    resuming = config.init_from == "resume"

    if resuming:
        if config.resume_run_dir is None:
            raise ValueError("resume_run_dir must be set when init_from='resume'")
        run_dir = Path(config.resume_run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run_dir not found: {run_dir}")
    else:
        if is_master:
            run_dir = resolve_run_dir(config.runs_dir, config.run_name, resume=False)
        else:
            run_dir = None

        if world_size > 1:
            import torch.distributed as dist
            if is_master:
                run_dir_str   = str(run_dir)
                run_dir_len   = torch.tensor(len(run_dir_str), dtype=torch.long, device=device)
                dist.broadcast(run_dir_len, src=0)
                run_dir_bytes = torch.ByteTensor(list(run_dir_str.encode())).to(device)
                dist.broadcast(run_dir_bytes, src=0)
            else:
                run_dir_len   = torch.tensor(0, dtype=torch.long, device=device)
                dist.broadcast(run_dir_len, src=0)
                run_dir_bytes = torch.zeros(run_dir_len.item(), dtype=torch.uint8, device=device)
                dist.broadcast(run_dir_bytes, src=0)
                run_dir = Path(run_dir_bytes.cpu().numpy().tobytes().decode())

    run_dir = Path(run_dir)

    # ---------------------------------------------------------------- #
    # Logging
    # ---------------------------------------------------------------- #
    loggers = {}
    if is_master:
        loggers = _build_logger(run_dir, config.policy_class)
        from loguru import logger as _loguru
        _loguru.add(str(run_dir / "log.txt"), rotation="100 MB", enqueue=True)
        log = _loguru
    else:
        class _Noop:
            def info(self, *a, **k): pass
            def warning(self, *a, **k): pass
        log = _Noop()

    # ---------------------------------------------------------------- #
    # AMP autocast
    # ---------------------------------------------------------------- #
    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp     = config.use_fp16 and device_type == "cuda"
    if use_amp:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx   = torch.amp.autocast(device_type=device_type, dtype=dtype)
    else:
        dtype = torch.float32
        ctx   = nullcontext()

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and dtype == torch.float16))

    # ---------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------- #
    start_step    = 0
    start_iteration = 0
    best_val_isr  = 0.0
    val_step      = 0

    if resuming:
        ckpt = load_checkpoint(run_dir, device)
        saved_cfg = OneshotMICPOConfig.from_dict(ckpt["config"])
        if saved_cfg.policy_class != config.policy_class:
            raise ValueError(
                "Resume policy_class does not match the selected backend: "
                f"{saved_cfg.policy_class!r} != {config.policy_class!r}"
            )
        if cli_overrides:
            saved_cfg = saved_cfg.apply_overrides(cli_overrides)
        if saved_cfg.policy_class != config.policy_class:
            raise ValueError("Cannot change policy_class while resuming")
        config = saved_cfg.apply_overrides([f"--device={device_str}"])

        model_args = ckpt["model_args"]
        pi_theta   = policy_type.from_scratch(model_args, device)
        raw_sd = ckpt["model"]
        raw_sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in raw_sd.items()}
        missing, unexpected = pi_theta.load_state_dict(raw_sd, strict=False)
        if missing:
            log.warning(f"Missing keys when loading model: {missing}")
        if unexpected:
            log.warning(f"Unexpected keys when loading model: {unexpected}")

        start_step, start_iteration, rng_states = resume_position(
            ckpt, config.n_iters, world_size
        )
        best_val_isr = ckpt.get("best_val_loss", 0.0)
        val_step     = ckpt.get("val_step", 0)
    else:
        if config.init_from == "scratch":
            log.info("Initialising model from scratch")
        else:
            log.info(f"Loading pretrained weights from: {config.path_to_weights}")
        pi_theta = build_policy(config, device)

        if config.init_from == "scratch":
            model_args = _model_args_from_config(config)
        else:
            ckpt_tmp   = torch.load(config.path_to_weights, map_location=device, weights_only=False)
            model_args = dict(
                ckpt_tmp.get("model_config")
                or ckpt_tmp.get("model_args")
                or {}
            )
            model_args["n_comm_rounds"] = config.n_comm_rounds

    log.info(f"Parameters: {pi_theta.get_num_params() / 1e6:.2f}M")

    pi_theta.dir_alpha = config.dir_alpha

    if config.freeze_encoder:
        for p in pi_theta.representation_encoder.parameters():
            p.requires_grad_(False)
        log.info("Encoder frozen")

    # Reference policy.
    pi_ref = copy.deepcopy(pi_theta)  # inherits dir_alpha via deepcopy
    pi_ref.to(device).eval()
    for p in pi_ref.parameters():
        p.requires_grad_(False)

    # Optimizer.
    optimizer = pi_theta.configure_optimizers(
        config.weight_decay, config.lr, (config.beta1, config.beta2), device_type
    )

    if resuming:
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        ref_sd = ckpt.get("ref_model")
        if ref_sd is not None:
            ref_sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in ref_sd.items()}
            pi_ref.load_state_dict(ref_sd)
            log.info("pi_ref restored from checkpoint")

    # Compile BEFORE DDP — torch.compile must see the raw module.
    if config.use_compile and device_type == "cuda":
        log.info("Compiling model…")
        pi_theta = torch.compile(pi_theta, mode="reduce-overhead")

    # DDP wrap after compile.
    if world_size > 1:
        pi_theta = DDP(pi_theta, device_ids=[local_rank])
    raw_model = pi_theta.module if isinstance(pi_theta, DDP) else pi_theta

    # ---------------------------------------------------------------- #
    # Save config
    # ---------------------------------------------------------------- #
    if is_master:
        config.save_yaml(str(run_dir / "config.yaml"))

    # ---------------------------------------------------------------- #
    # Validation scenarios
    # ---------------------------------------------------------------- #
    val_scenarios = None
    if is_master:
        val_scenarios = generate_val_scenarios(config, run_dir)
        log.info(f"Validation set: {len(val_scenarios)} scenarios")

    barrier()

    # ---------------------------------------------------------------- #
    # Pre-allocate pi_old (plain uncompiled model; weights updated each iter)
    # Never deepcopy a compiled model — it can inherit fp16 graph state.
    # ---------------------------------------------------------------- #
    pi_old = policy_type.from_scratch(model_args, device)
    pi_old.eval()
    for p in pi_old.parameters():
        p.requires_grad_(False)

    def _sync_pi_old():
        pi_old.load_state_dict(uncompiled_state_dict(raw_model))

    # ---------------------------------------------------------------- #
    # Training loop
    # ---------------------------------------------------------------- #
    iter_rng    = np.random.default_rng(42 + rank)
    if resuming:
        # Restore after model, DDP, and validation setup, which may consume RNG.
        restore_rng_state(rng_states[rank], iter_rng, device)
    global_step = start_step
    completed_iterations = start_iteration

    log.info(f"Starting training — run_dir: {run_dir}")
    if resuming:
        log.info(
            f"Resuming at iteration {start_iteration}/{config.n_iters}, "
            f"optimizer step {global_step}"
        )
    _precision = "bf16" if use_amp and dtype == torch.bfloat16 else str(dtype)
    log.info(
        f"Execution mode: precision={_precision} "
        f"z0={getattr(config, 'rollout_z0_mode', 'dirichlet')} "
        "rollout_votes=sampled"
    )

    if is_master and not resuming:
        log.info("Running reference policy validation (step 0 baseline)…")
        ref_metrics = run_validation(pi_ref, val_scenarios, config, device)
        ref_metrics_prefixed = {k.replace("val/", "val/ref_"): v for k, v in ref_metrics.items()}
        log_scalars(loggers, ref_metrics_prefixed, 0)
        log.info(f"[ref baseline step 0] {ref_metrics}")

    for iteration in range(start_iteration, config.n_iters):

        t_iter_start = time.monotonic()

        # ------------------------------------------------------------ #
        # 1. Sync pi_old weights from current model (no deepcopy of compiled)
        # ------------------------------------------------------------ #
        _sync_pi_old()

        # ------------------------------------------------------------ #
        # 2. Build B × z0_groups × G env instances
        # ------------------------------------------------------------ #
        M      = config.z0_groups
        n_envs = config.B * M * config.G
        base_instances = []
        for _ in range(config.B):
            for _attempt in range(10):
                try:
                    base_instances.append(build_env_instance(config, iter_rng))
                    break
                except RuntimeError as _exc:
                    if _attempt == 9:
                        raise
                    log.warning("build_env_instance failed (%s); retrying…", _exc)

        env_instances = [inst for inst in base_instances for _ in range(M * config.G)]

        # ------------------------------------------------------------ #
        # 3. Collect rollouts (uses pi_old.rollout_act internally)
        # ------------------------------------------------------------ #
        rollout_runner  = build_rollout(pi_old, env_instances, config, device,
                                        n_z0_groups=M)
        buffer          = rollout_runner.collect()
        n_blocked_total = buffer.n_blocked_total
        if device.type == "cuda":
            torch.cuda.synchronize()

        # ------------------------------------------------------------ #
        # 4. Self-reward / CPR log_pi tensors (if requested)
        # ------------------------------------------------------------ #
        log_pi_self          = None
        log_pi_ref_full      = None
        logits_ref_r_cache   = None   # [G, H, N, K, 5]

        need_ref_forward = (
            (config.w_sr_cost != 0.0 or config.w_sr_makespan != 0.0)
            and config.selfreward_source == "reference"
        ) or config.w_sr_cpr != 0.0

        if need_ref_forward:
            log_pi_ref_full, logits_ref_r_cache = _compute_ref_logprobs(pi_ref, buffer, device, ctx)

        if config.w_sr_cost != 0.0 or config.w_sr_makespan != 0.0:
            if config.selfreward_source == "rollout":
                log_pi_self = buffer.log_pi_old_i           # [G, H, N] — sum over rounds
            else:
                log_pi_self = log_pi_ref_full

        # ------------------------------------------------------------ #
        # 5. Compute trajectory-wise returns [G, N]
        # ------------------------------------------------------------ #
        returns = compute_oneshot_return(buffer, log_pi_self, config)

        # ------------------------------------------------------------ #
        # 5b. CPR advantage [G, H, N]
        # ------------------------------------------------------------ #
        cpr_adv = None
        if config.w_sr_cpr != 0.0 and log_pi_ref_full is not None:
            cpr_adv = compute_cpr_advantage(
                buffer.log_pi_old_i, log_pi_ref_full, buffer, config, config.G
            )

        # ------------------------------------------------------------ #
        # 6. Compute advantages
        # ------------------------------------------------------------ #
        if config.process_supervision:
            ep_len = buffer.episode_length.float().clamp(min=1)
            G, H, N = buffer.obs.shape[:3]
            step_reward_proxy = (
                returns / ep_len.unsqueeze(-1)
            ).unsqueeze(1).expand(G, H, N)
            advantages = compute_reward_to_go_advantage_grouped(
                step_reward_proxy.contiguous(), config.G
            )
        elif config.reward_mix_post_norm:
            components = compute_oneshot_components(buffer, log_pi_self, config)
            advantages = torch.zeros_like(returns)
            for ind, alpha, w in components.values():
                A_ind = compute_advantage_grouped(ind, config.G)
                if alpha == 0.0:
                    A_comp = A_ind
                else:
                    team_bc = ind.mean(dim=-1, keepdim=True).expand_as(ind).contiguous()
                    A_team  = compute_advantage_grouped(team_bc, config.G)
                    A_comp  = (1.0 - alpha) * A_ind + alpha * A_team
                advantages = advantages + w * A_comp
            advantages = compute_advantage_grouped(advantages, config.G)
        else:
            advantages = compute_advantage_grouped(returns, config.G)

        # Compute the reward scale audit before top/bottom filtering, while the
        # full G-sized groups are still intact.
        reward_audit_values = {}
        if config.reward_audit:
            reward_components_for_audit = compute_oneshot_components(
                buffer, log_pi_self, config
            )
            for name, (ind, alpha, weight) in reward_components_for_audit.items():
                team = ind.mean(dim=-1)
                a_ind = compute_advantage_grouped(ind, config.G)
                if alpha == 0.0:
                    a_comp = a_ind
                else:
                    team_bc = team.unsqueeze(-1).expand_as(ind).contiguous()
                    a_team = compute_advantage_grouped(team_bc, config.G)
                    a_comp = (1.0 - alpha) * a_ind + alpha * a_team
                reward_audit_values[name] = {
                    "raw_mean": float(team.mean().item()),
                    "raw_std": float(team.std(unbiased=False).item()),
                    "norm_std": float(a_comp.std(unbiased=False).item()),
                    "effective_rms": float(
                        (weight * a_comp).square().mean().sqrt().item()
                    ),
                }

        # ------------------------------------------------------------ #
        # 7. Top-k / bottom-k filtering
        # ------------------------------------------------------------ #
        k_filter = config.top_bottom_k
        if k_filter > 0:
            if 2 * k_filter >= config.G:
                log.warning(
                    f"top_bottom_k={k_filter} with G={config.G}: 2k >= G — no filtering effect"
                )
            else:
                keep       = select_top_bottom_k_indices(returns, config.B * M, config.G, k_filter)
                buffer     = buffer.filter_envs(keep)
                advantages = advantages[keep]
                returns    = returns[keep]
                n_envs     = buffer.G
                if logits_ref_r_cache is not None:
                    logits_ref_r_cache = logits_ref_r_cache[keep]
                if cpr_adv is not None:
                    cpr_adv = cpr_adv[keep]

        # ------------------------------------------------------------ #
        # 8. Timestep subsampling (on MAPF step axis, all rounds kept)
        # ------------------------------------------------------------ #
        if not config.resample_per_epoch:
            sub_buffer      = subsample_buffer(buffer, config.timestep_sample_ratio, config.timestep_sample_k)
            adv_expanded    = build_adv_expanded(
                advantages, sub_buffer, n_envs, config.num_agents,
                config.process_supervision,
            )
            adv_expanded         = _apply_cpr(adv_expanded, cpr_adv, sub_buffer, config.w_sr_cpr)
            logits_ref_r_flat    = _subsample_ref_logits(logits_ref_r_cache, sub_buffer)

        if config.resample_per_epoch:
            logits_ref_r_flat = None

        # ------------------------------------------------------------ #
        # 9. Epoch loop
        # ------------------------------------------------------------ #
        epoch_losses, epoch_clip, epoch_kl = [], [], []
        epoch_ent, epoch_r_mean, epoch_r_max, epoch_entropy = [], [], [], []

        pi_theta.train()

        for epoch in range(config.n_epochs):
            if config.resample_per_epoch:
                sub_buffer      = subsample_buffer(buffer, config.timestep_sample_ratio, config.timestep_sample_k)
                adv_expanded    = build_adv_expanded(
                    advantages, sub_buffer, n_envs, config.num_agents,
                    config.process_supervision,
                )
                adv_expanded      = _apply_cpr(adv_expanded, cpr_adv, sub_buffer, config.w_sr_cpr)
                logits_ref_r_flat = _subsample_ref_logits(logits_ref_r_cache, sub_buffer)

            for (obs_mb, cid_mb, votes_mb, z0_mb, log_pi_old_r_mb,
                 adv_mb, logits_ref_r_mb) in iter_minibatches(
                sub_buffer, adv_expanded, config.minibatch_size, logits_ref_r_flat
            ):
                lr = get_lr(global_step, config)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # Reshape for forward_micpo: [batch, N, ...] → [batch*N, ...]
                batch_size = obs_mb.shape[0]
                N          = obs_mb.shape[1]
                R          = votes_mb.shape[-1]

                votes_flat = votes_mb.reshape(batch_size * N, R)
                z0_flat    = z0_mb.reshape(batch_size * N, 5)

                with ctx:
                    lp_r, lg_r = raw_model.forward_micpo(
                        obs_mb.nan_to_num(0.0),
                        cid_mb,
                        votes_flat,
                        z0_flat,
                    )
                    # lp_r: [batch*N, K]   lg_r: [batch*N, K, 5]
                    lp_r_view = lp_r.view(batch_size, N, R).float().nan_to_num(0.0)
                    lg_r_view = lg_r.view(batch_size, N, R, 5).float().nan_to_num(0.0)

                    # Reference logits for KL
                    if logits_ref_r_mb is None:
                        with torch.no_grad(), ctx:
                            vf  = votes_mb.reshape(batch_size * N, R)
                            z0f = z0_mb.reshape(batch_size * N, 5)
                            _, lg_ref_r = pi_ref.forward_micpo(
                                obs_mb.nan_to_num(0.0), cid_mb, vf, z0f
                            )
                            logits_ref_r = lg_ref_r.view(batch_size, N, R, 5).float().nan_to_num(0.0)
                    else:
                        logits_ref_r = logits_ref_r_mb   # [batch, N, K, 5] cached

                    L_clip = clip_pg_loss(lp_r_view, log_pi_old_r_mb, adv_mb, config.eps_clip)
                    L_kl   = kl_categorical(lg_r_view, logits_ref_r, config.alpha_kl)
                    L_ent  = entropy_bonus(lg_r_view, config.alpha_entropy)
                    loss   = total_loss(L_clip, L_kl, L_ent)

                scaler.scale(loss).backward()

                # DDP: forward_micpo bypasses DDP hooks, so manually sync grads.
                # Always allreduce every parameter (filling None with zeros) so
                # all ranks participate in the same set of collectives — skipping
                # None grads causes rank divergence when environments differ.
                if world_size > 1:
                    import torch.distributed as dist
                    for p in raw_model.parameters():
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(pi_theta.parameters(), config.grad_clip)

                _grads_ok = all(
                    p.grad is None or p.grad.isfinite().all()
                    for p in raw_model.parameters()
                )
                if _grads_ok:
                    scaler.step(optimizer)
                else:
                    log.warning(f"[step {global_step}] NaN/Inf gradients — skipping optimizer step")

                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    ratio = compute_importance_ratio(lp_r_view, log_pi_old_r_mb)
                    epoch_losses.append(loss.item())
                    epoch_clip.append(L_clip.item())
                    epoch_kl.append(L_kl.item())
                    epoch_ent.append(L_ent.item())
                    epoch_r_mean.append(ratio.mean().item())
                    epoch_r_max.append(ratio.max().item())
                    epoch_entropy.append(policy_entropy(lg_r_view).item())

                global_step += 1

        # ------------------------------------------------------------ #
        # 10. Update pi_ref if scheduled
        # ------------------------------------------------------------ #
        if config.ref_update_freq > 0 and (iteration + 1) % config.ref_update_freq == 0:
            pi_ref.load_state_dict(uncompiled_state_dict(raw_model))
            log.info(f"[iter {iteration}] pi_ref updated")

        # ------------------------------------------------------------ #
        # 11. Logging
        # ------------------------------------------------------------ #
        if is_master and iteration % config.logging_freq == 0:
            mean_loss = float(np.mean(epoch_losses))
            mean_clip = float(np.mean(epoch_clip))
            mean_kl   = float(np.mean(epoch_kl))
            mean_ent  = float(np.mean(epoch_ent))
            mean_adv  = float(advantages.mean().item())
            std_adv   = float(advantages.std().item())
            mean_ret  = float(returns.mean().item())
            std_ret   = float(returns.std().item())
            r_mean    = float(np.mean(epoch_r_mean))
            r_max     = float(np.max(epoch_r_max))
            ent = float(np.mean(epoch_entropy))

            isr           = compute_isr(buffer)
            csr           = compute_csr(buffer)
            mean_makespan = compute_mean_makespan(buffer)
            n_blocked     = n_blocked_total

            _C            = compute_cost(buffer.pos_after, buffer.goals, buffer.first_arrival,
                                         buffer.episode_length, config.cost_mode)
            mean_sum_cost = float(_C.sum(dim=-1).mean().item())
            mean_cost     = float(_C.mean().item())

            if r_mean > config.log_ratio_warn_threshold:
                log.warning(
                    f"[step {global_step}] mean importance ratio = {r_mean:.3f} > "
                    f"{config.log_ratio_warn_threshold}"
                )

            scalars = {
                "train/policy_loss":           mean_clip,
                "train/kl_loss":               mean_kl,
                "train/entropy_bonus":         mean_ent,
                "train/total_loss":            mean_loss,
                "train/mean_advantage":        mean_adv,
                "train/std_advantage":         std_adv,
                "train/mean_return":           mean_ret,
                "train/std_return":            std_ret,
                "train/isr":                   isr,
                "train/csr":                   csr,
                "train/mean_makespan":         mean_makespan,
                "train/mean_sum_cost":         mean_sum_cost,
                "train/mean_cost":             mean_cost,
                "train/n_blocked_moves":       n_blocked,
                "train/mean_policy_entropy":   ent,
                "train/entropy":               ent,
                "train/importance_ratio_mean": r_mean,
                "train/importance_ratio_max":  r_max,
                "train/lr":                    lr,
            }

            # Audit raw scale, within-group variability, and effective weighted
            # contribution of every active reward component.  This makes a
            # sparse component visible before it can silently dominate MICPO.
            reward_audit = []
            for name, audit in reward_audit_values.items():
                raw_mean = audit["raw_mean"]
                raw_std = audit["raw_std"]
                norm_std = audit["norm_std"]
                effective_rms = audit["effective_rms"]
                scalars[f"train/reward_raw_{name}_mean"] = raw_mean
                scalars[f"train/reward_raw_{name}_std"] = raw_std
                scalars[f"train/reward_norm_{name}_std"] = norm_std
                scalars[f"train/reward_effective_{name}_rms"] = effective_rms
                reward_audit.append(
                    f"{name}:raw_std={raw_std:.4g},norm_std={norm_std:.3f},rms={effective_rms:.3f}"
                )
            log_scalars(loggers, scalars, global_step)

            t_iter = time.monotonic() - t_iter_start
            audit_summary = (
                f"  reward_audit=[{' | '.join(reward_audit)}]"
                if config.reward_audit else ""
            )
            log.info(
                f"[step {global_step}] return={mean_ret:.3f}  "
                f"isr={isr:.3f}  csr={csr:.3f}  makespan={mean_makespan:.1f}  "
                f"blocked={n_blocked}  "
                f"loss={mean_loss:.4f}  iter={t_iter:.1f}s"
                f"{audit_summary}"
            )

        # ------------------------------------------------------------ #
        # 12. Validation
        # ------------------------------------------------------------ #
        if is_master and (iteration + 1) % config.val_freq_steps == 0:
            # Use uncompiled model for validation — avoids bf16 kernel recompilation
            # on GPUs < sm_80. _orig_mod shares parameters with the compiled model.
            val_model = getattr(raw_model, "_orig_mod", raw_model)
            val_metrics = run_validation(val_model, val_scenarios, config, device)
            log_scalars(loggers, val_metrics, global_step)
            log.info(f"[val step {global_step}] {val_metrics}")
            if val_metrics.get("val/mean_isr", 0.0) > best_val_isr:
                best_val_isr = val_metrics["val/mean_isr"]
            val_step = global_step

        # ------------------------------------------------------------ #
        # 13. Checkpointing
        # ------------------------------------------------------------ #
        completed_iterations = iteration + 1
        save_latest   = should_save_latest(completed_iterations, config.latest_ckpt_freq)
        save_periodic = should_save_periodic(completed_iterations, config.ckpt_freq)
        if save_latest or save_periodic:
            rng_states = gather_rng_states(
                capture_rng_state(iter_rng, device), world_size
            )
            if is_master:
                save_checkpoint(
                    run_dir=run_dir,
                    step=global_step,
                    completed_iterations=completed_iterations,
                    raw_model=raw_model,
                    optimizer=optimizer,
                    scaler=scaler if use_amp else None,
                    config=config,
                    model_args=model_args,
                    best_val_isr=best_val_isr,
                    val_step=val_step,
                    periodic=save_periodic,
                    rng_states=rng_states,
                    pi_ref=pi_ref,
                )
            barrier()

    # ---------------------------------------------------------------- #
    # Final checkpoint
    # ---------------------------------------------------------------- #
    rng_states = gather_rng_states(capture_rng_state(iter_rng, device), world_size)
    if is_master:
        save_checkpoint(
            run_dir=run_dir,
            step=global_step,
            completed_iterations=completed_iterations,
            raw_model=raw_model,
            optimizer=optimizer,
            scaler=scaler if use_amp else None,
            config=config,
            model_args=model_args,
            best_val_isr=best_val_isr,
            val_step=val_step,
            periodic=True,
            rng_states=rng_states,
            pi_ref=pi_ref,
        )
        log.info("Training complete.")
        loggers["tb"].close()
        if loggers.get("wandb"):
            loggers["wandb"].finish()

    barrier()
    cleanup_ddp()


if __name__ == "__main__":
    main()
