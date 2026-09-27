"""
OneshotBatchedRollout — collect G oneshot MAPF rollouts with DMM.

Each MAPF step calls model.rollout_act() which runs K communication rounds
and returns:
  - final actions  [G*N]      — used to step the environment
  - votes          [G*N, K]   — stored for deterministic replay
  - log_pi_rounds  [G*N, K]   — per-round log-probs (IS ratio at training)
  - z0             [G*N, 5]   — Dirichlet initial state (stored for replay)
  - z              [G*N, 5]   — final EMA state

Subsampling is on the MAPF step axis — all K rounds are kept together.
"""

from __future__ import annotations

import torch

from training.micpo.transition import OneshotBuffer


class OneshotBatchedRollout:
    """
    Manages G oneshot MAPF environments and collects one rollout.

    Args:
        policy:   DMM08MPolicy or DMMPolicy
        envs:     list of G POGEMA-GPU-backed environments (on_target="nothing")
        config:   OneshotMICPOConfig
        device:   torch.device
    """

    def __init__(self, policy, envs, config, device: torch.device,
                 n_z0_groups: int = 1):
        self.policy      = policy
        self.envs        = envs
        self.config      = config
        self.device      = device
        self.n_z0_groups = n_z0_groups

        self.G = len(envs)
        self.N = envs[0].num_agents

        self._init_pos  = [env.pos.clone() for env in envs]
        self._goals     = torch.stack([env.goals for env in envs])   # [G, N, 2]

    @torch.no_grad()
    def collect(self) -> OneshotBuffer:
        """Run up to H steps; return a fully populated OneshotBuffer."""
        cfg     = self.config
        G, N, H = self.G, self.N, cfg.max_horizon
        device  = self.device

        # Reset envs; each POGEMA-GPU tokenizer resets its action history.
        for g, env in enumerate(self.envs):
            env.reset(positions=self._init_pos[g].clone())

        buffer = OneshotBuffer(
            G=G, H=H, N=N,
            n_rounds=cfg.n_comm_rounds,
            context_size=cfg.context_size,
            max_num_neighbors=cfg.max_num_neighbors,
            device=device,
        )

        buffer.goals = self._goals.clone()

        first_arrival  = torch.full((G, N), H, dtype=torch.long,  device=device)
        episode_length = torch.full((G,),   H, dtype=torch.long,  device=device)
        env_done       = torch.zeros(G,         dtype=torch.bool,  device=device)
        arrived_ever   = torch.zeros(G, N,      dtype=torch.bool,  device=device)

        n_blocked_total = 0

        for step in range(H):
            # 1. Generate observations through the shared POGEMA-GPU tokenizer.
            obs_list, chat_ids_list = [], []
            for env in self.envs:
                observations, neighbors = env.observe()
                obs_list.append(observations)
                chat_ids_list.append(neighbors)

            obs_batch      = torch.stack(obs_list)        # [G, N, 256]
            chat_ids_batch = torch.stack(chat_ids_list)   # [G, N, 13]

            # 2. DMM rollout_act: K rounds, stochastic
            # Sample one z0 per agent [N, 5], shared across all G group members
            # — removes z0 luck from within-group advantage variance.
            if getattr(cfg, "rollout_z0_mode", "dirichlet") == "zero":
                # Uniform probabilities become exactly zero after centered-log
                # conversion inside either DMM architecture.
                z0_in = torch.full((G * N, 5), 0.2, device=device)
            else:
                dir_alpha   = getattr(cfg, "dir_alpha", 1.0)
                M           = self.n_z0_groups
                G_per_group = G // M
                z0_shared = torch.distributions.Dirichlet(
                    torch.full((M, N, 5), dir_alpha, device=device)
                ).sample()
                z0_in = z0_shared.unsqueeze(1).expand(
                    M, G_per_group, N, 5
                ).reshape(G * N, 5)

            # actions:       [G*N]
            # votes:         [G*N, K]
            # log_pi_rounds: [G*N, K]
            # z0:            [G*N, 5]
            amp_mode = cfg.rollout_amp
            if amp_mode not in {"none", "auto", "bf16"}:
                raise ValueError(f"Unknown rollout_amp: {amp_mode!r}")
            use_bf16 = device.type == "cuda" and (
                amp_mode == "bf16" or (amp_mode == "auto" and cfg.use_fp16)
            )
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                actions_flat, votes_flat, log_pi_flat, z0_flat, _ = \
                    self.policy.rollout_act(
                        obs_batch,
                        chat_ids_batch,
                        rollout_tau=getattr(cfg, "rollout_tau", 1.0),
                        z0_in=z0_in,
                    )

            # Reshape to [G, N, ...]
            K              = cfg.n_comm_rounds
            actions        = actions_flat.view(G, N)
            votes          = votes_flat.view(G, N, K)
            log_pi_rounds  = log_pi_flat.view(G, N, K)
            z0             = z0_flat.view(G, N, 5)

            # 4. epsilon-greedy mutation (on the final action only)
            epsilon = getattr(cfg, "rollout_epsilon", 0.0)
            if epsilon > 0.0:
                mutation_mask  = torch.rand(G, N, device=device) < epsilon
                random_actions = torch.randint(0, 5, (G, N), device=device)
                actions = torch.where(mutation_mask, random_actions, actions)

            # 5. Positions before step
            pos_before = torch.stack([env.pos.clone() for env in self.envs])   # [G, N, 2]

            # 6. Step environments
            for g, env in enumerate(self.envs):
                if not env_done[g]:
                    env.step(actions[g])
                else:
                    env.remember(actions[g])

            pos_after = torch.stack([env.pos for env in self.envs])   # [G, N, 2]

            # 7. Detect blocked moves
            blocked = (actions != 0) & (pos_after == pos_before).all(-1)
            n_blocked_total += int((blocked & ~env_done.unsqueeze(-1)).sum().item())

            # 8. Track first arrival
            on_goal      = (pos_after == self._goals).all(dim=-1)
            new_arrivals = on_goal & ~arrived_ever
            arrived_ever |= new_arrivals
            first_arrival = torch.where(
                new_arrivals,
                torch.full_like(first_arrival, step),
                first_arrival,
            )

            # 9. Store in buffer
            buffer.store_step(
                step=step,
                obs_batch=obs_batch,
                chat_ids_batch=chat_ids_batch,
                actions=actions,
                log_pi_rounds=log_pi_rounds,
                votes=votes,
                z0=z0,
                pos_before=pos_before,
                pos_after=pos_after,
            )

            # 10. Episode termination
            all_on_goal = on_goal.all(dim=-1)
            newly_done  = all_on_goal & ~env_done
            env_done   |= newly_done
            episode_length = torch.where(
                newly_done,
                torch.full_like(episode_length, step + 1),
                episode_length,
            )

            if env_done.all():
                break

        buffer.first_arrival   = first_arrival
        buffer.episode_length  = episode_length
        buffer.n_blocked_total = n_blocked_total
        return buffer


# ------------------------------------------------------------------ #
# Factory
# ------------------------------------------------------------------ #

def build_rollout(
    policy,
    env_instances,
    config,
    device: torch.device,
    n_z0_groups: int = 1,
) -> OneshotBatchedRollout:
    """
    Construct POGEMA-GPU environments with their shared observation tokenizer;
    return OneshotBatchedRollout ready for .collect().
    """
    from training.micpo.environment import MICPOEnvironment
    envs     = []

    for inst in env_instances:
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

    return OneshotBatchedRollout(
        policy=policy,
        envs=envs,
        config=config,
        device=device,
        n_z0_groups=n_z0_groups,
    )
