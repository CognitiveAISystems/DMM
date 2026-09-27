"""
OneshotMICPOConfig — hyperparameters shared by DMM08M and DMM MICPO training.

Episode structure:
  - on_target="nothing": agents stay at goal, may step aside cooperatively.
  - Episode ends when all N agents are simultaneously on their goal,
    or when max_horizon H steps are reached (whichever comes first).
  - Trajectory-wise (outcome) rewards; process_supervision toggle kept
    for experimentation.

Reward schema:
  C[n] — per-agent cost (goal_reached: steps to first arrival;
                         on_goal:       steps not on goal)

  R[n] = − w_makespan  ·  T_ep
         − w_cost      · [ (1−α_cost)    · C[n]       +  α_cost    · mean_n(C[n]) ]
         − w_blocked   · [ (1−α_blocked) · B[n]       +  α_blocked · mean_n(B[n]) ]
         + w_sr_makespan · SR_team
         + w_sr_cost   · [ (1−α_sr)      · SR_ind[n]  +  α_sr      · mean_n(SR_ind) ]

  T_ep   = episode length = makespan (all on goal) or H if unsolved.
  B[n]   = total blocked steps for agent n.
  SR_ind = Σ_h log_π(a[h,n]) · mask[h,n]            (sr_complement=False)
         = Σ_h log(1-π(a[h,n])) · mask[h,n]         (sr_complement=True)
         where log_π is the SUM of per-round log-probs for that MAPF step.
  SR_team = Σ_h agg_n(SR_signal[h,n]) over active agents
  mask follows cost_mode.
"""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class OneshotMICPOConfig:
    # ------------------------------------------------------------------ #
    # Run / I/O
    # ------------------------------------------------------------------ #
    run_name: str = "run"
    runs_dir: str = "runs"
    init_from: str = "pretrained"   # "scratch" | "pretrained" | "resume"
    resume_run_dir: Optional[str] = None
    path_to_weights: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Policy / model
    # ------------------------------------------------------------------ #
    policy_class: str = ""  # must be explicit: "dmm08m" | "dmm"
    device: str = "cuda"
    use_fp16: bool = True
    use_compile: bool = False
    freeze_encoder: bool = False

    # Architecture (read from checkpoint; override only for scratch)
    n_comm_rounds: int = 4
    n_encoder_layer: int = 3
    n_decoder_layer: int = 3
    n_head: int = 3
    n_embd: int = 192
    latent_embd: int = 96
    latent_tok_n: int = 32
    action_msg_feats: int = 96
    dropout: float = 0.0
    bias: bool = False

    # Model dynamics
    dt: float = 0.25
    tau: float = 1.0

    # Observation generator (must match weights)
    num_previous_actions: int = 5
    cost2go_value_limit: int = 20
    agents_radius: int = 5
    cost2go_radius: int = 5
    context_size: int = 256
    max_num_neighbors: int = 13

    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #
    num_agents: int = 32
    map_height: int = 32
    map_width: int = 32
    map_height_min: Optional[int] = None
    map_height_max: Optional[int] = None
    map_width_min:  Optional[int] = None
    map_width_max:  Optional[int] = None
    max_horizon: int = 256          # episode cap; T_ep = min(makespan, max_horizon)

    # Map / scenario mode
    env_mode: str = "random"        # "random"|"fixed_map"|"fixed_map_scenario"|"custom_map"|"custom_map_scenario"
    map_type: str = "random"        # "maze"|"random"|"house"|"mixed"|"mixed3"
    maze_fraction: float = 0.5
    room_fraction: float = 0.0
    obstacle_density: float = 0.2
    map_seed: Optional[int] = None
    scenario_seed: Optional[int] = None
    custom_map_path: Optional[str] = None

    maze_wall_components_min: int = 4
    maze_wall_components_max: int = 8

    # ------------------------------------------------------------------ #
    # MICPO
    # ------------------------------------------------------------------ #
    B: int = 1                      # distinct scenarios per iteration
    G: int = 8                      # group size: trajectories per z0 group
    z0_groups: int = 1             # number of distinct z0s per scenario (M); total groups = B*M
    n_epochs: int = 4
    minibatch_size: int = 0         # 0 = full buffer
    eps_clip: float = 0.2
    alpha_kl: float = 0.05
    alpha_entropy: float = 0.0
    rollout_tau: float = 1.0   # sampling temperature for rollout votes (>1 increases diversity)
    rollout_amp: str = "none"  # "none" | "auto" | "bf16"
    dir_alpha: float = 1.0     # Dirichlet concentration for z0: 1=uniform, >1=tighter near center
    rollout_z0_mode: str = "dirichlet"  # "dirichlet" | "zero"
    ref_update_freq: int = 0        # 0 = never
    timestep_sample_ratio: float = 0.5
    timestep_sample_k: int = 0          # if > 0, use directly; else k = ceil(H * ratio)
    resample_per_epoch: bool = True
    process_supervision: bool = False   # per-timestep RTG advantages; toggle kept for experiments
    reward_mix_post_norm: bool = False  # True → normalise each reward component within group, then mix with w_*

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #
    cost_mode: str = "on_goal"

    w_makespan: float = 1.0
    makespan_transform: str = "none"   # "none" | "log" | "sqrt"
    cost_transform: str = "none"

    w_cost: float = 0.0
    alpha_cost: float = 0.0

    w_blocked: float = 0.0
    alpha_blocked: float = 0.0

    w_move: float = 0.0
    alpha_move: float = 0.0

    w_sr_makespan: float = 0.0
    w_sr_cost: float = 0.0
    alpha_sr: float = 0.0
    selfreward_source: str = "rollout"   # "rollout" | "reference"
    sr_team_mode: str = "mean"
    sr_complement: bool = False

    w_sr_cpr: float = 0.0
    alpha_cpr: float = 0.0
    cpr_normalize_std: bool = True

    # ------------------------------------------------------------------ #
    # Optimizer
    # ------------------------------------------------------------------ #
    lr: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 200
    lr_decay_iters: int = 10000
    min_lr: float = 6e-5

    # ------------------------------------------------------------------ #
    # Training schedule
    # ------------------------------------------------------------------ #
    n_iters: int = 1000

    # ------------------------------------------------------------------ #
    # Exploration
    # ------------------------------------------------------------------ #
    rollout_epsilon: float = 0.0
    top_bottom_k: int = 0

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    latest_ckpt_freq: int = 50
    ckpt_freq: int = 500

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    val_freq_steps: int = 500
    n_val_scenarios: int = 64
    val_seed: int = 42
    val_save_path: str = "val_scenarios.pt"
    val_batch_size: int = 1
    val_auto_batch_on_oom: bool = True
    val_greedy: bool = True
    val_deterministic_rounds: bool = False  # z0=0 and argmax at every round
    val_num_agents: Optional[int] = None
    val_inference_seed: int = 0
    val_deterministic_inference: bool = False

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    logging_freq: int = 10
    log_ratio_warn_threshold: float = 2.0
    reward_audit: bool = False

    # ------------------------------------------------------------------ #
    # DDP
    # ------------------------------------------------------------------ #
    backend: str = "nccl"

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OneshotMICPOConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str) -> "OneshotMICPOConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        if not isinstance(d, dict):
            raise ValueError(f"MICPO config must be a mapping: {path}")
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown MICPO config keys: {sorted(unknown)}")
        return cls.from_dict(d)

    def save_yaml(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=True)

    def apply_overrides(self, overrides: list[str]) -> "OneshotMICPOConfig":
        """Apply --key=value CLI overrides. Returns a new config instance."""
        d = self.to_dict()
        for arg in overrides:
            if not arg.startswith("--") or "=" not in arg:
                raise ValueError(f"Override must be --key=value, got: {arg!r}")
            key, val_str = arg[2:].split("=", 1)
            if key not in d:
                raise ValueError(f"Unknown config key: {key!r}")
            try:
                val = ast.literal_eval(val_str)
            except (ValueError, SyntaxError):
                val = val_str
            orig = d[key]
            if orig is not None:
                try:
                    val = type(orig)(val)
                except (TypeError, ValueError):
                    pass
            d[key] = val
        return OneshotMICPOConfig.from_dict(d)
