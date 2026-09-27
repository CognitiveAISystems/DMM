#!/usr/bin/env bash
# The four teacher-forcing ablations: which mechanism matters, how much floor is
# needed, whether the mechanism must be active from the start, and whether the
# annealing schedule itself matters. Five seeds per configuration.
#
# The standard configuration (both mechanisms, floor 0.8) is not retrained here:
# every panel and table reads it from the round-generalization sweep, so that one
# configuration is reported from a single estimate. Run run_round_sweep.sh too.
set -euo pipefail
cd "$(dirname "$0")"

SEEDS="${SEEDS:-0 1 2 3 4}"
ITERS="${ITERS:-5000}"

# name:overrides — the teacher-forcing settings that differ from configs/corridor.py
VARIANTS=(
    "no_tf:--dirichlet_tf_beta_final=0.0 --round_tf_beta_final=0.0"
    "dironly:--round_tf_beta_final=0.0"
    "roundonly:--dirichlet_tf_beta_final=0.0"
    "tf02:--dirichlet_tf_beta_final=0.2 --round_tf_beta_final=0.2"
    "tf04:--dirichlet_tf_beta_final=0.4 --round_tf_beta_final=0.4"
    "tf06:--dirichlet_tf_beta_final=0.6 --round_tf_beta_final=0.6"
    "tf10:--dirichlet_tf_beta_final=1.0 --round_tf_beta_final=1.0"
    "flag_bothoff:--dirichlet_tf_on=False --dirichlet_tf_beta_final=0.0 --round_tf_on=False --round_tf_beta_final=0.0"
    "flag_dironly:--round_tf_on=False --round_tf_beta_final=0.0"
    "flag_roundonly:--dirichlet_tf_on=False --dirichlet_tf_beta_final=0.0"
    "const08:--dirichlet_tf_beta=0.8 --round_tf_beta=0.8"
    "const04:--dirichlet_tf_beta=0.4 --dirichlet_tf_beta_final=0.4 --round_tf_beta=0.4 --round_tf_beta_final=0.4"
)

for variant in "${VARIANTS[@]}"; do
    name="${variant%%:*}"
    overrides="${variant#*:}"
    for seed in $SEEDS; do
        # shellcheck disable=SC2086
        (cd ../.. && python -m training.pretrain ablations/corridor/configs/corridor.py \
            --seed="$seed" --max_iters="$ITERS" \
            --exp_dir_name="corridor_${name}_seed${seed}" $overrides)
    done
done

mkdir -p results

python eval/frequencies.py \
    --dmm "neither (tf=0)=runs/corridor_no_tf_seed*/ckpt_latest.pt" \
    --dmm "dirichlet-only=runs/corridor_dironly_seed*/ckpt_latest.pt" \
    --dmm "round-only=runs/corridor_roundonly_seed*/ckpt_latest.pt" \
    --out results/ablation_1_mechanism_isolation.json

python eval/frequencies.py \
    --dmm "0.0=runs/corridor_no_tf_seed*/ckpt_latest.pt" \
    --dmm "0.2=runs/corridor_tf02_seed*/ckpt_latest.pt" \
    --dmm "0.4=runs/corridor_tf04_seed*/ckpt_latest.pt" \
    --dmm "0.6=runs/corridor_tf06_seed*/ckpt_latest.pt" \
    --dmm "1.0=runs/corridor_tf10_seed*/ckpt_latest.pt" \
    --out results/ablation_2_floor_sweep.json

python eval/frequencies.py \
    --dmm "both off=runs/corridor_flag_bothoff_seed*/ckpt_latest.pt" \
    --dmm "dirichlet on=runs/corridor_flag_dironly_seed*/ckpt_latest.pt" \
    --dmm "round on=runs/corridor_flag_roundonly_seed*/ckpt_latest.pt" \
    --out results/ablation_3_onoff.json

python eval/frequencies.py \
    --dmm "constant 0.8=runs/corridor_const08_seed*/ckpt_latest.pt" \
    --dmm "annealed to 0.4=runs/corridor_tf04_seed*/ckpt_latest.pt" \
    --dmm "constant 0.4=runs/corridor_const04_seed*/ckpt_latest.pt" \
    --out results/ablation_4_anneal_vs_constant.json
