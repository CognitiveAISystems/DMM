#!/usr/bin/env bash
# The corridor comparison: DMM with and without the teacher-forcing floor, against
# LC-MAPF and MAGAT+, five independently trained seeds each.
set -euo pipefail
cd "$(dirname "$0")"

SEEDS="${SEEDS:-0 1 2 3 4}"
ITERS="${ITERS:-5000}"
TRAIN_DMM="python -m training.pretrain ablations/corridor/configs/corridor.py"

for seed in $SEEDS; do
    (cd ../.. && $TRAIN_DMM --seed="$seed" --max_iters="$ITERS" \
        --exp_dir_name="corridor_tf08_seed${seed}")
    (cd ../.. && $TRAIN_DMM --seed="$seed" --max_iters="$ITERS" \
        --exp_dir_name="corridor_no_tf_seed${seed}" \
        --dirichlet_tf_beta_final=0.0 --round_tf_beta_final=0.0)
    python train_lcmapf.py --seed="$seed" --iterations="$ITERS" \
        --out-dir="runs/lcmapf_seed${seed}"
    python train_magat.py --seed="$seed" --iterations="$ITERS" \
        --out-dir="runs/magat_seed${seed}"
    if [ -d third_party/upstream_hmagat ]; then
        python train_hmagat.py --seed="$seed" --iterations="$ITERS" \
            --out-dir="runs/hmagat_seed${seed}"
    fi
done

# HMAGAT joins the table only when setup.sh fetched it.
hmagat_column=()
if [ -d third_party/upstream_hmagat ]; then
    hmagat_column=(--hmagat "HMAGAT=runs/hmagat_seed*/ckpt_latest.pt")
fi

python eval/frequencies.py \
    --dmm "DMM (tf=0.8)=runs/corridor_tf08_seed*/ckpt_latest.pt" \
    --dmm "DMM (tf=0)=runs/corridor_no_tf_seed*/ckpt_latest.pt" \
    --lcmapf "LC-MAPF=runs/lcmapf_seed*/ckpt_latest.pt" \
    --magat "MAGAT+=runs/magat_seed*/ckpt_latest.pt" \
    ${hmagat_column[@]+"${hmagat_column[@]}"} \
    --out results/baseline.json
