#!/usr/bin/env bash
# Train at two, four and eight refinement rounds and evaluate every trained depth
# at every test depth. The four-round row is the standard configuration, and the
# figure and both tables read that cell as their single reference estimate.
set -euo pipefail
cd "$(dirname "$0")"

SEEDS="${SEEDS:-0 1 2 3 4}"
ITERS="${ITERS:-5000}"

for rounds in 2 8; do
    for seed in $SEEDS; do
        (cd ../.. && python -m training.pretrain ablations/corridor/configs/corridor.py \
            --seed="$seed" --max_iters="$ITERS" --n_comm_rounds="$rounds" \
            --exp_dir_name="corridor_rounds${rounds}_seed${seed}")
    done
done

mkdir -p results

# The four-round checkpoints come from run_baselines.sh.
python eval/round_matrix.py \
    --ckpt "2=runs/corridor_rounds2_seed*/ckpt_latest.pt" \
    --ckpt "4=runs/corridor_tf08_seed*/ckpt_latest.pt" \
    --ckpt "8=runs/corridor_rounds8_seed*/ckpt_latest.pt" \
    --test-rounds 2,4,8,12 \
    --out results/round_generalization_matrix.json
