# Intent-Communication Ablation

This directory contains the raw per-instance results and plotting code for the
intent-communication figure. Fixed `DMM-3M` and `DMM-MICPO-3M` checkpoints are
evaluated on Mazes at the trained depth of four refinement rounds, under
inference-time perturbations of what each agent broadcasts. The weights and each
agent's own intent update are unchanged; only the transmitted message differs.

## Variants

- `full` — the standard message: the learned feature `h` and the evolving intent `z`;
- `no-h` — broadcast `z` only;
- `no-z` — broadcast `h` only;
- `z0-only` — broadcast the initial intent at every round while the agent's own
  intent still evolves normally;
- `shuffled` — permute assembled messages across agents by a derangement, keeping
  message content but breaking its correspondence to the receiving agent.

`full` is the unablated policy at the trained depth, so it is the same
configuration as `K4` in
[`../04_round_depth_ablation/`](../04_round_depth_ablation/).

## Data layout

`data/02-mazes/<model>-<variant>.json` holds one record per episode in the same
envelope as the other raw experiment files: 896 per file, 128 for each of the
seven agent counts from 8 to 80. No episodes are excluded.

The figure reports success rate with 95% Wilson score intervals, one panel per
model, across team sizes.

## Rebuild the figure

From this directory, run:

```bash
uv run --no-project --python 3.11 --with-requirements requirements.txt \
  python scripts/plot_intent_ablation.py
```

The script checks that every variant contributes 128 episodes per agent count
and writes `figure/intent_ablation.pdf`. The pinned Matplotlib version fixes the
figure geometry the paper crops.
