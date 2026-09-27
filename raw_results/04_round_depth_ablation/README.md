# Refinement-Depth Ablation

This directory contains the raw per-instance results and plotting code for the
inference-time refinement-depth figure. `DMM-3M` and `DMM-MICPO-3M` are both
trained with four refinement rounds and executed here at
`K_test` in {1, 2, 3, 4, 8, 12} without retraining, using the bundled
checkpoints and the POGEMA protocol: Dirichlet initial intent, sampled
communication rounds, no collision shielding.

## Data layout

`data/<family>/<model>-K<k>.json` holds one record per episode, in the same
envelope as the other raw experiment files in this repository. There are four
families, six depths and two models, for 48 files:

- `01-random`: 1,024 episodes, 8 agent counts up to 96;
- `02-mazes`: 896 episodes, 7 agent counts up to 80;
- `03-warehouse`: 768 episodes, 6 agent counts up to 192;
- `04-movingai`: 512 episodes, 4 agent counts up to 256.

Every agent count has 128 episodes. Unlike the main POGEMA figure no episodes
are excluded, so the seven known-infeasible ones count as failures identically
for every configuration and all depths stay on one task set.

The figure reports the largest team size per family — Random 96, Mazes 80,
Warehouse 192, Cities Tiles 256. Rows are success rate, makespan and
agent-agent collisions; success rate uses 95% Wilson score intervals, the other
two normal intervals. Unsolved episodes are counted at the evaluation horizon
rather than excluded, so the makespan row includes episodes that only reach
success at greater depth.

## Rebuild the figure

From this directory, run:

```bash
uv run --no-project --python 3.11 --with-requirements requirements.txt \
  python scripts/plot_round_depth.py
```

The script checks that every depth contributes exactly 128 episodes at the
reported team size and writes `figure/round_depth_ablation.pdf`. The pinned
Matplotlib version fixes the figure geometry the paper crops.
