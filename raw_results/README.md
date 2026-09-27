# Raw experiment results and figure sources

This directory contains the data, plotting scripts and rendered figures used
for the paper:

- `01_pogema_benchmark/`: per-instance JSON for the 3,193-task POGEMA
  comparison and `scripts/plot_pogema.py`.
- `02_movingai_benchmark/`: per-task JSON for the 1,600-task MovingAI
  comparison, additional sampling baselines, and `scripts/plot_movingai.py`.
- `03_million_agents/`: 32 run records, the published scalability table,
  spatial figure data, and its plotting scripts.
- `04_round_depth_ablation/`: per-episode JSON for the six inference-time
  refinement depths on four POGEMA families, and `scripts/plot_round_depth.py`.
- `05_intent_ablation/`: per-episode JSON for the broadcast perturbations on
  Mazes, and `scripts/plot_intent_ablation.py`.
- `06_corridor_ablations/`: corridor joint-action frequencies, the depth and
  teacher-forcing sweeps, both LaTeX tables and their scripts.

Each package has its own README and plotting requirements. Figure scripts read
the files in this directory.
