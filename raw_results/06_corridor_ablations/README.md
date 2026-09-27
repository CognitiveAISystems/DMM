# Corridor Ablations

Corridor joint-action frequencies, the refinement-depth and teacher-forcing
sweeps built on them, and the code that renders the figure and both tables. The
experiment itself is in [`../../ablations/corridor/`](../../ablations/corridor/).

In the corridor two expert solutions exist and either agent may yield at the
ambiguous state. Every configuration is trained from scratch with five seeds and
sampled 1,000 times per seed. The reported quantity is the valid joint-action
frequency `p_valid = p(wait, left) + p(right, wait)`, the mass on the two
coordinated resolutions; `0.5` is what two independent marginals would give.

`p_valid` is summed per seed first, and the mean and 95% Student-t interval are
taken across the five per-seed values. Adding the separately reported per-action
intervals would be wrong, since the sum of two correlated intervals is not the
interval of the sum.

## Data

- `round_generalization_matrix.json` — three training depths against four test depths.
- `baseline.json` — DMM without the teacher-forcing floor, LC-MAPF, MAGAT+, HMAGAT.
- `ablation_1_mechanism_isolation.json` — each mechanism alone, the excluded one
  annealed to a zero floor.
- `ablation_3_onoff.json` — the same isolation with the excluded mechanism
  disabled from the start of training.
- `ablation_2_floor_sweep.json` — the shared floor swept over its range.
- `ablation_4_anneal_vs_constant.json` — annealed against constant schedules.

The standard configuration — both mechanisms, floor 0.8, four rounds — appears
in every panel and both tables, and always resolves to
`round_generalization_matrix.json`, cell `train=4/test=4`, the same estimate the
paper's opening figure uses. The sweep files therefore carry only the
configurations that differ from it.

## Rebuild

```bash
python scripts/generate_corridor_tables.py

uv run --no-project --python 3.11 --with-requirements requirements.txt \
  python scripts/plot_corridor_ablations.py
```

The table script needs only the standard library and writes
`tables/corridor_freq.json` and `tables/corridor_depth_composition.json`, so the
published table numbers trace back to this directory. The plot script writes
`figure/corridor_ablations.pdf`: refinement depth, which teacher-forcing
mechanism matters, the floor sweep, and the schedule comparison. The pinned
Matplotlib version fixes the geometry the paper crops.
