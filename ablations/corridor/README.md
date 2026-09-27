# Corridor conflict

Two agents in a one-wide corridor with a single passing bay. Two expert
trajectories solve it — either agent can yield — and they share one state, where
that choice is made. Each model is trained from scratch on those two
trajectories by imitation, then sampled 1,000 times at the shared state to
measure how often the two agents produce a coordinated joint action.

`scenario.py` defines the grid and both trajectories; every model encodes the
same states through its own observation pipeline.

## Setup

```bash
CUDA_TAG=cu126 ./setup.sh
```

Installs the extra dependencies and writes the datasets: the tokenizer shared by
DMM and LC-MAPF (`data/part_0_0.arrow`, 14 rows) and the MAGAT+ graphs. The
tokenizer compiles from source, so a C++ toolchain is required.

HMAGAT builds its observations and hypergraphs through its own pipeline, which
reaches into the rest of its benchmark suite, so it is fetched instead of
vendored: `WITH_HMAGAT=1 ./setup.sh` clones it at a pinned commit and writes
`data/hmagat_graphs.pt`. Without it the other three models are unaffected and
`run_baselines.sh` omits that column. See
[`third_party/PROVENANCE.md`](third_party/PROVENANCE.md) for upstream commits
and licences.

## Training

DMM trains through the repository's pretraining loop, so the corridor is a
configuration rather than a separate trainer, and each ablation is a
command-line override on it:

```bash
cd ../..
python -m training.pretrain ablations/corridor/configs/corridor.py \
    --seed=0 --exp_dir_name=corridor_tf08_seed0
```

The baselines have their own full-batch trainers: `train_lcmapf.py`,
`train_magat.py` and `train_hmagat.py`.

Every reported number is a mean across five independently trained seeds, so a
single run cannot separate a property of the mechanism from one initialisation.
The sweep scripts run those seeds and write the result files:

| Script | Produces |
| --- | --- |
| `run_baselines.sh` | `results/baseline.json` |
| `run_tf_sweep.sh` | `results/ablation_{1,2,3,4}_*.json` |
| `run_round_sweep.sh` | `results/round_generalization_matrix.json` |

Those match the published data in
[`../../raw_results/06_corridor_ablations/`](../../raw_results/06_corridor_ablations/).
The standard configuration — both mechanisms, floor 0.8, four rounds — is
trained once by `run_baselines.sh`, and the round sweep reuses those
checkpoints, so every panel and table reports it from one estimate.

## Evaluation

`eval/frequencies.py` takes one column per configuration, each a glob matching
one checkpoint per seed:

```bash
python eval/frequencies.py \
    --dmm "DMM (tf=0.8)=runs/corridor_tf08_seed*/ckpt_latest.pt" \
    --lcmapf "LC-MAPF=runs/lcmapf_seed*/ckpt_latest.pt" \
    --n 1000 --out results/baseline.json
```

It reports the four joint-action frequencies and their valid sum, each a mean
with a 95% Student-t interval across seeds. `eval/round_matrix.py` does the same
across trained and executed depths, which is valid because the refinement rounds
share weights.
