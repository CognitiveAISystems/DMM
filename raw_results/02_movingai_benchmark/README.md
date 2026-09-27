# MovingAI Benchmark

This directory contains the per-task data used for the MovingAI benchmark
figure. The benchmark has 32 maps and 50 scenarios per map (25 `even` and 25
`random`), for 1,600 tasks per method. The `maze-128-128-1` map is excluded.

## Data

The `data` directory contains one JSON list per method:

- `LG-LaCAM.json`
- `LaGAT.json`
- `MAPF-LNS2.json`
- `HMAGAT.json` (probabilistic sampling, PIBT + RSE16)
- `DMM-MICPO-08M.json` (argmax communication rounds, PIBT + RSE16)
- `DMM-MICPO-3M.json` (argmax communication rounds, PIBT + RSE16)

Every list has 1,600 records in the same envelope as the other raw experiment
files in this repository:

```json
{
  "metrics": {
    "CSR": 1.0,
    "ep_length": 479,
    "SoC": 215072,
    "makespan": 479,
    "runtime": 12.2583
  },
  "env_grid_search": {
    "num_agents": 950,
    "map_name": "Berlin_1_256",
    "scenario_type": "even",
    "scenario_id": 10,
    "task_key": "Berlin_1_256|even|10|950"
  },
  "algorithm": "DMM-MICPO-3M"
}
```

`runtime` is the solver-reported runtime in seconds rather than worker/process
wall time. Unsolved tasks remain in the files with `CSR = 0.0` and null
solution metrics. This makes coverage directly reproducible and keeps all
methods aligned on exactly the same logical task set.

The values relative to the virtual best are intentionally not stored:
they are derived from the six per-task SoC values when constructing the
figure.

The main HMAGAT file has 1,576 solved tasks out of 1,600.

`movingai_extra/` contains the other decoding/RSE configurations: HMAGAT
sampling without RSE and argmax with RSE, plus DMM-08M and DMM-3M sampled
rounds with and without RSE. With the main-figure files, there are three
configurations per learned method. These extra files are not inputs to the
main six-method figure. See their README for coverage and timing semantics.

## Rebuild the figure

From this directory, run:

```bash
uv run --no-project --python 3.11 --with-requirements requirements.txt \
  python scripts/plot_movingai.py
```

The script reads the six JSON lists directly. It validates their task sets,
computes the per-task virtual-best SoC, coverage, virtual-best wins, quantiles,
runtime medians, and runtime KDEs, and writes
`figure/mapf-soc-runtime-movingai1600.pdf`.
