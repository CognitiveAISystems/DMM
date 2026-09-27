# POGEMA Benchmark

This directory contains the raw per-instance results and plotting code for the
combined POGEMA benchmark figure.

The evaluation consists of four map families:

- `01-random`: 1,020 valid tasks;
- `02-mazes`: 896 valid tasks;
- `03-warehouse`: 768 valid tasks;
- `04-movingai`: 509 valid tasks.

Together they form the 3,193-task evaluation set used in the figure.

## Data layout

Each `data/<split>` directory contains results for the nine displayed methods:

- `MAPF-GPT-85M`
- `MAPF-GPT-DDG-2M`
- `MAGAT+`
- `HMAGAT`
- `LC-MAPF-3M`
- `DMM-08M`
- `DMM-3M`
- `DMM-MICPO-08M`
- `DMM-MICPO-3M`

It also contains `LaCAM.json`, which supplies the reference SoC for the lower
row and is not displayed as a separate method.

The upper row reports CSR as a function of the number of agents, with 95%
confidence intervals. The lower row reports the SoC ratio relative to LaCAM on
the common instance set. The canonical valid task set is the 3,193 instances
present in the `DMM-MICPO-08M` logs; every method is filtered to those logical
instances before plotting.

## Rebuild the figure

From this directory, run:

```bash
uv run --no-project --python 3.11 --with-requirements requirements.txt \
  python scripts/plot_pogema.py
```

The script validates the raw JSON structure, rejects duplicate logical tasks,
checks the expected valid-task count for every split, and writes
`figure/CSR-SOC-dmmv2-combined.pdf`.
