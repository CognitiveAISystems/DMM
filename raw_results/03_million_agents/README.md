# Million-agent scalability results

This package contains the results behind the million-agent table and figure in
the paper. DMM-MICPO-0.8M and GPU-PIBT were evaluated on four matched
2304×2304 mazes (seeds 101, 202, 303, 404) at each of 131,072, 262,144,
524,288, and 1,048,576 agents. The horizon was 32,768 steps.

- `data/dmm_08m/` and `data/gpu_pibt/`: the 32 per-run result records as
  compressed JSON. Timing arrays and numerical metrics are retained; machine
  paths and other identifying provenance fields are omitted.
- `data/published_table.csv`: the rounded values printed in the paper.
- `data/seed303_spatial_fine.json.xz`: the spatial aggregates for the
  16,345-step rollout shown in the figure.
- `scripts/plotting.py`: the plotting implementation used to construct the
  spatial and timing panels; `scripts/render.py` renders them from the
  included data.
- `figure/05_million_agent_rollout.pdf`: the final paper figure. Its inset and
  callout were composed on top of the data-driven panels.

Recompute the table directly from the run records with:

```bash
python3 scripts/summarize.py
```

To render the underlying figure panels:

```bash
uv venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/render.py --output figure/seed303_reproduced.pdf
```
