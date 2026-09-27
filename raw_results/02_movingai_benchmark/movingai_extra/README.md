# Additional MovingAI results

These are the additional decoding/RSE results on the same 1,600 MovingAI
tasks as the main figure. They are not inputs to `scripts/plot_movingai.py`;
that figure uses the sampling + RSE HMAGAT and argmax + RSE DMM files in
`../data/`.

| File | Mode | Solved |
| --- | --- | ---: |
| `HMAGAT.json` | HMAGAT sampling, no RSE | 1,556 |
| `HMAGAT-argmax-rse.json` | HMAGAT argmax + RSE16 | 1,543 |
| `DMM-MICPO-08M.json` | DMM-08M sampled rounds, no RSE | 1,593 |
| `DMM-MICPO-08M-sampling-rse.json` | DMM-08M sampled rounds + RSE16 | 1,592 |
| `DMM-MICPO-3M-G24.json` | DMM-3M sampled rounds, no RSE | 1,599 |
| `DMM-MICPO-3M-G24-sampling-rse.json` | DMM-3M sampled rounds + RSE16 | 1,599 |

Together with the three main-figure files, these provide sampling, sampling
+ RSE, and argmax + RSE for each learned method. All nine files contain 1,600
unique task keys; 1,526 tasks are solved by all nine configurations.

The sampled-round DMM runs with RSE use shared dynamic GPU batches. Their
per-task solver `runtime` is null; `worker_wall_time`, when present, is a
separate worker-observed quantity and must not be substituted for solver
runtime or total evaluator wall time. The no-RSE DMM-08M results also have
null per-task solver runtime.
