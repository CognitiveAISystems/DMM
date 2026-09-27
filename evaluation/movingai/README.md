# MovingAI-1600

[`run.py`](run.py) and [`summarize.py`](summarize.py) implement the standalone
MovingAI experiment for DMM-MICPO-08M and DMM-MICPO-3M. The supported
entry point is `python -m evaluation.run --model DMM-MICPO-08M --benchmark movingai`
(or `--model DMM-MICPO-3M`). It uses the fixed
1,600-task manifest, zero z0, argmax communication rounds, CS-PIBT, RSE16, and a 5,000-step
limit. Each process keeps one model on one visible GPU and claims tasks from a
local queue. The 1,600-row manifest, 32 maps, and 1,600 scenario files are
bundled here alongside the evaluation code.
The archive is checked and unpacked automatically. See the repository
[`README.md`](../../README.md#evaluation) for commands.

The evaluator reproduces the main-figure argmax + RSE mode. The sampled-round
ablations in `raw_results/` are not selectable through this entry point.
