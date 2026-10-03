# POGEMA benchmark

The runner accepts `DMM-08M`, `DMM-3M`, `DMM-MICPO-08M`, and
`DMM-MICPO-3M`. It evaluates the same 3,193 episodes as the paper's POGEMA
figure: 1,020 random, 896 mazes, 768 warehouse, and 509 MovingAI-family
episodes. This experiment is **unshielded**: neither CS-PIBT nor RSE is used.
The horizons are 128 steps for the first three families and 256 for the
MovingAI family.

The 3,200-episode task pool contains **seven known infeasible episodes**, which
are excluded from evaluation rather than counted as model failures. The
exclusions are listed in
[`excluded-infeasible.tsv`](excluded-infeasible.tsv):

- `01-random_0775_N80`
- `01-random_0903_N96`
- `01-random_0909_N96`
- `01-random_0914_N96`
- `04-movingai_0144_N128`
- `04-movingai_0272_N192`
- `04-movingai_0400_N256`

The map/scenario archive and manifest define all 3,200 task keys. The loader
verifies these keys, the seven exclusions, and per-family counts before
constructing the 3,193 POGEMA-GPU tasks.
The map/scenario archive is unpacked automatically into the local output
cache; no random map generation happens during evaluation.

Run `python -m evaluation.run --model DMM-08M --benchmark pogema` (or another
listed model) after compiling its package. Each invocation uses one visible
GPU. Multiple invocations on one host share the local queue and claim distinct
tasks. The default preflight compares single-task and packed actions/metrics
on representative episodes.

Compile POGEMA packages from the four Hugging Face checkpoints
(downloaded automatically into `weights/`) using
`python -m evaluation.compile_aoti --benchmark pogema`, selecting
`--precision fp32` or `--precision hybrid`. The default for pretrained 3M is
FP32; the other three models default to hybrid. In hybrid mode, both 0.8M
checkpoints use an FP16 network and both 3M checkpoints use a BF16 network,
with FP32 sampling and consensus dynamics.
The compiler records eager/AOTI action disagreement rates rather than
requiring bitwise-identical actions from reduced-precision kernels.
