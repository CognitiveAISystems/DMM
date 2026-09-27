# Evaluation

The entry point is `python -m evaluation.run --model MODEL --benchmark
BENCHMARK`. It supports two single-GPU benchmark families; the four-GPU
scalability experiment has a separate runner:

| Experiment | Scope | Task set |
| --- | --- | --- |
| [`pogema/`](pogema/) | DMM-08M, DMM-3M, DMM-MICPO-08M, DMM-MICPO-3M; no collision shielding | 3,193 episodes; 7 infeasible episodes excluded |
| [`movingai/`](movingai/) | Two MICPO models with CS-PIBT and RSE | 1,600 tasks across 32 maps |
| [`one_million/`](one_million/) | DMM-MICPO-08M and GPU-PIBT; four GPUs per run | 4 mazes, each with 1,048,576 agents |

The two single-GPU benchmarks use one persistent process per visible GPU and
the same local task-queue worker. Compiled packages are read from
`compiled/<model>-<benchmark>.pt2`, and results are written to `eval_results/`;
`--package` and `--output-root` optionally override those
locations. Each benchmark's manifest and archive live beside its code in
[`pogema/`](pogema/) or [`movingai/`](movingai/) and are verified and unpacked
into the output directory's `.benchmark_cache` on first use. No algorithm
knobs are exposed by the entry point.

The POGEMA package uses Dirichlet z0 and samples communication votes. The
MovingAI package uses zero z0 and argmax votes at all four rounds. Both take
the final argmax action. A package compiled for one benchmark is rejected by
the other.

## Precision modes

Compile any model with `--precision fp32` or `--precision hybrid`. Without the
flag, the 0.8M models and MICPO 3M use hybrid precision, while pretrained 3M
uses FP32. In hybrid mode, the 0.8M networks use FP16 and the 3M networks use
BF16; sampling and consensus dynamics stay FP32.

FP32 gives the highest numerical fidelity when reproducing the reported
results. Hybrid precision can provide an additional speedup, but it does not
guarantee identical actions or benchmark results. The compiler checks output
validity and records reduced-precision action differences against eager
inference at several input sizes. Keep package and output paths separate when
comparing modes:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m evaluation.compile_aoti --model DMM-MICPO-3M --benchmark movingai --precision fp32 --output compiled/DMM-MICPO-3M-movingai-fp32.pt2
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m evaluation.compile_aoti --model DMM-MICPO-3M --benchmark movingai --precision hybrid --output compiled/DMM-MICPO-3M-movingai-hybrid.pt2
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m evaluation.run --model DMM-MICPO-3M --benchmark movingai --package compiled/DMM-MICPO-3M-movingai-fp32.pt2 --output-root eval_results/fp32
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m evaluation.run --model DMM-MICPO-3M --benchmark movingai --package compiled/DMM-MICPO-3M-movingai-hybrid.pt2 --output-root eval_results/hybrid
```

The GPU simulator and its native CUDA sources are bundled in
[`pogema_gpu/`](../pogema_gpu/) and installed by the root `uv` project. Follow
the Linux/CUDA setup in the [root README](../README.md); four model-only
checkpoints are bundled under [`checkpoints/`](../checkpoints/), while the
compiled packages are generated on the evaluation host.

The four-GPU scalability experiment is separate from those single-GPU
benchmarks. Its source is in [`one_million/`](one_million/) and its scenario
generation and launch commands are in the [root README](../README.md).
