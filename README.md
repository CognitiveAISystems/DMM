<p align="center">
  <img src="assets/dmm_banner_v3_2x.svg" alt="Animated DMM multi-agent pathfinding banner" width="100%" />
</p>

# Decentralized Master-Mind

Code for DMM-08M and DMM-3M: pretraining, MICPO training,
POGEMA-GPU evaluation, four model checkpoints, benchmark inputs, and the raw
results and plotting scripts used for the paper.

## Contents

| Directory | Purpose |
| --- | --- |
| `model/` | The two checkpoint-compatible DMM architectures |
| `training/pretrain/` | DMM pretraining data loader, configs and shared training loop |
| `training/micpo/` | MICPO loop and configs; observations use the bundled POGEMA-GPU tokenizer |
| `checkpoints/` | Four model-only checkpoints without optimizer state |
| `pogema_gpu/` | Bundled CUDA environment used by training and evaluation |
| `evaluation/` | AOTI compilation, POGEMA/MovingAI evaluation, and the million-agent run |
| `ablations/` | Refinement-depth, intent-communication, and corridor-conflict studies |
| `raw_results/` | Experiment JSON, plotting scripts and paper figures |

The checkpoints contain FP32 model weights and configuration, but no optimizer
state. `DMM-08M` and `DMM-3M` are pretrained; `DMM-MICPO-08M` and
`DMM-MICPO-3M` are the MICPO-tuned models.

## Setup

Run the commands from the repository root. Checkpoints, maps, manifests, and
figure inputs are read from this checkout.

Use Linux x86-64, an NVIDIA GPU, a CUDA toolkit with `nvcc`, and GCC 11.
PyTorch is pinned to **2.13.0+cu126**. The code was smoke-tested on H100 with
CUDA 12.4.

```bash
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
uv sync --locked
```

The first GPU run compiles the POGEMA-GPU extensions. Download the pretraining
Arrow dataset with
`uv run --locked --group train python -m training.pretrain.download_dataset`.

## Training

```bash
uv sync --locked --group train
uv run --locked --group train torchrun --standalone --nproc_per_node=4 -m training.pretrain training/pretrain/DMM_3M.py
uv run --locked --group train torchrun --standalone --nproc_per_node=4 -m training.pretrain training/pretrain/DMM-08M.py
uv run --locked --group train torchrun --standalone --nproc_per_node=4 -m training.micpo --config training/micpo/dmm_08m.yaml
uv run --locked --group train torchrun --standalone --nproc_per_node=4 -m training.micpo --config training/micpo/dmm_3m.yaml
```

Both architectures use the same MICPO entry point and B=4, G=24, 500-iteration
recipe with 1.0 SoC and 0.3 blocked-step reward weights. The MICPO configs
select their policy classes and pretrained checkpoints.

## AOTI compilation

Compile one package per verified model/benchmark pair on the intended CUDA host.
POGEMA uses Dirichlet z0 and sampled communication; MovingAI uses zero z0 and
argmax communication:

```bash
mkdir -p compiled
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m evaluation.compile_aoti --model DMM-08M --benchmark pogema --output compiled/DMM-08M-pogema.pt2
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m evaluation.compile_aoti --model DMM-3M --benchmark pogema --output compiled/DMM-3M-pogema.pt2
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m evaluation.compile_aoti --model DMM-MICPO-08M --benchmark pogema --output compiled/DMM-MICPO-08M-pogema.pt2
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m evaluation.compile_aoti --model DMM-MICPO-3M --benchmark pogema --output compiled/DMM-MICPO-3M-pogema.pt2
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m evaluation.compile_aoti --model DMM-MICPO-08M --benchmark movingai --output compiled/DMM-MICPO-08M-movingai.pt2
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m evaluation.compile_aoti --model DMM-MICPO-3M --benchmark movingai --output compiled/DMM-MICPO-3M-movingai.pt2
```

Each compile writes a `.pt2` package and the `.pt2.json` runtime contract
used by the evaluator. The compiler checks the package at several input sizes
and runs a short POGEMA-GPU episode. Compilation options are documented in
[`evaluation/README.md`](evaluation/README.md).
`--max-autotune` enables slower performance tuning; it is not required for
correctness.

## Evaluation

After compiling the packages above, run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m evaluation.run --model DMM-08M --benchmark pogema
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m evaluation.run --model DMM-3M --benchmark pogema
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m evaluation.run --model DMM-MICPO-08M --benchmark movingai
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m evaluation.run --model DMM-MICPO-3M --benchmark movingai
```

Packages are read from `compiled/<model>-<benchmark>.pt2` and results are written to
`eval_results/`. Use `--package` or `--output-root` only for a different location.

POGEMA uses the frozen 3,193-task set (seven known infeasible tasks excluded)
without collision shielding. MovingAI uses 1,600 tasks, zero z0, argmax communication,
CS-PIBT, RSE16, and a 5,000-step limit. One process owns one visible GPU and
keeps the model loaded; multiple processes share a local resumable task queue.
See `evaluation/README.md` for details.

## One million agents

The scalability run uses four H100 GPUs jointly for one 1,048,576-agent
instance, with four 2304×2304 POGEMA mazes (seeds 101, 202, 303, 404), a
32,768-step limit, sampled DMM communication, and a GPU-PIBT baseline.
Generate the scenarios and run both methods:

```bash
uv run --locked python -m evaluation.one_million.generate --out eval_results/one_million/scenarios
uv run --locked python -m evaluation.one_million.run --algorithm both --output-root eval_results/one_million
```

`--resume` continues a stopped run from its saved step state.

## Ablations

The [POGEMA ablations](ablations/README.md) test executed refinement depth
across the benchmark families and intent-communication variants on Mazes for
`DMM-3M` and `DMM-MICPO-3M`. The
[corridor-conflict study](ablations/corridor/README.md) trains models on a
two-agent passing-bay scenario and measures coordinated actions across five
independent training seeds. Both READMEs give the commands and output formats.

The corresponding data and figures are in
[refinement depth](raw_results/04_round_depth_ablation/),
[intent communication](raw_results/05_intent_ablation/), and
[corridor ablations](raw_results/06_corridor_ablations/).

## Results and figures

`raw_results/` contains the POGEMA, MovingAI, and ablation result data, their
plot scripts, figure dependencies and rendered figures. Each subfolder has
short reproduction instructions.
