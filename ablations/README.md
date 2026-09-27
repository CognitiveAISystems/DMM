# POGEMA ablations

Refinement-depth and intent-communication ablations for `DMM-3M` and
`DMM-MICPO-3M`, run from `checkpoints/` on the frozen POGEMA episodes in
`evaluation/pogema/`. Both are inference-time interventions: the weights and
each agent's own intent update are untouched, only the executed depth or the
transmitted message changes. The policy runs eagerly, so a depth is a runtime
argument rather than a compiled package, and `evaluation/` is unaffected.

## Refinement depth

Evaluates the trained checkpoints at `K_test` in {1, 2, 3, 4, 8, 12} on all four
families:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m ablations.run --experiment round --model DMM-3M
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m ablations.run --experiment round --model DMM-MICPO-3M
```

## Intent communication

Perturbs what each agent broadcasts, at the trained depth, on Mazes:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m ablations.run --experiment intent --model DMM-3M
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m ablations.run --experiment intent --model DMM-MICPO-3M
```

`full` broadcasts the learned feature and the evolving intent, `no-h` and `no-z`
drop one of the two, `z0-only` keeps broadcasting the initial intent while the
agent's own intent still evolves, and `shuffled` permutes assembled messages
across agents by a derangement. That permutation covers the packed agent axis,
so `shuffled` evaluates one environment at a time and is correspondingly slower.

Each experiment defaults to the domains and depths it was published with;
`--families`, `--rounds` and `--modes` override them. `--max-agents` bounds how
many agents are packed into one forward pass, defaulting to the model's profile
limit; eager activations are larger than the compiled evaluation's, so lower it
if a run runs out of memory. Packing does not affect results — each environment
keeps its own communication graph and episode RNG.

## Output

Results go to `ablation_results/<family>/<model>-<suffix>.json`, with `K<depth>`
for a depth sweep and the intervention name otherwise, in the same per-episode
envelope as `raw_results/`. No episodes are excluded, so every agent count
contributes 128.

The published data and figures are in
[`../raw_results/04_round_depth_ablation/`](../raw_results/04_round_depth_ablation/)
and [`../raw_results/05_intent_ablation/`](../raw_results/05_intent_ablation/).
Re-running reproduces them closely but not bit-exactly: episode RNG draws depend
on the executed depth.
