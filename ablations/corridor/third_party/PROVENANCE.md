# Vendored baselines

The corridor experiment trains every baseline from scratch on the two expert
trajectories, so only network definitions are vendored here — no trained
weights, datasets or benchmark harnesses. Each package keeps its upstream
licence next to the code it covers.

| Directory | Upstream | Commit | Licence |
| --- | --- | --- | --- |
| `magat_plus/` | https://github.com/proroklab/lagat | `69e0611d10567daf76db37fe9ca6af92766df188` (2026-02-13) | `magat_plus/LICENSE` |
| `lc_mapf/` | https://github.com/CognitiveAISystems/LC-MAPF | `f794cfb5be74057a0f43d35e078dea9c6ae98d0a` (2026-07-27) | `lc_mapf/LICENSE` |

## What was taken

- `magat_plus/`: `magat/agents.py` and `magat/model/` — `get_model` and the GNN
  definitions it selects between.
- `lc_mapf/`: `model.py`, plus `observation_generator.{cpp,h}`. The observation
  generator is the tokenizer that DMM and LC-MAPF share, so it also builds the
  corridor training data for both.

## HMAGAT is fetched, not vendored

HMAGAT builds its observations and hypergraphs through its own pipeline, which
reaches into the rest of its benchmark suite, and its graphs unpickle through
those same modules. Rather than copy that tree — or trim it and risk changing the
defaults that decide how hypergraphs are built — `setup.sh` clones
https://github.com/proroklab/hmagat at
`f664ae816c7f926606c2b22eef2cb8d92878acf3` (2026-03-03) into
`upstream_hmagat/`, which is untracked, and uses it unmodified. Skipping that
step leaves the other three models unaffected.

## Modifications

- `magat_plus/generate_additional_data.py` contains only `any_additional_data`,
  extracted verbatim from the upstream module of that name. `agents.py` needs
  that one function, while the rest of the upstream module imports its
  repository's full benchmark harness (expert runners, dataset loaders and the
  other baselines). Extracting it keeps the vendored tree to the networks alone.

No other file is modified. Empty `__init__.py` files were added where the
vendored subset needed package markers.
