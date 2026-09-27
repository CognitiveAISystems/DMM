"""Evaluate the refinement-depth and intent ablations on the frozen POGEMA episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ablations.loader import load_dmm
from ablations.policy import MODES, AblationPolicy
from ablations.tasks import FAMILIES, load_family
from evaluation.models import MODELS

ROOT = Path(__file__).resolve().parents[1]
ABLATION_MODELS = tuple(name for name, spec in MODELS.items()
                        if spec["architecture"] == "canonical-dmm")
K_TEST = (1, 2, 3, 4, 8, 12)
K_TRAIN = 4


EXPERIMENTS = {
    "round": {"rounds": K_TEST, "modes": ("full",), "families": FAMILIES},
    "intent": {"rounds": (K_TRAIN,), "modes": MODES, "families": ("02-mazes",)},
}


def configurations(experiment: str, rounds, modes):
    """A depth sweep varies the executed depth; the interventions run at the trained depth.

    The file name follows from that: a depth for the sweep, the intervention otherwise.
    """
    if experiment == "round":
        for depth in rounds:
            yield depth, "full", f"K{depth}"
    else:
        for mode in modes:
            yield rounds[0], mode, mode


def episode_rows(tasks, records, label: str) -> list[dict]:
    completed = {record["task_id"]: record for record in records}
    rows = []
    for task in tasks:
        record = completed.get(task.task_id)
        if record is None or record["metrics"] is None:
            raise ValueError(f"episode did not complete: {task.task_id}")
        provenance = task.provenance
        environment = ({"num_agents": task.num_agents, "map_name": provenance["map_name"]}
                       if provenance["named_map"]
                       else {"seed": provenance["generation_seed"], "num_agents": task.num_agents})
        rows.append({"metrics": record["metrics"], "env_grid_search": environment,
                     "algorithm": label})
    return rows


def report(label: str, every: int):
    """Episode-level progress, since one configuration is thousands of episodes."""
    def observe(_record, completed, total):
        if completed % every == 0 or completed == total:
            print(f"  {label}: {completed}/{total} episodes", flush=True)
    return observe


def evaluate(model_name: str, depth: int, mode: str, tasks: list, *,
             max_agents: int, label: str) -> list[dict]:
    import torch
    from pogema_gpu.toolbox.cuda_ragged import evaluate_ragged

    net, _, _ = load_dmm(model_name, depth, torch.device("cuda:0"))
    policy = AblationPolicy(net, model_name=model_name, precision=MODELS[model_name]["precision"],
                            rounds=depth, mode=mode, max_num_agents=max_agents)
    # A derangement spans the packed agent axis, so the shuffled variant evaluates one
    # environment at a time and messages stay inside their own episode.
    output = evaluate_ragged(tasks, policy, num_envs=1 if mode == "shuffled" else 128,
                             max_batch_agents=max_agents, record_trace=False,
                             movement_backend="resolved", progress=report(label, 64))
    if output["completed_tasks"] != len(tasks):
        raise RuntimeError("evaluation was incomplete")
    return output["records"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=tuple(EXPERIMENTS))
    parser.add_argument("--model", required=True, choices=ABLATION_MODELS)
    parser.add_argument("--rounds", type=int, nargs="+",
                        help="executed refinement depths (default: the experiment's own)")
    parser.add_argument("--modes", nargs="+", choices=MODES,
                        help="broadcast interventions (default: the experiment's own)")
    parser.add_argument("--families", nargs="+", choices=FAMILIES,
                        help="POGEMA families (default: the experiment's own)")
    parser.add_argument("--output-root", type=Path, default=ROOT / "ablation_results")
    parser.add_argument("--cache-root", type=Path,
                        default=ROOT / "ablation_results" / ".benchmark_cache")
    parser.add_argument("--max-agents", type=int,
                        help="packed agent budget; defaults to the model profile")
    args = parser.parse_args(argv)

    defaults = EXPERIMENTS[args.experiment]
    rounds = args.rounds or defaults["rounds"]
    modes = args.modes or defaults["modes"]
    families = args.families or defaults["families"]

    max_agents = args.max_agents or MODELS[args.model]["max_num_agents"]
    output_root = args.output_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    for family in families:
        print(f"{family}: loading episodes", flush=True)
        tasks = load_family(family, cache_root)
        for depth, mode, suffix in configurations(args.experiment, rounds, modes):
            label = f"{args.model}-{suffix}"
            records = evaluate(args.model, depth, mode, tasks,
                               max_agents=max_agents, label=f"{family} {label}")
            target = output_root / family / f"{label}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w") as stream:
                json.dump(episode_rows(tasks, records, label), stream)
            print(json.dumps({"family": family, "algorithm": label,
                              "episodes": len(tasks), "output": str(target)}))


if __name__ == "__main__":
    main()
