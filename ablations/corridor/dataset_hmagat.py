"""Build the HMAGAT graph dataset for the corridor scenario.

HMAGAT constructs its hypergraphs and observations through its own pipeline, which
depends on the rest of its benchmark suite, so that repository is fetched by
`WITH_HMAGAT=1 ./setup.sh` rather than vendored, and used unmodified here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

CORRIDOR = Path(__file__).resolve().parent
UPSTREAM = CORRIDOR / "third_party" / "upstream_hmagat"

from scenario import A_GOAL, B_GOAL, GRID, TRAJECTORIES  # noqa: E402

# HMAGAT's production flags, with one exception: the colour percentage is raised
# from 0.1 to 0.34, because the colouring takes int(free_cells * percentage)
# clusters and this grid's six free cells would otherwise give zero.
FLAGS = [
    "--obs_radius", "5",
    "--save_termination_state",
    "--hypergraph_comm_radius", "7",
    "--hyperedge_generation_method", "kmeans",
    "--hypergraph_num_updates", "10",
    "--hypergraph_wait_one",
    "--hypergraph_initial_colperc", "0.34",
    "--hypergraph_final_colperc", "0.34",
    "--add_data_cost_to_go",
    "--normalize_cost_to_go",
    "--clamp_cost_to_go", "1.0",
    "--use_lists",
    "--imitation_learning_model", "DirectionalHMAGAT",
    "--hyperedge_feature_generator", "magat",
    "--final_feature_generator", "magat",
    "--model_residuals", "all",
    "--use_edge_attr",
    "--use_edge_attr_for_messages", "positions+manhattan",
    "--edge_attr_cnn_mode", "MLP",
    "--load_positions_separately",
    "--train_on_terminated_agents",
    "--recursive_oe",
    "--cnn_mode", "ResNetLarge_withMLP",
]


def upstream():
    if not UPSTREAM.is_dir():
        raise SystemExit(
            f"{UPSTREAM} is missing; run WITH_HMAGAT=1 ./setup.sh to fetch it"
        )
    sys.path.insert(0, str(UPSTREAM))


def build_args():
    from hmagat.convert_to_imitation_dataset import add_imitation_dataset_args
    from hmagat.generate_additional_data import add_additional_data_args
    from hmagat.generate_hypergraphs import add_hypergraph_generation_args
    from hmagat.run_expert import add_expert_dataset_args
    from hmagat.temperature_training import add_temperature_sampling_args
    from hmagat.training_args import add_training_args

    parser = argparse.ArgumentParser()
    for add_arguments in (add_expert_dataset_args, add_imitation_dataset_args,
                          add_hypergraph_generation_args, add_additional_data_args,
                          add_training_args, add_temperature_sampling_args):
        parser = add_arguments(parser)
    return parser.parse_args(FLAGS)


def instance(positions):
    from pogema import GridConfig, pogema_v0

    config = GridConfig(
        map=GRID, size=max(len(GRID), len(GRID[0])),
        agents_xy=[list(position) for position in positions],
        targets_xy=[list(A_GOAL), list(B_GOAL)], num_agents=2,
        max_episode_steps=20, obs_radius=5, observation_type="MAPF",
        collision_system="soft", on_target="nothing", seed=0,
    )
    environment = pogema_v0(config)
    observations, _ = environment.reset()
    return config, environment, observations


def build(output: Path) -> None:
    upstream()
    from hmagat.modules.agents import get_model
    from hmagat.runtime_data_generation import get_runtime_data_generator

    args = build_args()
    _, hypergraph_model, dataset_kwargs = get_model(args, torch.device("cpu"))

    graphs = []
    for trajectory in TRAJECTORIES:
        for step in range(trajectory.horizon):
            config, environment, observations = instance(trajectory.positions[step])
            generate = get_runtime_data_generator(
                grid_config=config, args=args, hypergraph_model=hypergraph_model,
                dataset_kwargs=dataset_kwargs, use_target_vec=args.use_target_vec,
            )
            graph = generate(observations, environment)
            graph.y = torch.tensor(trajectory.actions[step], dtype=torch.long)
            graphs.append(graph)
            environment.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, output)
    print(f"{output}: {len(graphs)} graphs, node features {tuple(graphs[0].x.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CORRIDOR / "data" / "hmagat_graphs.pt")
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
