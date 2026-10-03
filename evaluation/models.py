"""The four DMM variants and their AOTI policy loader."""

import json
from pathlib import Path

from model.weights import resolve_weights


def load_checkpoint_model(name: str, path: Path, device):
    """Load one release checkpoint with the same key normalization for all evaluators."""
    from dataclasses import fields

    import torch

    from model.dmm import DMM, DMMConfig
    from model.dmm_08m import DMM08M, DMM08MConfig

    spec = MODELS[name]
    payload = torch.load(resolve_weights(path), map_location="cpu", weights_only=True)
    options = dict(payload["model_args"], n_comm_rounds=4)
    model_type, config_type = ((DMM08M, DMM08MConfig)
                               if spec["architecture"] == "dmm-08m"
                               else (DMM, DMMConfig))
    allowed = {field.name for field in fields(config_type)}
    config = config_type(**{key: value for key, value in options.items()
                            if key in allowed})
    model = model_type(config)
    state = {}
    for original, tensor in payload["model"].items():
        key = original
        while key.startswith(("_orig_mod.", "module.", "net.")):
            key = key.split(".", 1)[1]
        if key in state:
            raise ValueError(f"duplicate normalized tensor name: {key}")
        state[key] = tensor
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config, int(payload["iter_num"])


MODELS = {
    "DMM-08M": {
        "architecture": "dmm-08m", "training": "pretrain",
        "benchmarks": ("pogema",), "precision": "fp16-fp32", "max_num_agents": 2580,
    },
    "DMM-3M": {
        "architecture": "canonical-dmm", "training": "pretrain",
        "benchmarks": ("pogema",), "precision": "fp32", "max_num_agents": 256,
    },
    "DMM-MICPO-08M": {
        "architecture": "dmm-08m", "training": "micpo",
        "benchmarks": ("pogema", "movingai"), "precision": "fp16-fp32",
        "max_num_agents": 8192,
    },
    "DMM-MICPO-3M": {
        "architecture": "canonical-dmm", "training": "micpo",
        "benchmarks": ("pogema", "movingai"), "precision": "bf16-fp32",
        "max_num_agents": 8192,
    },
}

ROUND_MODES = {"pogema": "sample", "movingai": "argmax"}


def precision_for(model: str, mode: str | None = None) -> str:
    """Resolve the requested compilation mode; None keeps the model default."""
    spec = MODELS[model]
    if mode is None:
        return spec["precision"]
    if mode == "fp32":
        return "fp32"
    if mode == "hybrid":
        return "fp16-fp32" if spec["architecture"] == "dmm-08m" else "bf16-fp32"
    raise ValueError(f"unsupported precision mode: {mode}")


def package_precision(model: str, package: Path) -> str:
    contract = json.loads(Path(str(package) + ".json").read_text())
    if contract.get("model_name") != model:
        raise ValueError(f"{model} package model mismatch")
    precision = contract.get("precision")
    if precision not in {precision_for(model, "fp32"), precision_for(model, "hybrid")}:
        raise ValueError(f"{model} package has unsupported precision: {precision}")
    return precision


def round_mode_for(model: str, benchmark: str) -> str:
    if benchmark not in MODELS[model]["benchmarks"]:
        raise ValueError(f"{model} is not a verified {benchmark} model")
    return ROUND_MODES[benchmark]


def policy_for(model: str, package: Path, *, benchmark: str, shielded: bool,
               max_num_agents: int | None = None):
    from evaluation.adapter import DMMRaggedPolicy

    spec = MODELS[model]
    agent_limit = spec["max_num_agents"] if max_num_agents is None else max_num_agents
    if not 1 <= agent_limit <= spec["max_num_agents"]:
        raise ValueError("AOTI agent limit is outside the model profile")
    policy = DMMRaggedPolicy(
        package, model_name=model, architecture=spec["architecture"],
        precision=package_precision(model, package), max_num_agents=agent_limit,
        benchmark=benchmark, round_mode=round_mode_for(model, benchmark),
        shielded=shielded,
    )
    policy.configure_cost_to_go(cache_mode="vendor-window")
    return policy
