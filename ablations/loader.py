"""Load a release checkpoint for a chosen inference-time refinement depth."""

from __future__ import annotations

from pathlib import Path

from evaluation.models import MODELS, load_checkpoint_model

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "weights"


def load_dmm(model: str, rounds: int, device):
    """Return the checkpoint's model unrolled for `rounds` communication rounds."""
    if MODELS[model]["architecture"] != "canonical-dmm":
        raise ValueError(f"the ablations cover the 3M architecture only, not {model}")
    if rounds < 1:
        raise ValueError("refinement rounds must be positive")
    net, config, step = load_checkpoint_model(model, CHECKPOINTS / f"{model}.pt", device)
    # The shared loader pins the trained depth; rounds share weights, so the
    # ablations vary the depth at inference without touching the checkpoint.
    config.n_comm_rounds = rounds
    return net, config, step
