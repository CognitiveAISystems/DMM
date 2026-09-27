"""Load corridor checkpoints and the ambiguous state each model family expects."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import fields
import glob
from pathlib import Path
import sys

import torch

CORRIDOR = Path(__file__).resolve().parents[1]
ROOT = CORRIDOR.parents[1]
THIRD_PARTY = CORRIDOR / "third_party"
UPSTREAM_HMAGAT = THIRD_PARTY / "upstream_hmagat"
DATA = CORRIDOR / "data"

# Row 1 of the tokenized dataset is t=1 of the first trajectory: the state both
# experts share, where either agent may yield.
AMBIGUOUS_ROW = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _hmagat_path() -> None:
    """HMAGAT is fetched rather than vendored, and its graphs unpickle through it."""
    if not UPSTREAM_HMAGAT.is_dir():
        raise SystemExit(f"{UPSTREAM_HMAGAT} is missing; run WITH_HMAGAT=1 ./setup.sh")
    sys.path.insert(0, str(UPSTREAM_HMAGAT))


def _unwrap(state_dict: dict) -> dict:
    """Drop the wrapper prefixes a compiled or distributed run leaves on its keys."""
    unwrapped = {}
    for key, tensor in state_dict.items():
        while key.startswith(("_orig_mod.", "module.", "net.")):
            key = key.split(".", 1)[1]
        unwrapped[key] = tensor
    return unwrapped


def load_dmm(checkpoint: Path, rounds: int | None = None, device: str = DEVICE):
    """DMM from a corridor checkpoint, optionally unrolled to a different depth.

    Overriding the depth is valid because the rounds share weights, so a
    checkpoint trained at one depth loads cleanly and runs at another.
    """
    sys.path.insert(0, str(ROOT))
    from model.dmm import DMM, DMMConfig

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    options = dict(payload["model_args"])
    if rounds is not None:
        options["n_comm_rounds"] = rounds
    allowed = {field.name for field in fields(DMMConfig)}
    model = DMM(DMMConfig(**{k: v for k, v in options.items() if k in allowed}))
    model.load_state_dict(_unwrap(payload["model"]), strict=True)
    return model.to(device).eval()


def load_lcmapf(checkpoint: Path, device: str = DEVICE):
    sys.path.insert(0, str(THIRD_PARTY))
    from lc_mapf.model import LCMAPF, Config

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    allowed = {field.name for field in fields(Config)}
    options = {k: v for k, v in payload["model_args"].items() if k in allowed}
    model = LCMAPF(Config(**options))
    model.load_state_dict(_unwrap(payload["model"]), strict=True)
    return model.to(device).eval()


def load_magat(checkpoint: Path, device: str = DEVICE):
    sys.path.insert(0, str(THIRD_PARTY))
    from magat_plus.magat.agents import get_model

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model, _ = get_model(Namespace(**payload["model_args"]), device)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()


def load_hmagat(checkpoint: Path, device: str = DEVICE):
    _hmagat_path()
    from hmagat.modules.agents import get_model

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model, _, _ = get_model(Namespace(**payload["model_args"]), device)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()


def ambiguous_tokens(device: str = DEVICE):
    """The ambiguous state as DMM and LC-MAPF read it: obs [1,2,256], neighbors [1,2,13]."""
    import pyarrow as pa

    with pa.memory_map(str(DATA / "part_0_0.arrow")) as source:
        table = pa.ipc.open_file(source).read_all()
    observation = table["input_tensors"].to_pylist()[AMBIGUOUS_ROW]
    neighbors = table["agents_in_obs"].to_pylist()[AMBIGUOUS_ROW]
    return (torch.tensor(observation, dtype=torch.int, device=device)[None],
            torch.tensor(neighbors, dtype=torch.int, device=device)[None])


def ambiguous_graph(name: str, device: str = DEVICE):
    """The ambiguous state as the graph MAGAT+ or HMAGAT consumes."""
    if name == "hmagat":
        _hmagat_path()
    graphs = torch.load(DATA / f"{name}_graphs.pt", weights_only=False)
    return graphs[AMBIGUOUS_ROW].to(device)


def checkpoints(label: str, pattern: str) -> list[str]:
    """Expand one glob into the checkpoint of each training seed."""
    found = sorted(glob.glob(pattern))
    if not found and Path(pattern).is_file():
        found = [pattern]
    if not found:
        raise FileNotFoundError(f"no checkpoints for {label!r}: {pattern!r}")
    return found
