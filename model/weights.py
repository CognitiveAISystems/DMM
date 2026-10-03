"""Resolve release weights from a local file or the pinned Hugging Face release."""

from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"
REPO_ID = "tviskaron/DMM"
REVISION = "b8cc8258bf4872ec23aec5332090453c52115889"
MODEL_NAMES = ("DMM-08M", "DMM-3M", "DMM-MICPO-08M", "DMM-MICPO-3M")


def resolve_weights(path: str | Path) -> Path:
    """Use local weights; download only missing canonical release paths.

    Custom paths must exist, so a typo cannot silently select release weights.
    HF_HUB_OFFLINE=1 disables network access through huggingface_hub.
    """
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    if path.parent != WEIGHTS_DIR or path.name not in {
        f"{name}.pt" for name in MODEL_NAMES
    }:
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(
            repo_id=REPO_ID, filename=path.name, revision=REVISION,
            local_dir=WEIGHTS_DIR,
        ))
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {path.name} from {REPO_ID}@{REVISION}. "
            f"Check connectivity or place the checkpoint at {path}. "
            "For offline runs, download weights before setting HF_HUB_OFFLINE=1."
        ) from exc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="model names; defaults to all four")
    args = parser.parse_args()
    for name in args.models or MODEL_NAMES:
        if name not in MODEL_NAMES:
            parser.error(f"unknown model: {name}; choose from {MODEL_NAMES}")
        print(resolve_weights(WEIGHTS_DIR / f"{name}.pt"))
