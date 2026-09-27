"""Build/load helper for the exact CUDA PIBT extension."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import shutil


@lru_cache(maxsize=1)
def load_cuda_pibt(verbose: bool = False):
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent
    cuda_flags = ["-O3", "--use_fast_math"]
    for version in (12, 11, 10):
        compiler = shutil.which(f"g++-{version}")
        if compiler:
            os.environ.setdefault("CXX", compiler)
            cuda_flags.append(f"--compiler-bindir={compiler}")
            break
    return load(
        name="dmm_million_cuda_pibt_batched_v6",
        sources=[str(root / "cuda_pibt.cpp"), str(root / "cuda_pibt_kernel.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=cuda_flags,
        verbose=verbose,
    )


def main() -> None:
    load_cuda_pibt(verbose=True)
    print("dmm_cuda_pibt extension is ready")


if __name__ == "__main__":
    main()
