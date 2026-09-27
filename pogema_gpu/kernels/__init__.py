"""Lazy, explicitly requested CUDA extensions; importing this module is CPU-safe."""

import functools
import hashlib
import os
from pathlib import Path
import sys


@functools.lru_cache(maxsize=None)
def extension(kind="native"):
    import torch
    from torch.utils.cpp_extension import load

    if kind not in {"native", "pibt"}:
        raise ValueError("unknown CUDA extension")
    source = Path(__file__).with_name(kind + ".cu")
    # Torch disables CUDA half conversions by default. CUDA 12.3's headers
    # themselves require these conversions when compiled as C++20 (Torch 2.13).
    # These kernels use integer/float tensors, not implicit half arithmetic.
    cuda_flags = ["-O3", "-U__CUDA_NO_HALF_CONVERSIONS__",
                  "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"]
    headers = b''.join(p.read_bytes() for p in sorted(source.parent.glob('*.cuh')))
    identity = (source.read_bytes() + headers + str((torch.__version__, torch.version.cuda, sys.version,
                os.environ.get("TORCH_CUDA_ARCH_LIST"), os.environ.get("CUDA_HOME"),
                os.environ.get("CXX"), cuda_flags)).encode())
    name = "pogema_gpu_" + kind + "_" + hashlib.sha256(identity).hexdigest()[:16]
    cache = Path(os.environ.get("POGEMA_GPU_CACHE", Path.home() / ".cache/pogema-gpu")) / name
    cache.mkdir(parents=True, exist_ok=True)
    return load(name, [str(source)], build_directory=str(cache),
                extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags,
                verbose=os.environ.get("POGEMA_GPU_BUILD_VERBOSE") == "1")
