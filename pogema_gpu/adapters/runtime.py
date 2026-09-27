"""Numerical runtime settings for AOTI policies."""

# AOTI still calls external ATen/cuBLAS/cuDNN operations.
NATIVE_RUNTIME = {"matmul_tf32": True, "cudnn_tf32": True,
                  "bf16_reduction": True, "cudnn_benchmark": False,
                  "cudnn_deterministic": False}


def native_runtime_settings(runtime):
    return {"matmul_tf32": runtime.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": runtime.backends.cudnn.allow_tf32,
            "bf16_reduction": runtime.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "cudnn_benchmark": runtime.backends.cudnn.benchmark,
            "cudnn_deterministic": runtime.backends.cudnn.deterministic}


def configure_native_runtime(runtime):
    runtime.backends.cuda.matmul.allow_tf32 = NATIVE_RUNTIME["matmul_tf32"]
    runtime.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = NATIVE_RUNTIME["bf16_reduction"]
    runtime.backends.cudnn.allow_tf32 = NATIVE_RUNTIME["cudnn_tf32"]
    runtime.backends.cudnn.benchmark = NATIVE_RUNTIME["cudnn_benchmark"]
    runtime.backends.cudnn.deterministic = NATIVE_RUNTIME["cudnn_deterministic"]
    return native_runtime_settings(runtime)


def check_native_runtime(runtime):
    if native_runtime_settings(runtime) != NATIVE_RUNTIME:
        raise RuntimeError("native DMM numerical settings changed after initialization; "
                           "use an isolated persistent worker for incompatible policy profiles")
