"""Run the four-seed, four-GPU million-agent DMM/GPU-PIBT evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (101, 202, 303, 404)
AGENTS = 1_048_576
HORIZON = 32_768


def _completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return result.get("status") == "completed" and (
        result.get("solved", False) or int(result.get("steps", 0)) >= HORIZON
    )


def _summary(raw_dir: Path) -> dict:
    rows = [json.loads(path.read_text()) for path in sorted(raw_dir.glob("seed*_n*_4gpu.json"))
            if _completed(path)]
    keys = ("sr", "isr", "makespan", "sum_cost", "wall_seconds")
    means = {key: sum(float(row[key]) for row in rows) / len(rows)
             for key in keys if rows and all(key in row for row in rows)}
    return {"completed": len(rows), "expected": len(SEEDS), "means": means, "results": rows}


def _command(algorithm: str, seed: int, master: Path, checkpoint: Path,
             root: Path, resume: bool) -> tuple[list[str], Path, Path]:
    tag = f"seed{seed}_n{AGENTS}_4gpu"
    result = root / "raw" / f"{tag}.json"
    state = root / "state" / tag
    command = [
        str(Path(sys.executable).with_name("torchrun")), "--standalone", "--nproc-per-node=4",
        "-m", f"evaluation.one_million.distributed_{algorithm}",
        "--scenario-master", str(master), "--agents", str(AGENTS),
        "--horizon", str(HORIZON), "--bfs-chunk", "2048",
        "--cache-radius", "128", "--pibt-resolver", "components",
        "--output", str(result), "--state-dir", str(state),
        "--checkpoint-every", "256", "--trajectory-actions",
        str(root / "trajectories" / f"{tag}.actions.u8.gz"),
    ]
    if algorithm == "dmm":
        command.extend((
            "--agent-chunk", "32768", "--checkpoint", str(checkpoint),
            "--skip-settled-neighborhoods", "--stochastic-policy",
            "--stochastic-tail-seed", str(seed),
        ))
    else:
        command.extend(("--preference-seed", str(seed)))
    latest = state / "latest.json"
    if resume and latest.is_file():
        command.extend(("--resume-from", str(latest)))
    elif latest.exists() or result.exists():
        raise ValueError(f"unfinished evaluation for seed {seed} requires --resume")
    return command, result, root / "logs" / f"{tag}.log"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("dmm", "gpu-pibt", "both"), default="both")
    parser.add_argument("--output-root", type=Path, default=ROOT / "eval_results" / "one_million")
    parser.add_argument("--scenario-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "DMM-MICPO-08M.pt")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from evaluation.one_million.resume import atomic_json, run_logged

    gpu_ids = tuple(part.strip() for part in args.gpus.split(","))
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        parser.error("the million-agent evaluation requires four distinct GPUs")
    compiler = shutil.which("gcc-11")
    cxx = shutil.which("g++-11")
    if not compiler or not cxx:
        parser.error("CUDA extension compilation requires GCC/G++ 11")
    output_root = args.output_root.expanduser().resolve()
    scenario_root = (args.scenario_root or output_root / "scenarios").expanduser().resolve()
    masters = {seed: scenario_root / f"master_seed{seed}.pt" for seed in SEEDS}
    for master in masters.values():
        if not master.is_file():
            parser.error(f"Missing {master}; run python -m evaluation.one_million.generate")
    checkpoint = args.checkpoint.expanduser().resolve()
    if args.algorithm in ("dmm", "both") and not checkpoint.is_file():
        parser.error(f"Missing DMM checkpoint: {checkpoint}")

    output_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=",".join(gpu_ids), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               CC=compiler, CXX=cxx, MAX_JOBS="4",
               TORCH_CUDA_ARCH_LIST=env.get("TORCH_CUDA_ARCH_LIST", "9.0"),
               TORCH_EXTENSIONS_DIR=str(output_root / ".torch_extensions"))
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.4")
    subprocess.run([sys.executable, "-c", (
        "import torch; assert torch.cuda.device_count()==4; "
        "from pogema_gpu.kernels.cuda_bfs import load_cuda_bfs; "
        "from evaluation.one_million.cuda_pibt import load_cuda_pibt; "
        "bfs=load_cuda_bfs(); z=torch.zeros((8,8),dtype=torch.int8,device='cuda'); "
        "p=torch.tensor([[3,3]],device='cuda'); g=torch.tensor([[5,5]],device='cuda'); "
        "assert bfs.raw_bfs_cost2go(z,p,g,8,8,15).shape==(1,961); "
        "load_cuda_pibt()")], cwd=ROOT, env=env, check=True)

    algorithms = ("dmm", "pibt") if args.algorithm == "both" else (
        ("pibt",) if args.algorithm == "gpu-pibt" else ("dmm",))
    for algorithm in algorithms:
        root = output_root / ("dmm_micpo_08m" if algorithm == "dmm" else "gpu_pibt")
        for folder in ("raw", "logs", "state", "trajectories"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        status_path = root / "queue_status.json"
        for seed in SEEDS:
            result = root / "raw" / f"seed{seed}_n{AGENTS}_4gpu.json"
            if _completed(result):
                continue
            command, result, log_path = _command(
                algorithm, seed, masters[seed], checkpoint, root, args.resume)
            atomic_json(status_path, {"algorithm": algorithm, "status": "running",
                                      "seed": seed, "completed": _summary(root / "raw")["completed"]})
            with log_path.open("a" if args.resume else "w") as log:
                returncode = run_logged(command, log, env=env, cwd=ROOT)
            if returncode or not _completed(result):
                atomic_json(status_path, {"algorithm": algorithm, "status": "failed",
                                          "seed": seed, "exit_code": returncode})
                return returncode or 1
            atomic_json(root / "summary.json", _summary(root / "raw"))
        atomic_json(status_path, {"algorithm": algorithm, "status": "completed",
                                  "completed": len(SEEDS), "expected": len(SEEDS)})
        atomic_json(root / "summary.json", _summary(root / "raw"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
