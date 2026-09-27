"""Durable distributed evaluation state and recoverable gzip trajectories.

BFS windows are deterministic scratch space: only their centers/validity are
saved. Rebuild them before resuming measured steps. Checkpoints retain two
generations and publish latest.json only after every rank has fsynced its file.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def add_resume_arguments(parser):
    parser.add_argument("--state-dir", help="directory for durable evaluation state")
    parser.add_argument("--checkpoint-every", type=int, default=256,
                        help="save every N completed steps, plus final/interrupt state")
    parser.add_argument("--resume-from", help="checkpoint latest.json or generation manifest.json")


def run_identity(args, world, algorithm):
    keys = ("agents", "agent_chunk", "bfs_chunk", "cache_radius", "pibt_resolver",
            "skip_settled_neighborhoods", "stochastic_policy", "stochastic_tail_patience",
            "stochastic_tail_max_unresolved", "stochastic_tail_seed", "preference_seed")
    identity = {key: getattr(args, key) for key in keys if hasattr(args, key)}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    identity.update(algorithm=algorithm, world_size=world,
                    master_sha256=file_sha256(args.scenario_master),
                    torch=str(torch.__version__), cuda=torch.version.cuda,
                    trajectory_enabled=bool(args.trajectory_actions),
                    git_sha=revision.stdout.strip() if revision.returncode == 0 else None)
    for key in ("checkpoint", "config"):
        if getattr(args, key, None):
            identity[key + "_sha256"] = file_sha256(getattr(args, key))
    return identity


class EvaluationState:
    def __init__(self, args, rank, world, device, algorithm):
        if args.checkpoint_every < 1:
            raise ValueError("--checkpoint-every must be positive")
        self.args, self.rank, self.world, self.device = args, rank, world, device
        self.root = Path(args.state_dir).resolve() if args.state_dir else None
        if args.resume_from and self.root is None:
            raise ValueError("--resume-from requires --state-dir")
        self.identity = run_identity(args, world, algorithm)
        self.loaded, self.manifest = {}, {}
        self.start_step = 0
        self.solved = False
        self.stop_signal = 0
        self.io_seconds = 0.0
        self.restore_seconds = 0.0
        self.peak = [0.0, 0.0]
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)
            if not args.resume_from and (self.root / "latest.json").exists():
                raise ValueError("state already exists; use --resume-from or a new --state-dir")
            signal.signal(signal.SIGINT, self._request_stop)
            signal.signal(signal.SIGTERM, self._request_stop)
        if args.resume_from:
            source = Path(args.resume_from).resolve()
            self.manifest = json.loads(source.read_text())
            if self.manifest.get("format") != "distributed-eval-state-v1":
                raise ValueError("unsupported evaluation checkpoint format")
            if self.manifest["identity"] != self.identity:
                raise ValueError("resume inputs/configuration/world size differ from checkpoint")
            self.start_step = int(self.manifest["completed_steps"])
            self.solved = bool(self.manifest["solved"])
            if args.horizon < self.start_step:
                raise ValueError("new horizon is smaller than the saved completed step")
            entry = self.manifest["rank_files"][rank]
            state_file = source.parent / entry["path"]
            if file_sha256(state_file) != entry["sha256"]:
                raise ValueError(f"checkpoint hash mismatch: {state_file}")
            self.loaded = torch.load(state_file, map_location="cpu", weights_only=False)
            self.peak = self.loaded["peak"]
            if rank == 0:
                print(f"RESUME completed_steps={self.start_step} new_horizon={args.horizon}", flush=True)

    def _request_stop(self, signum, frame):
        self.stop_signal = signum

    def should_stop(self):
        if not self.root:
            return False
        flag = torch.tensor(self.stop_signal, dtype=torch.int32, device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        self.stop_signal = int(flag.item())
        return bool(self.stop_signal)

    def due(self, completed_steps, done, stopping):
        return self.root is not None and (
            completed_steps % self.args.checkpoint_every == 0 or done or stopping
            or completed_steps == self.args.horizon)

    def save(self, payload, completed_steps, solved, trajectory, wall_seconds):
        started = time.perf_counter()
        # One immutable directory per checkpoint; never overwrite the last good state.
        token = [f"step{completed_steps:08d}-{time.time_ns()}" if self.rank == 0 else None]
        dist.broadcast_object_list(token, src=0, device=self.device)
        generation = self.root / token[0]
        generation.mkdir(exist_ok=True)
        trajectory_state = trajectory.checkpoint() if self.rank == 0 and trajectory else None
        peak = [max(self.peak[0], torch.cuda.max_memory_allocated(self.device)),
                max(self.peak[1], torch.cuda.max_memory_reserved(self.device))]
        value = dict(payload=cpu_tree(payload), peak=peak, wall_seconds=wall_seconds,
                     trajectory=trajectory_state,
                     rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                              torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(self.device)))
        path = generation / f"rank{self.rank}.pt"
        with path.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        entry = dict(path=str(path.relative_to(self.root)), sha256=file_sha256(path))
        entries = [None] * self.world
        dist.all_gather_object(entries, entry)
        if self.rank == 0:
            manifest = dict(format="distributed-eval-state-v1", identity=self.identity,
                            completed_steps=completed_steps, solved=bool(solved), rank_files=entries)
            # Generation manifest is independently usable; paths relative to it.
            atomic_json(generation / "manifest.json", {
                **manifest, "rank_files": [{**e, "path": Path(e["path"]).name} for e in entries]})
            atomic_json(self.root / "latest.json", manifest)
            generations = sorted(self.root.glob("step*-*"), key=lambda p: p.stat().st_mtime)
            for stale in generations[:-2]:
                shutil.rmtree(stale)
            print(f"CHECKPOINT step={completed_steps} path={self.root / 'latest.json'}", flush=True)
        dist.barrier()
        self.io_seconds += time.perf_counter() - started

    def restore_rng(self):
        if not self.loaded:
            return
        rng = self.loaded["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], self.device)

    def report(self):
        return dict(resumed_from=self.args.resume_from, resumed_at_step=self.start_step,
                    cache_rebuild_and_restore_seconds=self.restore_seconds,
                    checkpoint_io_seconds_this_process=self.io_seconds,
                    state_dir=str(self.root) if self.root else None)


def restore_tensor(target, saved):
    """Copy a tensor or the saved prefix of a horizon-sized metric tensor."""
    if target.shape == saved.shape:
        target.copy_(saved)
    elif target.ndim == saved.ndim == 1 and len(saved) <= len(target):
        target[:len(saved)].copy_(saved)
    else:
        raise ValueError(f"checkpoint tensor shape {saved.shape} cannot fill {target.shape}")


def rebuild_cache(cache, saved, *, observation):
    """Recreate exactly the windows at saved centers, outside measured steps."""
    from pogema_gpu.kernels.cuda_bfs import raw_bfs_cost2go
    if observation:
        centers, valid, windows = cache._cache_center, cache._cache_valid, cache._cache_windows
        obstacles, goals = cache.obstacles_bool, cache.goals_t
        chunk = cache.cfg.bfs_chunk_size
    else:
        centers, valid, windows = cache.cache_centers, cache.cache_valid, cache.cache_windows
        obstacles, goals, chunk = cache.obstacles, cache.goals, cache.bfs_chunk_size
    centers.copy_(saved["centers"])
    valid.copy_(saved["valid"])
    indices = valid.nonzero(as_tuple=True)[0]
    for offset in range(0, len(indices), chunk):
        batch = indices[offset:offset+chunk]
        windows[batch] = raw_bfs_cost2go(obstacles, centers[batch], goals[batch],
                                       cache.height, cache.width, cache.cache_radius)


class Trajectory:
    """Append gzip members; checkpoint only complete, fsynced member boundaries."""
    def __init__(self, path, saved=None):
        self.path = Path(path).resolve()
        self.paths = [self.path, Path(str(self.path) + ".pibt_changes.bitpack.gz")]
        self.temps = [Path(str(p) + ".tmp") for p in self.paths]
        self.digests = [hashlib.sha256(), hashlib.sha256()]
        self.counts = [0, 0]
        self.raw = [None, None]
        self.streams = [None, None]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for i, target in enumerate(self.temps):
            if saved:
                entry = saved[i]
                candidates = [Path(entry["temporary"]), Path(entry["final"])]
                source = next((p for p in candidates if p.exists() and p.stat().st_size >= entry["offset"]), None)
                if source is None:
                    raise ValueError("checkpoint trajectory prefix is missing or truncated")
                recovering = Path(str(target) + ".recovering")
                with source.open("rb") as src, recovering.open("wb") as dst:
                    remaining = entry["offset"]
                    while remaining:
                        block = src.read(min(8 << 20, remaining))
                        if not block:
                            raise ValueError("truncated checkpoint trajectory")
                        dst.write(block)
                        remaining -= len(block)
                    dst.flush()
                    os.fsync(dst.fileno())
                with gzip.open(recovering, "rb") as stream:
                    for block in iter(lambda: stream.read(8 << 20), b""):
                        self.digests[i].update(block)
                        self.counts[i] += len(block)
                if self.digests[i].hexdigest() != entry["sha256"] or self.counts[i] != entry["bytes"]:
                    raise ValueError("checkpoint trajectory prefix hash/length mismatch")
                recovering.replace(target)
            else:
                if target.exists() or self.paths[i].exists():
                    raise FileExistsError(f"refusing to overwrite trajectory: {target}")
                target.touch()

    def append(self, actions, overridden):
        blocks = [actions.to(device="cpu", dtype=torch.uint8).numpy().tobytes(),
                  np.packbits(overridden.to(device="cpu").numpy(), bitorder="little").tobytes()]
        for i, block in enumerate(blocks):
            if self.streams[i] is None:
                self.raw[i] = self.temps[i].open("ab")
                self.streams[i] = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw[i],
                                                compresslevel=1, mtime=0)
            self.streams[i].write(block)
            self.digests[i].update(block)
            self.counts[i] += len(block)

    def checkpoint(self):
        entries = []
        for i in range(2):
            if self.streams[i] is not None:
                self.streams[i].close()
                self.raw[i].flush()
                os.fsync(self.raw[i].fileno())
                self.raw[i].close()
                self.raw[i] = self.streams[i] = None
            entries.append(dict(temporary=str(self.temps[i]), final=str(self.paths[i]),
                                offset=self.temps[i].stat().st_size, bytes=self.counts[i],
                                sha256=self.digests[i].hexdigest()))
        sync_directory(self.path.parent)
        return entries

    def finish(self):
        self.checkpoint()
        for source, target in zip(self.temps, self.paths):
            source.replace(target)
        sync_directory(self.path.parent)

    def metadata(self, steps, agents):
        return dict(actions=str(self.path), compression="gzip-level-1", dtype="uint8",
                    shape=[steps, agents], uncompressed_bytes=self.counts[0],
                    uncompressed_sha256=self.digests[0].hexdigest(),
                    moves_drow_dcol=[[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]],
                    reconstruction="positions[t+1] = positions[t] + moves[actions[t]]; initial state is in master",
                    pibt_changes=dict(path=str(self.paths[1]), compression="gzip-level-1",
                                      encoding="numpy.packbits", bitorder="little", shape=[steps, agents],
                                      packed_bytes_per_step=(agents+7)//8,
                                      uncompressed_bytes=self.counts[1],
                                      uncompressed_sha256=self.digests[1].hexdigest(),
                                      meaning="1 iff exact PIBT changed the policy argmax"))


def run_logged(command, log, *, env=None, cwd=None):
    """Forward termination to torchrun and keep draining its checkpoint logs."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, start_new_session=True,
                               env=env, cwd=cwd)
    requested = 0

    def stop(signum, frame):
        nonlocal requested
        if not requested:
            requested = signum
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        result = process.wait()
        return 128 + requested if requested else result
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
