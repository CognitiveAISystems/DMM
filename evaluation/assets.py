"""Materialize the versioned benchmark archives into a local evaluation cache."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

EVALUATION_ROOT = Path(__file__).resolve().parent
ARCHIVES = {
    "movingai": (
        "maps-scenarios.tar.gz", "hydra/benchmarks/mapf_tracker/", 1632,
    ),
    "pogema": (
        "instances.tar.gz", "instances/", 6400,
    ),
}


def materialize(benchmark: str, cache_root: Path) -> Path:
    """Safely unpack the bundled archive, shared by local GPU workers."""
    if benchmark not in ARCHIVES:
        raise ValueError(f"unknown benchmark: {benchmark}")
    filename, prefix, expected_files = ARCHIVES[benchmark]
    archive = EVALUATION_ROOT / benchmark / filename
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / benchmark
    with (cache_root / f".{benchmark}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            marker = destination / ".complete.json"
            if not marker.is_file():
                raise ValueError(f"incomplete benchmark cache: {destination}")
            record = json.loads(marker.read_text())
            count = sum(item.is_file() and item.name != ".complete.json"
                        for item in destination.rglob("*"))
            if record.get("files") != expected_files or count != expected_files:
                raise ValueError(f"modified benchmark cache: {destination}")
            return destination
        temporary = Path(tempfile.mkdtemp(prefix=f".{benchmark}-", dir=cache_root))
        try:
            seen: set[str] = set()
            with tarfile.open(archive, "r:gz") as source:
                for member in source:
                    relative = PurePosixPath(member.name)
                    if (not member.isfile() or relative.is_absolute()
                            or ".." in relative.parts or member.name in seen
                            or not member.name.startswith(prefix)):
                        raise ValueError(f"unsafe benchmark archive member: {member.name}")
                    seen.add(member.name)
                    target = temporary.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stream = source.extractfile(member)
                    if stream is None:
                        raise ValueError(f"unreadable archive member: {member.name}")
                    with stream, target.open("xb") as output:
                        shutil.copyfileobj(stream, output)
            count = len(seen)
            if count != expected_files:
                raise ValueError(f"{benchmark} archive contains {count}, not {expected_files} files")
            (temporary / ".complete.json").write_text(json.dumps({"files": count}))
            os.rename(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary)
            raise
    return destination
