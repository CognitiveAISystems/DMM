"""Portable tasks: layouts are data, never instructions to a random generator."""

from dataclasses import dataclass, field
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
from pathlib import Path

PROFILE = "pogema-1.3.2a4-soft-nothing-v1"
VARIABLE_PROFILE = "pogema-1.3.2a4-soft-nothing-variable-horizon-v1"
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(value):
    rows = tuple(value.split() if isinstance(value, str) else value)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("obstacles must be a nonempty rectangular grid")
    if any(set(row) - {".", "#"} for row in rows):
        raise ValueError("obstacles use only '.' (free) and '#' (wall)")
    return rows


@dataclass(frozen=True)
class Task:
    task_id: str
    obstacles: tuple[str, ...]
    starts: tuple[tuple[int, int], ...]
    goals: tuple[tuple[int, int], ...]
    obs_radius: int = 5
    policy_seed: int = 20260905
    # Includes Pogema's artificial wall and any cells visible beyond that wall.
    observation_obstacles: tuple[str, ...] | None = None
    provenance: dict = field(default_factory=dict, compare=False)
    horizon: int = 128

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a nonempty string")
        object.__setattr__(self, "obstacles", _rows(self.obstacles))
        for name in ("starts", "goals"):
            points = tuple(tuple(point) for point in getattr(self, name))
            if not points or any(len(p) != 2 or any(type(x) is not int for x in p) for p in points):
                raise ValueError(f"{name} must contain integer (row, column) pairs")
            for row, col in points:
                if not (0 <= row < self.height and 0 <= col < self.width):
                    raise ValueError(f"{name} contains an out-of-bounds position")
                if self.obstacles[row][col] != ".":
                    raise ValueError(f"{name} contains a wall position; tasks never clear walls")
            object.__setattr__(self, name, points)
        if len(self.starts) != len(self.goals) or len(set(self.starts)) != len(self.starts):
            raise ValueError("starts must be unique and match the number of goals")
        if type(self.obs_radius) is not int or not 1 <= self.obs_radius <= 128:
            raise ValueError("obs_radius must be an integer in [1, 128]")
        if type(self.policy_seed) is not int or not 0 <= self.policy_seed < 2**63:
            raise ValueError("policy_seed must be an integer in [0, 2**63)")
        if type(self.horizon) is not int or not 1 <= self.horizon <= 65535:
            raise ValueError("horizon must be an integer in [1, 65535]")
        r = self.obs_radius
        padded = self.observation_obstacles
        if padded is None:
            cells = [["."] * (self.width + 2*r) for _ in range(self.height + 2*r)]
            for i in range(r-1, r+self.height+1):
                for j in range(r-1, r+self.width+1):
                    cells[i][j] = (self.obstacles[i-r][j-r]
                                   if r <= i < r+self.height and r <= j < r+self.width else "#")
            padded = tuple("".join(row) for row in cells)
        padded = _rows(padded)
        if len(padded) != self.height+2*r or len(padded[0]) != self.width+2*r:
            raise ValueError("observation_obstacles has the wrong border width")
        if tuple(row[r:-r] for row in padded[r:-r]) != self.obstacles:
            raise ValueError("observation_obstacles interior differs from obstacles")
        ring = [padded[r-1][j] for j in range(r-1, r+self.width+1)]
        ring += [padded[r+self.height][j] for j in range(r-1, r+self.width+1)]
        ring += [padded[i][j] for i in range(r-1, r+self.height+1) for j in (r-1, r+self.width)]
        if any(cell != "#" for cell in ring):
            raise ValueError("the artificial boundary must be closed")
        object.__setattr__(self, "observation_obstacles", padded)

    @property
    def height(self):
        return len(self.obstacles)

    @property
    def width(self):
        return len(self.obstacles[0])

    @property
    def num_agents(self):
        return len(self.starts)

    @property
    def bucket(self):
        return self.height, self.width, self.num_agents, self.obs_radius

    @property
    def layout_hash(self):
        return content_hash({"obstacles": self.obstacles, "starts": self.starts,
                             "goals": self.goals, "observation_obstacles": self.observation_obstacles,
                             "obs_radius": self.obs_radius, "profile": self.profile, "horizon": self.horizon})

    @property
    def profile(self):
        return PROFILE if self.horizon == 128 else VARIABLE_PROFILE

    def to_dict(self):
        return {"task_id": self.task_id, "obstacles": self.obstacles, "starts": self.starts,
                "goals": self.goals, "obs_radius": self.obs_radius, "policy_seed": self.policy_seed,
                "observation_obstacles": self.observation_obstacles, "layout_hash": self.layout_hash,
                "provenance": self.provenance, "horizon": self.horizon}

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        expected = value.pop("layout_hash")
        task = cls(**value)
        if task.layout_hash != expected:
            raise ValueError(f"layout hash mismatch: {task.task_id}")
        return task


@contextmanager
def _task_file(path, *, writing=False):
    """Optional deterministic gzip; uncompressed JSON remains the default."""
    path = Path(path)
    with path.open("xb" if writing else "rb") as raw:
        stream = (gzip.GzipFile(filename="", fileobj=raw, mode="wb" if writing else "rb", mtime=0)
                  if path.suffix == ".gz" else raw)
        with io.TextIOWrapper(stream, encoding="utf-8") as handle:
            yield handle


def save_tasks(path, tasks, *, shared_maps=False):
    tasks = list(tasks)
    _unique(tasks)
    payload = {"schema": "pogema-gpu-tasks-v1", "profile": PROFILE, "horizon": 128,
               "coordinates": "zero-based row,column in the unpadded map",
               "actions": MOVES, "tasks": [task.to_dict() for task in tasks]}
    if any(task.horizon != 128 for task in tasks):
        payload.update(schema="pogema-gpu-tasks-v2", profile=VARIABLE_PROFILE, horizon="per-task")
    if shared_maps:
        # Store repeated million-cell layouts once, without changing any Task
        # value or layout hash. Positions are still explicit and ordered.
        maps = {}
        for item in payload["tasks"]:
            layout = {key: item.pop(key) for key in ("obstacles", "observation_obstacles")}
            reference = content_hash(layout)
            maps.setdefault(reference, layout)
            item["map_ref"] = reference
        payload.update(schema="pogema-gpu-tasks-v3", profile=VARIABLE_PROFILE,
                       horizon="per-task", maps=maps)
    # Exclusive creation prevents accidentally replacing an accepted dataset.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _task_file(path, writing=True) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_tasks(path):
    with _task_file(path) as handle:
        value = json.load(handle)
    shared = value.get("schema") == "pogema-gpu-tasks-v3"
    if shared:
        maps = value.pop("maps")
        if not isinstance(maps, dict) or not maps:
            raise ValueError("shared task maps must be nonempty")
        for reference, layout in maps.items():
            if (not isinstance(layout, dict) or set(layout) != {"obstacles", "observation_obstacles"}
                    or content_hash(layout) != reference):
                raise ValueError("shared map hash or fields mismatch")
        used = set()
        for item in value["tasks"]:
            reference = item.pop("map_ref")
            if reference not in maps or "obstacles" in item or "observation_obstacles" in item:
                raise ValueError("invalid or ambiguous shared map reference")
            item.update(maps[reference])
            used.add(reference)
        if used != set(maps):
            raise ValueError("unused shared maps")
        value["schema"] = "pogema-gpu-tasks-v2"
    legacy = (value.get("schema"),value.get("profile"),value.get("horizon")) == ("pogema-gpu-tasks-v1",PROFILE,128)
    variable = (value.get("schema"),value.get("profile"),value.get("horizon")) == ("pogema-gpu-tasks-v2",VARIABLE_PROFILE,"per-task")
    if not (legacy or variable) or value.get("actions") != [list(m) for m in MOVES]:
        raise ValueError("unsupported task schema or compatibility profile")
    if legacy and any(item.get("horizon",128)!=128 for item in value["tasks"]):
        raise ValueError("v1 tasks require horizon 128")
    if variable and any("horizon" not in item for item in value["tasks"]):
        raise ValueError("v2 tasks require explicit horizons")
    tasks = [Task.from_dict(item) for item in value["tasks"]]
    _unique(tasks)
    return tasks


def _unique(tasks):
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("task sets must be nonempty with unique task IDs")
