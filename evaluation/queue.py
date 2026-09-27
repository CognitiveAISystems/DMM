"""Single-host, local-filesystem task queue for independent GPU processes.

SQLite transactions assign disjoint tasks. A worker holds an OS lock for its
lifetime, so another process can recover its unfinished claims after a crash
without relying on a timeout during long episodes.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import sqlite3
import uuid


TERMINAL = ("solved", "no_solution", "error")


class LocalTaskQueue:
    def __init__(self, output_dir: Path):
        self.root = Path(output_dir)
        self.database = self.root / "tasks.sqlite3"
        self.lock_dir = self.root / "workers"

    def _connect(self):
        connection = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextmanager
    def _transaction(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self, metadata: dict, keys: list[str]) -> str:
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate task keys")
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        with self._transaction() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id=1), "
                "payload TEXT NOT NULL, signature TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS tasks (key TEXT PRIMARY KEY, ordinal INTEGER NOT NULL "
                "UNIQUE, state TEXT NOT NULL, owner TEXT, attempts INTEGER NOT NULL DEFAULT 0, "
                "last_error TEXT)"
            )
            row = connection.execute("SELECT payload, signature FROM metadata WHERE id=1").fetchone()
            if row is None:
                signature = uuid.uuid4().hex
                allowed = {"workers", "tasks.sqlite3", "tasks.sqlite3-wal", "tasks.sqlite3-shm"}
                unexpected = [path.name for path in self.root.iterdir() if path.name not in allowed]
                if unexpected:
                    raise ValueError(
                        "output directory contains files from another evaluation: "
                        + ", ".join(sorted(unexpected)[:5])
                    )
                connection.execute(
                    "INSERT INTO metadata(id,payload,signature) VALUES(1,?,?)",
                    (payload, signature),
                )
                connection.executemany(
                    "INSERT INTO tasks(key,ordinal,state) VALUES(?,?,'pending')",
                    ((key, index) for index, key in enumerate(keys)),
                )
            else:
                if row[0] != payload:
                    raise ValueError("output directory belongs to a different evaluation")
                signature = row[1]
            actual = [
                row[0] for row in connection.execute("SELECT key FROM tasks ORDER BY ordinal")
            ]
            if actual != keys:
                raise ValueError("task manifest does not match the existing queue")
        return signature

    @contextmanager
    def worker(self):
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        path = self.lock_dir / f"{token}.lock"
        with path.open("x") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield token
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _owner_alive(self, token: str) -> bool:
        path = self.lock_dir / f"{token}.lock"
        if not path.exists():
            return False
        with path.open("r") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
                return False

    def _artifact_state(self, key: str, signature: str) -> str | None:
        folder = self.root / "records" / key
        summary = folder / "summary.json"
        episode = folder / "episode.json.gz"
        if not summary.is_file():
            return None
        if not episode.is_file():
            raise ValueError(f"missing episode artifact for accepted result: {key}")
        record = json.loads(summary.read_text())
        if record.get("base_key") != key or record.get("signature") != signature:
            raise ValueError(f"existing result does not match evaluation: {key}")
        if record.get("status") not in TERMINAL[:2]:
            raise ValueError(f"invalid result status for {key}")
        return record["status"]

    def _recover(self, connection, current_owner: str, signature: str) -> None:
        owners = [
            row[0] for row in connection.execute(
                "SELECT DISTINCT owner FROM tasks WHERE state='running' AND owner!=?",
                (current_owner,),
            )
        ]
        for owner in owners:
            if self._owner_alive(owner):
                continue
            rows = connection.execute(
                "SELECT key,attempts FROM tasks WHERE state='running' AND owner=?",
                (owner,),
            ).fetchall()
            for key, attempts in rows:
                state = self._artifact_state(key, signature)
                if state is None:
                    state = "error" if attempts >= 3 else "pending"
                connection.execute(
                    "UPDATE tasks SET state=?,owner=NULL WHERE key=?",
                    (state, key),
                )

    def claim(self, owner: str, signature: str, limit: int) -> list[str]:
        if limit < 1:
            raise ValueError("claim limit must be positive")
        with self._transaction() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._recover(connection, owner, signature)
            keys = [
                row[0] for row in connection.execute(
                    "SELECT key FROM tasks WHERE state='pending' ORDER BY ordinal LIMIT ?",
                    (limit,),
                )
            ]
            connection.executemany(
                "UPDATE tasks SET state='running',owner=?,attempts=attempts+1 WHERE key=?",
                ((owner, key) for key in keys),
            )
        return keys

    def complete(self, owner: str, key: str, state: str) -> None:
        if state not in TERMINAL[:2]:
            raise ValueError("invalid terminal state")
        with self._transaction() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE tasks SET state=?,owner=NULL WHERE key=? AND owner=? AND state='running'",
                (state, key, owner),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"task is not claimed by this worker: {key}")

    def fail_claimed(self, owner: str, signature: str, message: str) -> None:
        with self._transaction() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT key,attempts FROM tasks WHERE owner=? AND state='running'", (owner,)
            ).fetchall()
            for key, attempts in rows:
                state = self._artifact_state(key, signature)
                if state is None:
                    state = "error" if attempts >= 3 else "pending"
                connection.execute(
                    "UPDATE tasks SET state=?,owner=NULL,last_error=? WHERE key=?",
                    (state, message[:1000] if state == "error" else None, key),
                )

    def status(self) -> dict:
        with self._transaction() as connection:
            counts = dict(connection.execute(
                "SELECT state,COUNT(*) FROM tasks GROUP BY state"
            ))
        total = sum(counts.values())
        return {
            "expected": total,
            "completed": counts.get("solved", 0) + counts.get("no_solution", 0),
            "solved": counts.get("solved", 0),
            "no_solution": counts.get("no_solution", 0),
            "errors": counts.get("error", 0),
            "pending": counts.get("pending", 0),
            "running": counts.get("running", 0),
        }

    def terminal_keys(self) -> dict[str, str]:
        with self._transaction() as connection:
            return dict(connection.execute(
                "SELECT key,state FROM tasks WHERE state IN ('solved','no_solution','error')"
            ))
