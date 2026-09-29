from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from trainctl.core.models import (
    Checkpoint,
    CheckpointEvalStatus,
    EvalJob,
    EvalStatus,
    MetricPoint,
    Run,
    RunStatus,
)


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class SQLiteStore:
    """SQLite persistence hidden behind the state-store API."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pid INTEGER,
                    pgid INTEGER,
                    step INTEGER,
                    total_steps INTEGER,
                    started_at TEXT,
                    finished_at TEXT,
                    exit_code INTEGER,
                    log_path TEXT,
                    checkpoint_dir TEXT,
                    latest_checkpoint TEXT,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    step INTEGER,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_metrics_lookup
                    ON metrics(run_id, name, id);
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    step INTEGER,
                    created_at TEXT NOT NULL,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    evaluation_status TEXT NOT NULL,
                    UNIQUE(run_id, path),
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    job_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    checkpoint_path TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    exit_code INTEGER,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def save_run(self, run: Run) -> None:
        values = (
            run.run_id, run.name, json.dumps(run.command), run.status.value,
            run.pid, run.pgid, run.step, run.total_steps,
            run.started_at.isoformat() if run.started_at else None,
            run.finished_at.isoformat() if run.finished_at else None,
            run.exit_code, str(run.log_path) if run.log_path else None,
            str(run.checkpoint_dir) if run.checkpoint_dir else None,
            str(run.latest_checkpoint) if run.latest_checkpoint else None,
            json.dumps(run.metrics), run.created_at.isoformat(),
        )
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    name=excluded.name, command_json=excluded.command_json,
                    status=excluded.status, pid=excluded.pid, pgid=excluded.pgid,
                    step=excluded.step, total_steps=excluded.total_steps,
                    started_at=excluded.started_at, finished_at=excluded.finished_at,
                    exit_code=excluded.exit_code, log_path=excluded.log_path,
                    checkpoint_dir=excluded.checkpoint_dir,
                    latest_checkpoint=excluded.latest_checkpoint,
                    metrics_json=excluded.metrics_json""",
                values,
            )

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> Run:
        return Run(
            run_id=row["run_id"], name=row["name"], command=json.loads(row["command_json"]),
            status=RunStatus(row["status"]), pid=row["pid"], pgid=row["pgid"],
            step=row["step"], total_steps=row["total_steps"],
            started_at=_dt(row["started_at"]), finished_at=_dt(row["finished_at"]),
            exit_code=row["exit_code"], log_path=Path(row["log_path"]) if row["log_path"] else None,
            checkpoint_dir=Path(row["checkpoint_dir"]) if row["checkpoint_dir"] else None,
            latest_checkpoint=Path(row["latest_checkpoint"]) if row["latest_checkpoint"] else None,
            metrics=json.loads(row["metrics_json"]), created_at=_dt(row["created_at"]),  # type: ignore[arg-type]
        )

    def get_run(self, run_id: str) -> Run | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return self._row_to_run(row) if row else None

    def list_runs(self, limit: int = 100) -> list[Run]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def save_metric(self, point: MetricPoint) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO metrics(run_id, step, name, value, timestamp) VALUES (?, ?, ?, ?, ?)",
                (point.run_id, point.step, point.name, point.value, point.timestamp.isoformat()),
            )

    @staticmethod
    def _row_to_metric(row: sqlite3.Row) -> MetricPoint:
        return MetricPoint(run_id=row["run_id"], step=row["step"], name=row["name"],
                           value=row["value"], timestamp=_dt(row["timestamp"]))  # type: ignore[arg-type]

    def latest_metric(self, run_id: str, name: str) -> MetricPoint | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM metrics WHERE run_id=? AND name=? ORDER BY id DESC LIMIT 1",
                (run_id, name),
            ).fetchone()
        return self._row_to_metric(row) if row else None

    def metric_history(self, run_id: str, name: str, limit: int | None = None) -> list[MetricPoint]:
        sql = "SELECT * FROM metrics WHERE run_id=? AND name=? ORDER BY id ASC"
        params: tuple[object, ...] = (run_id, name)
        if limit is not None:
            sql = (
                "SELECT * FROM (SELECT * FROM metrics WHERE run_id=? AND name=? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id ASC"
            )
            params = (run_id, name, limit)
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [self._row_to_metric(row) for row in rows]

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, path) DO UPDATE SET step=excluded.step,
                    metrics_json=excluded.metrics_json,
                    evaluation_status=excluded.evaluation_status""",
                (checkpoint.checkpoint_id, checkpoint.run_id, str(checkpoint.path), checkpoint.step,
                 checkpoint.created_at.isoformat(), json.dumps(checkpoint.metrics),
                 checkpoint.evaluation_status.value),
            )

    @staticmethod
    def _row_to_checkpoint(row: sqlite3.Row) -> Checkpoint:
        return Checkpoint(
            checkpoint_id=row["checkpoint_id"], run_id=row["run_id"], path=Path(row["path"]),
            step=row["step"], created_at=_dt(row["created_at"]),  # type: ignore[arg-type]
            metrics=json.loads(row["metrics_json"]),
            evaluation_status=CheckpointEvalStatus(row["evaluation_status"]),
        )

    def list_checkpoints(self, run_id: str) -> list[Checkpoint]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM checkpoints WHERE run_id=? ORDER BY created_at ASC", (run_id,)
            ).fetchall()
        return [self._row_to_checkpoint(row) for row in rows]

    def save_eval_job(self, job: EvalJob) -> None:
        values = (
            job.job_id, job.run_id, str(job.checkpoint_path), json.dumps(job.command), job.status.value,
            job.created_at.isoformat(), job.started_at.isoformat() if job.started_at else None,
            job.finished_at.isoformat() if job.finished_at else None, job.exit_code, json.dumps(job.result),
        )
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET status=excluded.status,
                    started_at=excluded.started_at, finished_at=excluded.finished_at,
                    exit_code=excluded.exit_code, result_json=excluded.result_json""",
                values,
            )
