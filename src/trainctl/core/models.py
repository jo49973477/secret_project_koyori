from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


class EvalStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CheckpointEvalStatus(str, Enum):
    NOT_EVALUATED = "not_evaluated"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class Run:
    run_id: str
    name: str
    command: list[str]
    status: RunStatus = RunStatus.CREATED
    pid: int | None = None
    pgid: int | None = None
    step: int | None = None
    total_steps: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    log_path: Path | None = None
    checkpoint_dir: Path | None = None
    latest_checkpoint: Path | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class MetricPoint:
    run_id: str
    step: int | None
    name: str
    value: float
    timestamp: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class Checkpoint:
    checkpoint_id: str
    run_id: str
    path: Path
    step: int | None
    created_at: datetime = field(default_factory=utc_now)
    metrics: dict[str, float] = field(default_factory=dict)
    evaluation_status: CheckpointEvalStatus = CheckpointEvalStatus.NOT_EVALUATED


@dataclass(slots=True)
class EvalJob:
    job_id: str
    run_id: str
    checkpoint_path: Path
    command: list[str]
    status: EvalStatus = EvalStatus.CREATED
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    result: dict[str, Any] = field(default_factory=dict)

