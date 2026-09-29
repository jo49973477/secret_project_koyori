from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .models import utc_now


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    occurred_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunStarted(Event):
    run_id: str
    name: str
    pid: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCompleted(Event):
    run_id: str
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class RunFailed(Event):
    run_id: str
    exit_code: int | None
    tail: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricUpdated(Event):
    run_id: str
    name: str
    value: float
    step: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointSaved(Event):
    run_id: str
    path: Path
    step: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalStarted(Event):
    job_id: str
    run_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalCompleted(Event):
    job_id: str
    run_id: str
    result: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalFailed(Event):
    job_id: str
    run_id: str
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class DiskLow(Event):
    path: Path
    percent_used: float


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuWarning(Event):
    message: str
    run_id: str | None = None

