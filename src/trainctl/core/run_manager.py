from __future__ import annotations

import os
import uuid
from pathlib import Path

from trainctl.storage.base import StateStore

from .event_bus import EventBus
from .events import RunCompleted, RunFailed, RunStarted
from .models import Run, RunStatus, utc_now


class InvalidRunTransition(ValueError):
    pass


_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.CREATED: {RunStatus.STARTING, RunStatus.FAILED},
    RunStatus.STARTING: {RunStatus.RUNNING, RunStatus.FAILED},
    RunStatus.RUNNING: {RunStatus.PAUSED, RunStatus.EVALUATING, RunStatus.COMPLETED, RunStatus.FAILED},
    RunStatus.PAUSED: {RunStatus.RUNNING, RunStatus.FAILED},
    RunStatus.EVALUATING: {RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.FAILED},
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: set(),
}


class RunManager:
    def __init__(self, store: StateStore, event_bus: EventBus) -> None:
        self.store = store
        self.event_bus = event_bus

    def create(
        self,
        name: str,
        command: list[str],
        *,
        log_path: Path | None = None,
        checkpoint_dir: Path | None = None,
        total_steps: int | None = None,
    ) -> Run:
        if not command:
            raise ValueError("A training command is required")
        run = Run(
            run_id=uuid.uuid4().hex[:12], name=name, command=list(command),
            log_path=log_path, checkpoint_dir=checkpoint_dir, total_steps=total_steps,
        )
        self.store.save_run(run)
        return run

    def transition(self, run: Run, status: RunStatus) -> Run:
        if status == run.status:
            return run
        if status not in _TRANSITIONS[run.status]:
            raise InvalidRunTransition(f"Cannot transition {run.status.value} -> {status.value}")
        run.status = status
        self.store.save_run(run)
        return run

    async def starting(self, run: Run) -> None:
        self.transition(run, RunStatus.STARTING)

    async def started(self, run: Run, pid: int, pgid: int | None) -> None:
        run.pid = pid
        run.pgid = pgid
        run.started_at = utc_now()
        self.transition(run, RunStatus.RUNNING)
        await self.event_bus.publish(RunStarted(run_id=run.run_id, name=run.name, pid=pid))

    async def completed(self, run: Run, exit_code: int = 0) -> None:
        run.exit_code = exit_code
        run.finished_at = utc_now()
        self.transition(run, RunStatus.COMPLETED)
        await self.event_bus.publish(RunCompleted(run_id=run.run_id, exit_code=exit_code))

    async def failed(
        self, run: Run, exit_code: int | None, *, tail: tuple[str, ...] = (), reason: str | None = None
    ) -> None:
        run.exit_code = exit_code
        run.finished_at = utc_now()
        self.transition(run, RunStatus.FAILED)
        await self.event_bus.publish(
            RunFailed(run_id=run.run_id, exit_code=exit_code, tail=tail, reason=reason)
        )

    def update_metric(self, run: Run, name: str, value: float, step: int | None) -> None:
        run.metrics[name] = value
        if step is not None:
            run.step = step
        self.store.save_run(run)

    def latest(self) -> Run | None:
        runs = self.store.list_runs(limit=1)
        return runs[0] if runs else None

    def reconcile(self) -> list[Run]:
        """Mark records whose saved PID is no longer alive as failed."""
        changed: list[Run] = []
        for run in self.store.list_runs():
            if run.status not in {RunStatus.STARTING, RunStatus.RUNNING, RunStatus.PAUSED, RunStatus.EVALUATING}:
                continue
            alive = False
            if run.pid:
                try:
                    os.kill(run.pid, 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                except PermissionError:
                    # The process exists even if this user cannot signal it.
                    alive = True
            if not alive:
                run.status = RunStatus.FAILED
                run.finished_at = utc_now()
                self.store.save_run(run)
                changed.append(run)
        return changed
