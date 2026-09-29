from __future__ import annotations

import uuid
from pathlib import Path

from trainctl.storage.base import StateStore

from .event_bus import EventBus
from .events import CheckpointSaved
from .models import Checkpoint


class CheckpointManager:
    def __init__(self, store: StateStore, event_bus: EventBus) -> None:
        self.store = store
        self.event_bus = event_bus

    async def register(
        self, run_id: str, path: Path, step: int | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Checkpoint:
        existing = next((item for item in self.list(run_id) if item.path == path), None)
        if existing:
            return existing
        checkpoint = Checkpoint(
            checkpoint_id=uuid.uuid4().hex, run_id=run_id, path=path,
            step=step, metrics=metrics or {},
        )
        self.store.save_checkpoint(checkpoint)
        run = self.store.get_run(run_id)
        if run:
            run.latest_checkpoint = path
            self.store.save_run(run)
        await self.event_bus.publish(CheckpointSaved(run_id=run_id, path=path, step=step))
        return checkpoint

    def list(self, run_id: str) -> list[Checkpoint]:
        return self.store.list_checkpoints(run_id)

    def latest(self, run_id: str) -> Checkpoint | None:
        checkpoints = self.list(run_id)
        return checkpoints[-1] if checkpoints else None

    def by_step(self, run_id: str, step: int) -> Checkpoint | None:
        return next((item for item in self.list(run_id) if item.step == step), None)

    def best(self, run_id: str, metric: str, *, maximize: bool = True) -> Checkpoint | None:
        candidates = [item for item in self.list(run_id) if metric in item.metrics]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item.metrics[metric], reverse=maximize)[0]

