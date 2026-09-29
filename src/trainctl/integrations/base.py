from __future__ import annotations

from typing import Protocol

from trainctl.core.events import Event
from trainctl.core.models import Checkpoint, Run


class TrainingAdapter(Protocol):
    def parse_line(self, run_id: str, line: str) -> list[Event]: ...
    def discover_checkpoints(self, run: Run) -> list[Checkpoint]: ...


class NotificationAdapter(Protocol):
    async def __call__(self, event: Event) -> None: ...

    async def close(self) -> None: ...
