from __future__ import annotations

from collections import deque
from pathlib import Path

from trainctl.agent.resource_monitor import DiskInfo, ResourceMonitor, ResourceSummary
from trainctl.core.models import Run
from trainctl.storage.base import StateStore


def is_authorized(user_id: int | None, allowed_users: set[int]) -> bool:
    """Deny by default; an empty allowlist permits nobody."""
    return user_id is not None and user_id in allowed_users


class CommandService:
    """UI-neutral queries used by Telegram and future frontends."""

    def __init__(self, store: StateStore, resources: ResourceMonitor, disk_paths: list[Path]) -> None:
        self.store = store
        self.resources = resources
        self.disk_paths = disk_paths

    def status(self) -> Run | None:
        runs = self.store.list_runs(limit=1)
        return runs[0] if runs else None

    async def gpu(self) -> ResourceSummary:
        return await self.resources.get_gpu_summary()

    async def disk(self) -> list[DiskInfo]:
        return await self.resources.get_disk_summary(self.disk_paths)

    def tail(self, count: int = 20) -> tuple[Run | None, list[str]]:
        run = self.status()
        count = max(1, min(count, 200))
        if not run or not run.log_path or not run.log_path.exists():
            return run, []
        lines: deque[str] = deque(maxlen=count)
        try:
            with run.log_path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    lines.append(line.rstrip("\r\n"))
        except OSError:
            return run, []
        return run, list(lines)

