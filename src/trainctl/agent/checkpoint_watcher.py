from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from trainctl.core.checkpoint_manager import CheckpointManager
from trainctl.core.models import Run

logger = logging.getLogger(__name__)


class CheckpointWatcher:
    """Low-overhead polling watcher with a configurable filename regex."""

    def __init__(
        self, manager: CheckpointManager, pattern: str,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        self.manager = manager
        self.pattern = re.compile(pattern)
        self.poll_interval_seconds = max(0.1, poll_interval_seconds)
        self._seen: set[Path] = set()

    async def scan(self, run: Run) -> int:
        if not run.checkpoint_dir or not run.checkpoint_dir.exists():
            return 0
        found = 0
        for path in sorted(run.checkpoint_dir.iterdir()):
            if path in self._seen:
                continue
            match = self.pattern.search(path.name)
            if not match:
                continue
            step_text = match.groupdict().get("step")
            step = int(step_text) if step_text is not None else None
            self._seen.add(path)
            await self.manager.register(run.run_id, path.resolve(), step)
            found += 1
        return found

    async def watch(self, run: Run, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.scan(run)
            except Exception:
                logger.exception("Checkpoint scan failed for run %s", run.run_id)
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.poll_interval_seconds)
            except TimeoutError:
                pass
        await self.scan(run)

