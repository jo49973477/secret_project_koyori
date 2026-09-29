from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from trainctl.core.events import MetricUpdated
from trainctl.core.metric_store import MetricStore
from trainctl.core.models import Run
from trainctl.core.run_manager import RunManager
from trainctl.integrations.base import TrainingAdapter

from .checkpoint_watcher import CheckpointWatcher
from .log_watcher import LogBuffer

logger = logging.getLogger(__name__)


class Supervisor:
    """Launch and supervise a command without involving a shell."""

    def __init__(
        self,
        run_manager: RunManager,
        metric_store: MetricStore,
        adapter: TrainingAdapter,
        *,
        ring_buffer_lines: int = 200,
        checkpoint_watcher: CheckpointWatcher | None = None,
    ) -> None:
        self.run_manager = run_manager
        self.metric_store = metric_store
        self.adapter = adapter
        self.ring_buffer_lines = ring_buffer_lines
        self.checkpoint_watcher = checkpoint_watcher
        self._buffers: dict[str, LogBuffer] = {}

    def tail(self, run_id: str, count: int = 20) -> list[str]:
        buffer = self._buffers.get(run_id)
        return buffer.tail(count) if buffer else []

    async def supervise(self, run: Run, *, cwd: Path | None = None) -> Run:
        if not run.log_path:
            raise ValueError("Run log_path must be set before supervision")
        run.log_path.parent.mkdir(parents=True, exist_ok=True)
        buffer = self._buffers.setdefault(run.run_id, LogBuffer(self.ring_buffer_lines))
        await self.run_manager.starting(run)
        try:
            process = await asyncio.create_subprocess_exec(
                *run.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                start_new_session=(os.name == "posix"),
            )
        except (OSError, ValueError) as exc:
            await self.run_manager.failed(run, None, reason=str(exc))
            return run

        pgid: int | None = None
        if os.name == "posix":
            try:
                pgid = os.getpgid(process.pid)
            except ProcessLookupError:
                pass
        await self.run_manager.started(run, process.pid, pgid)
        stop_watcher = asyncio.Event()
        watcher_task: asyncio.Task[None] | None = None
        if self.checkpoint_watcher and run.checkpoint_dir:
            watcher_task = asyncio.create_task(self.checkpoint_watcher.watch(run, stop_watcher))

        assert process.stdout is not None
        try:
            with run.log_path.open("a", encoding="utf-8", buffering=1) as log_file:
                async for raw_line in process.stdout:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    log_file.write(line + "\n")
                    buffer.append(line)
                    for event in self.adapter.parse_line(run.run_id, line):
                        if isinstance(event, MetricUpdated):
                            await self.metric_store.record(
                                event.run_id, event.name, event.value, event.step
                            )
                            self.run_manager.update_metric(
                                run, event.name, event.value, event.step
                            )
            exit_code = await process.wait()
        finally:
            stop_watcher.set()
            if watcher_task:
                await watcher_task

        if exit_code == 0:
            await self.run_manager.completed(run, exit_code)
        else:
            await self.run_manager.failed(
                run, exit_code, tail=tuple(buffer.tail(min(50, self.ring_buffer_lines)))
            )
        return run

