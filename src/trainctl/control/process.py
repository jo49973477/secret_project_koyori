from __future__ import annotations

import os
import signal

from trainctl.core.models import Run


class ProcessControlError(RuntimeError):
    pass


class ProcessController:
    """Low-level POSIX process controls.

    SIGSTOP/SIGCONT are intentionally exposed only as a local fallback. They
    can be unsafe for distributed DDP/NCCL/DeepSpeed jobs. A future cooperative
    controller can implement the same interface at training-step boundaries.
    """

    @staticmethod
    def _target(run: Run) -> int:
        if run.pgid and os.name == "posix":
            return -run.pgid
        if run.pid:
            return run.pid
        raise ProcessControlError("Run has no process ID")

    def pause(self, run: Run) -> None:
        if os.name != "posix":
            raise ProcessControlError("Low-level pause is only supported on POSIX")
        os.kill(self._target(run), signal.SIGSTOP)

    def resume(self, run: Run) -> None:
        if os.name != "posix":
            raise ProcessControlError("Low-level resume is only supported on POSIX")
        os.kill(self._target(run), signal.SIGCONT)

    def terminate(self, run: Run) -> None:
        os.kill(self._target(run), signal.SIGTERM)

