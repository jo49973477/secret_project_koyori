from __future__ import annotations

from datetime import datetime, timezone

from trainctl.agent.resource_monitor import DiskInfo, ResourceSummary
from trainctl.core.events import (
    CheckpointSaved,
    DiskLow,
    EvalCompleted,
    EvalFailed,
    Event,
    GpuWarning,
    RunCompleted,
    RunFailed,
    RunStarted,
)
from trainctl.core.models import Run


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_status(run: Run | None) -> str:
    if not run:
        return "No experiments have been recorded."
    lines = [f"🧪 {run.name}", "", f"Status: {run.status.value.upper()}"]
    if run.step is not None:
        suffix = f" / {run.total_steps}" if run.total_steps is not None else ""
        lines.append(f"Step: {run.step}{suffix}")
    if "loss" in run.metrics:
        lines.append(f"Loss: {run.metrics['loss']:.6g}")
    if run.started_at:
        end = run.finished_at or datetime.now(timezone.utc)
        lines.append(f"Elapsed: {_duration((end - run.started_at).total_seconds())}")
    if run.pid:
        lines.append(f"PID: {run.pid}")
    if run.exit_code is not None:
        lines.append(f"Exit code: {run.exit_code}")
    return "\n".join(lines)


def format_gpu(summary: ResourceSummary) -> str:
    lines = ["GPU resources"]
    if not summary.gpus:
        lines.append(summary.gpu_error or "No NVIDIA GPUs detected.")
    for gpu in summary.gpus:
        util = "n/a" if gpu.utilization_percent is None else f"{gpu.utilization_percent:.0f}%"
        memory = "n/a"
        if gpu.memory_used_mb is not None and gpu.memory_total_mb is not None:
            memory = f"{gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f} MiB"
        temp = "n/a" if gpu.temperature_c is None else f"{gpu.temperature_c:.0f}°C"
        power = "" if gpu.power_w is None else f", {gpu.power_w:.0f} W"
        lines.append(f"GPU {gpu.index} {gpu.name}: {util}, {memory}, {temp}{power}")
    lines.append(f"System RAM: {summary.ram_used_percent:.1f}% used")
    return "\n".join(lines)


def _bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def format_disk(disks: list[DiskInfo]) -> str:
    if not disks:
        return "No configured disk paths could be inspected."
    lines = ["Disk usage"]
    for disk in disks:
        marker = " ⚠️" if disk.warning else ""
        lines.append(
            f"{disk.path}: {disk.percent_used:.1f}% used, {_bytes(disk.free_bytes)} free{marker}"
        )
    return "\n".join(lines)


def format_tail(run: Run | None, lines: list[str]) -> str:
    if not run:
        return "No experiments have been recorded."
    if not lines:
        return f"No log output is available for {run.name}."
    body = "\n".join(lines)
    return f"{run.name} — latest {len(lines)} lines\n\n{body}"[-4000:]


def format_event(event: Event) -> str | None:
    if isinstance(event, RunStarted):
        return f"Training started: {event.name} (PID {event.pid}, run {event.run_id})"
    if isinstance(event, RunCompleted):
        return f"Training completed: run {event.run_id}"
    if isinstance(event, RunFailed):
        reason = event.reason or f"exit code {event.exit_code}"
        tail = "\n".join(event.tail[-10:])
        return f"Training failed: run {event.run_id} ({reason})" + (f"\n\n{tail}" if tail else "")
    if isinstance(event, CheckpointSaved):
        step = f" at step {event.step}" if event.step is not None else ""
        return f"Checkpoint saved{step}: {event.path}"
    if isinstance(event, EvalCompleted):
        return f"Evaluation completed: job {event.job_id}, run {event.run_id}"
    if isinstance(event, EvalFailed):
        return f"Evaluation failed: job {event.job_id} ({event.reason})"
    if isinstance(event, DiskLow):
        return f"Disk space warning: {event.path} is {event.percent_used:.1f}% used"
    if isinstance(event, GpuWarning):
        return f"GPU warning: {event.message}"
    return None
