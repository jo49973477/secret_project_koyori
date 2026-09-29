from __future__ import annotations

from dataclasses import dataclass

from trainctl.core.events import (
    CheckpointSaved, DiskLow, EvalCompleted, EvalFailed, Event, GpuWarning,
    RunCompleted, RunFailed, RunStarted,
)

MAX_CONTENT = 2000
MAX_EMBED_DESCRIPTION = 4096
MAX_FIELD_VALUE = 1024
MAX_EMBED_TOTAL = 6000


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _bounded_fields(title: str, description: str, fields: list[dict[str, str | bool]]) -> list[dict[str, str | bool]]:
    remaining = max(0, MAX_EMBED_TOTAL - len(title) - len(description))
    result: list[dict[str, str | bool]] = []
    for field in fields[:25]:
        name = _truncate(str(field["name"]), 256)
        value = _truncate(str(field["value"]), min(MAX_FIELD_VALUE, max(1, remaining - len(name))))
        cost = len(name) + len(value)
        if cost > remaining:
            break
        result.append({"name": name, "value": value, "inline": bool(field.get("inline", False))})
        remaining -= cost
    return result


@dataclass(frozen=True, slots=True)
class DiscordMessage:
    content: str | None = None
    embeds: tuple[dict, ...] = ()

    def payload(self) -> dict:
        result: dict = {"allowed_mentions": {"parse": []}}
        if self.content:
            result["content"] = _truncate(self.content, MAX_CONTENT)
        if self.embeds:
            result["embeds"] = list(self.embeds[:10])
        return result


def format_event(event: Event) -> DiscordMessage | None:
    title = description = ""
    color = 0x5865F2
    fields: list[dict[str, str | bool]] = []
    if isinstance(event, RunStarted):
        title, color = "🚀 Training Started", 0x3498DB
        fields = [{"name": "Run ID", "value": event.run_id, "inline": True}, {"name": "PID", "value": str(event.pid), "inline": True}]
        description = f"Experiment: **{event.name}**\nStarted: {event.occurred_at.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}"
    elif isinstance(event, RunCompleted):
        title, color, description = "✅ Training Completed", 0x2ECC71, f"Run `{event.run_id}` completed successfully."
    elif isinstance(event, RunFailed):
        title, color = "🚨 Training Failed", 0xE74C3C
        fields = [{"name": "Run ID", "value": event.run_id, "inline": True}, {"name": "Exit Code", "value": str(event.exit_code if event.exit_code is not None else "unknown"), "inline": True}]
        if event.reason:
            fields.append({"name": "Reason", "value": _truncate(event.reason, MAX_FIELD_VALUE), "inline": False})
        if event.tail:
            fields.append({"name": "Recent Logs", "value": _truncate("\n".join(event.tail[-8:]), MAX_FIELD_VALUE), "inline": False})
        description = "The supervised training process exited unsuccessfully."
    elif isinstance(event, CheckpointSaved):
        title, color = "💾 Checkpoint Saved", 0x95A5A6
        fields = [{"name": "Path", "value": _truncate(str(event.path), MAX_FIELD_VALUE), "inline": False}]
        if event.step is not None:
            fields.append({"name": "Step", "value": str(event.step), "inline": True})
        description = f"Run `{event.run_id}` saved a checkpoint."
    elif isinstance(event, EvalCompleted):
        title, color = "📊 Evaluation Completed", 0x2ECC71
        description = f"Evaluation `{event.job_id}` for run `{event.run_id}` completed."
        if event.result:
            fields = [{"name": name, "value": _truncate(str(value), MAX_FIELD_VALUE), "inline": True} for name, value in list(event.result.items())[:25]]
    elif isinstance(event, EvalFailed):
        title, color = "⚠️ Evaluation Failed", 0xE74C3C
        description = f"Run `{event.run_id}`, job `{event.job_id}`: {_truncate(event.reason, MAX_EMBED_DESCRIPTION)}"
    elif isinstance(event, DiskLow):
        title, color, description = "⚠️ Disk Space Warning", 0xF1C40F, f"`{event.path}` is {event.percent_used:.1f}% full."
    elif isinstance(event, GpuWarning):
        title, color, description = "⚠️ GPU Warning", 0xF1C40F, _truncate(event.message, MAX_EMBED_DESCRIPTION)
    else:
        return None
    title = _truncate(title, 256)
    description = _truncate(description, MAX_EMBED_DESCRIPTION)
    embed = {"title": title, "description": description, "color": color, "fields": _bounded_fields(title, description, fields), "timestamp": event.occurred_at.isoformat()}
    return DiscordMessage(embeds=(embed,))
