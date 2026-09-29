from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _default_data_dir() -> Path:
    configured = os.environ.get("TRAINCTL_HOME")
    return _path(configured) if configured else _path("~/.local/share/trainctl")


@dataclass(slots=True)
class TelegramConfig:
    enabled: bool = False
    token: str | None = None
    allowed_users: set[int] = field(default_factory=set)


@dataclass(slots=True)
class DiscordConfig:
    enabled: bool = False
    webhook_url: str | None = None


@dataclass(slots=True)
class StorageConfig:
    database: Path = field(default_factory=lambda: _default_data_dir() / "trainctl.db")


@dataclass(slots=True)
class LoggingConfig:
    directory: Path = field(default_factory=lambda: _default_data_dir() / "logs")
    level: str = "INFO"


@dataclass(slots=True)
class MonitoringConfig:
    disk_warning_percent: float = 90.0
    disk_paths: list[Path] = field(default_factory=list)


@dataclass(slots=True)
class CheckpointConfig:
    poll_interval_seconds: float = 5.0
    pattern: str = r"(?:checkpoint|ckpt)[-_]?(?P<step>\d+)"


@dataclass(slots=True)
class AppConfig:
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    source: Path | None = None


def _table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def load_config(path: Path | None = None) -> AppConfig:
    """Load defaults, then TOML, then environment overrides."""
    config = AppConfig()
    env_path = os.environ.get("TRAINCTL_CONFIG")
    candidate = path or (_path(env_path) if env_path else _path("~/.config/trainctl/config.toml"))
    data: dict[str, Any] = {}
    if candidate.exists():
        with candidate.open("rb") as handle:
            data = tomllib.load(handle)
        config.source = candidate

    notifications = _table(data, "notifications")
    telegram = _table(notifications, "telegram") or _table(data, "telegram")
    config.telegram.enabled = bool(telegram.get("enabled", config.telegram.enabled))
    config.telegram.allowed_users = {int(value) for value in telegram.get("allowed_users", [])}

    discord = _table(notifications, "discord") or _table(data, "discord")
    config.discord.enabled = bool(discord.get("enabled", False))
    config.discord.webhook_url = discord.get("webhook_url")

    storage = _table(data, "storage")
    if "database" in storage:
        config.storage.database = _path(storage["database"])
    logging_data = _table(data, "logging")
    if "directory" in logging_data:
        config.logging.directory = _path(logging_data["directory"])
    config.logging.level = str(logging_data.get("level", config.logging.level)).upper()

    monitoring = _table(data, "monitoring")
    config.monitoring.disk_warning_percent = float(
        monitoring.get("disk_warning_percent", config.monitoring.disk_warning_percent)
    )
    config.monitoring.disk_paths = [_path(value) for value in monitoring.get("disk_paths", [])]

    checkpoints = _table(data, "checkpoints")
    config.checkpoints.poll_interval_seconds = float(
        checkpoints.get("poll_interval_seconds", config.checkpoints.poll_interval_seconds)
    )
    config.checkpoints.pattern = str(checkpoints.get("pattern", config.checkpoints.pattern))

    token = os.environ.get("TRAINCTL_TELEGRAM_TOKEN")
    if token:
        config.telegram.token = token
        config.telegram.enabled = True
    webhook = os.environ.get("TRAINCTL_DISCORD_WEBHOOK_URL") or os.environ.get("DISCORD_WEBHOOK_URL")
    if webhook:
        config.discord.webhook_url = webhook
        # An explicit TOML disabled value remains authoritative.
        if "enabled" not in discord:
            config.discord.enabled = True
    allowed = os.environ.get("TRAINCTL_TELEGRAM_ALLOWED_USERS")
    if allowed is not None:
        try:
            config.telegram.allowed_users = {
                int(value.strip()) for value in allowed.split(",") if value.strip()
            }
        except ValueError as exc:
            raise ValueError("TRAINCTL_TELEGRAM_ALLOWED_USERS must be comma-separated integers") from exc
    database = os.environ.get("TRAINCTL_DATABASE")
    if database:
        config.storage.database = _path(database)
    log_dir = os.environ.get("TRAINCTL_LOG_DIR")
    if log_dir:
        config.logging.directory = _path(log_dir)
    if not config.monitoring.disk_paths:
        config.monitoring.disk_paths = [config.logging.directory, config.storage.database.parent]
    return config
