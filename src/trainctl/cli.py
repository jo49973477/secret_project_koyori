from __future__ import annotations

import asyncio
import json
import logging
import shlex
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from trainctl.agent.checkpoint_watcher import CheckpointWatcher
from trainctl.agent.resource_monitor import ResourceMonitor
from trainctl.agent.supervisor import Supervisor
from trainctl.config import AppConfig, load_config
from trainctl.core.checkpoint_manager import CheckpointManager
from trainctl.core.event_bus import EventBus
from trainctl.core.metric_store import MetricStore
from trainctl.core.run_manager import RunManager
from trainctl.integrations.generic import GenericTrainingAdapter
from trainctl.integrations.notification import NotificationRouter
from trainctl.integrations.telegram.commands import CommandService
from trainctl.integrations.telegram.formatter import format_status
from trainctl.storage.sqlite import SQLiteStore

app = typer.Typer(
    name="trainctl",
    help="Lightweight deep-learning experiment supervision.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Inspect trainctl configuration.")
app.add_typer(config_app, name="config")
logger = logging.getLogger(__name__)


def _configure_logging(config: AppConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, config.logging.level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _services(config: AppConfig) -> tuple[SQLiteStore, EventBus, RunManager, MetricStore]:
    store = SQLiteStore(config.storage.database)
    bus = EventBus()
    return store, bus, RunManager(store, bus), MetricStore(store, bus)


def _enable_run_notifications(config: AppConfig, bus: EventBus) -> NotificationRouter:
    adapters = []
    if config.telegram.enabled:
        if not config.telegram.token or not config.telegram.allowed_users:
            logger.warning("Telegram notifications enabled but token or allowed user IDs are missing; channel disabled")
        else:
            try:
                from telegram import Bot
                from trainctl.integrations.telegram.bot import TelegramNotifier
                adapters.append(TelegramNotifier(Bot(config.telegram.token), config.telegram.allowed_users))
            except ImportError:
                logger.warning("Telegram is configured but the optional 'telegram' extra is not installed")
    if config.discord.enabled:
        if not config.discord.webhook_url:
            logger.warning("Discord notifications enabled but webhook URL is missing; channel disabled")
        else:
            try:
                import httpx  # noqa: F401  # fail closed at setup if the optional transport is absent
                from trainctl.integrations.discord import DiscordWebhookAdapter
                adapters.append(DiscordWebhookAdapter(config.discord.webhook_url))
            except (ImportError, ValueError) as exc:
                detail = str(exc) if isinstance(exc, ValueError) else "install the optional 'discord' extra (httpx)"
                logger.warning("Discord notification channel disabled: %s", detail)
    return NotificationRouter(bus, adapters)


@app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run_command(
    command: Annotated[
        list[str],
        typer.Argument(help="Command and arguments to supervise (place after --)."),
    ],
    name: Annotated[str, typer.Option("--name", "-n", help="Human-readable run name.")] = "experiment",
    checkpoint_dir: Annotated[
        Path | None, typer.Option("--checkpoint-dir", help="Directory to poll for checkpoints.")
    ] = None,
    total_steps: Annotated[
        int | None, typer.Option("--total-steps", min=1, help="Optional progress denominator.")
    ] = None,
) -> None:
    """Start a child process, capture logs, metrics, and final status."""
    if not command:
        raise typer.BadParameter("Provide a command after `--`, for example: -- python train.py")
    config = load_config()
    _configure_logging(config)
    store, bus, run_manager, metric_store = _services(config)
    notification_router = _enable_run_notifications(config, bus)
    try:
        run = run_manager.create(
            name=name, command=command,
            checkpoint_dir=checkpoint_dir.expanduser().resolve() if checkpoint_dir else None,
            total_steps=total_steps,
        )
        safe_name = "".join(character if character.isalnum() or character in "-_" else "-" for character in name)
        run.log_path = config.logging.directory / f"{safe_name}-{run.run_id}.log"
        store.save_run(run)
        checkpoint_manager = CheckpointManager(store, bus)
        watcher = CheckpointWatcher(
            checkpoint_manager, config.checkpoints.pattern,
            config.checkpoints.poll_interval_seconds,
        )
        supervisor = Supervisor(
            run_manager, metric_store, GenericTrainingAdapter(), checkpoint_watcher=watcher
        )
        typer.echo(f"Starting {name!r} ({run.run_id}): {shlex.join(command)}")
        typer.echo(f"Log: {run.log_path}")
        async def supervise_and_close():
            try:
                return await supervisor.supervise(run)
            finally:
                await notification_router.close()

        result = asyncio.run(supervise_and_close())
        typer.echo(format_status(result))
        if result.exit_code not in (0, None):
            raise typer.Exit(code=result.exit_code if 0 < result.exit_code < 256 else 1)
        if result.exit_code is None:
            raise typer.Exit(code=1)
    finally:
        store.close()


@app.command()
def status() -> None:
    """Show the most recently created run."""
    config = load_config()
    store, _, run_manager, _ = _services(config)
    try:
        run_manager.reconcile()
        typer.echo(format_status(run_manager.latest()))
    finally:
        store.close()


@app.command()
def daemon() -> None:
    """Run the user-level Telegram frontend and coordinator."""
    config = load_config()
    _configure_logging(config)
    if not config.telegram.token:
        raise typer.BadParameter("TRAINCTL_TELEGRAM_TOKEN is required to start the Telegram daemon")
    if not config.telegram.allowed_users:
        raise typer.BadParameter("TRAINCTL_TELEGRAM_ALLOWED_USERS must contain at least one user ID")
    store, bus, run_manager, _ = _services(config)
    run_manager.reconcile()
    resources = ResourceMonitor(config.monitoring.disk_warning_percent)
    service = CommandService(store, resources, config.monitoring.disk_paths)
    try:
        from trainctl.integrations.telegram.bot import TelegramAdapter, TelegramDependencyError

        adapter = TelegramAdapter(config.telegram.token, config.telegram.allowed_users, service, bus)
    except TelegramDependencyError as exc:
        store.close()
        raise typer.BadParameter(str(exc)) from exc
    typer.echo("trainctl daemon started; Telegram long polling is active")
    try:
        adapter.run_polling()
    finally:
        store.close()


@config_app.command("show")
def config_show() -> None:
    """Print effective non-secret configuration."""
    config = load_config()
    payload = {
        "source": str(config.source) if config.source else None,
        "telegram": {
            "enabled": config.telegram.enabled,
            "token_configured": bool(config.telegram.token),
            "allowed_users": sorted(config.telegram.allowed_users),
        },
        "discord": {"enabled": config.discord.enabled, "webhook_configured": bool(config.discord.webhook_url)},
        "storage": {"database": str(config.storage.database)},
        "logging": {"directory": str(config.logging.directory), "level": config.logging.level},
        "monitoring": {
            "disk_warning_percent": config.monitoring.disk_warning_percent,
            "disk_paths": [str(path) for path in config.monitoring.disk_paths],
        },
        "checkpoints": asdict(config.checkpoints),
    }
    typer.echo(json.dumps(payload, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
