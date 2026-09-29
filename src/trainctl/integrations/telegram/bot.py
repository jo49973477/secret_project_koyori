from __future__ import annotations

import logging
from collections.abc import Callable

from trainctl.core.event_bus import EventBus
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

from .commands import CommandService, is_authorized
from .formatter import format_disk, format_event, format_gpu, format_status, format_tail

logger = logging.getLogger(__name__)


class TelegramDependencyError(RuntimeError):
    pass


def _telegram_types() -> tuple[type, type]:
    try:
        from telegram.ext import Application, CommandHandler
    except ImportError as exc:
        raise TelegramDependencyError(
            "Telegram support is optional; install it with `pip install 'trainctl[telegram]'`"
        ) from exc
    return Application, CommandHandler


class TelegramNotifier:
    def __init__(self, bot: object, recipients: set[int]) -> None:
        self.bot = bot
        self.recipients = recipients

    async def __call__(self, event: Event) -> None:
        message = format_event(event)
        if not message:
            return
        for chat_id in self.recipients:
            try:
                await self.bot.send_message(chat_id=chat_id, text=message[-4000:])  # type: ignore[attr-defined]
            except Exception:
                logger.exception("Could not send Telegram event notification to %s", chat_id)

    async def close(self) -> None:
        # The Telegram application owns the shared bot lifecycle.
        return None


def subscribe_notifications(event_bus: EventBus, notifier: TelegramNotifier) -> None:
    for event_type in (
        RunStarted, RunCompleted, RunFailed, CheckpointSaved,
        EvalCompleted, EvalFailed, DiskLow, GpuWarning,
    ):
        event_bus.subscribe(event_type, notifier)


class TelegramAdapter:
    def __init__(
        self, token: str, allowed_users: set[int], service: CommandService,
        event_bus: EventBus,
    ) -> None:
        if not allowed_users:
            raise ValueError("Telegram is enabled but no allowed users are configured")
        Application, CommandHandler = _telegram_types()
        self.allowed_users = allowed_users
        self.service = service
        self.application = Application.builder().token(token).build()
        handlers: dict[str, Callable] = {
            "status": self.status, "gpu": self.gpu, "tail": self.tail, "disk": self.disk,
            "plot": self.planned, "eval": self.planned, "best": self.planned,
            "compare": self.planned, "pause": self.planned, "resume": self.planned,
        }
        for command, callback in handlers.items():
            self.application.add_handler(CommandHandler(command, callback))
        subscribe_notifications(
            event_bus, TelegramNotifier(self.application.bot, allowed_users)
        )

    async def _authorized(self, update: object) -> bool:
        user = getattr(update, "effective_user", None)
        if not is_authorized(getattr(user, "id", None), self.allowed_users):
            message = getattr(update, "effective_message", None)
            if message:
                await message.reply_text("Unauthorized.")
            return False
        return True

    async def status(self, update: object, context: object) -> None:
        if await self._authorized(update):
            await update.effective_message.reply_text(format_status(self.service.status()))  # type: ignore[attr-defined]

    async def gpu(self, update: object, context: object) -> None:
        if await self._authorized(update):
            await update.effective_message.reply_text(format_gpu(await self.service.gpu()))  # type: ignore[attr-defined]

    async def disk(self, update: object, context: object) -> None:
        if await self._authorized(update):
            await update.effective_message.reply_text(format_disk(await self.service.disk()))  # type: ignore[attr-defined]

    async def tail(self, update: object, context: object) -> None:
        if not await self._authorized(update):
            return
        args = getattr(context, "args", [])
        try:
            count = int(args[0]) if args else 20
        except ValueError:
            await update.effective_message.reply_text("Usage: /tail [1-200]")  # type: ignore[attr-defined]
            return
        run, lines = self.service.tail(count)
        await update.effective_message.reply_text(format_tail(run, lines))  # type: ignore[attr-defined]

    async def planned(self, update: object, context: object) -> None:
        if await self._authorized(update):
            await update.effective_message.reply_text("This structured command is planned, but not implemented yet.")  # type: ignore[attr-defined]

    def run_polling(self) -> None:
        self.application.run_polling(allowed_updates=None)
