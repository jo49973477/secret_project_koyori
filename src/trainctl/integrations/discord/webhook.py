from __future__ import annotations

import asyncio
import logging
import math
from pathlib import Path
from urllib.parse import urlsplit

from trainctl.core.events import Event

from .formatter import DiscordMessage, format_event

logger = logging.getLogger(__name__)


class DiscordWebhookAdapter:
    """Outgoing Discord webhook transport. No bot token or inbound API is used."""

    def __init__(self, webhook_url: str, *, client=None, retries: int = 2, timeout: float = 8.0) -> None:
        parsed = urlsplit(webhook_url) if isinstance(webhook_url, str) else None
        segments = parsed.path.strip("/").split("/") if parsed else []
        if (
            parsed is None
            or parsed.scheme != "https"
            or parsed.hostname not in {"discord.com", "discordapp.com"}
            or segments[:2] != ["api", "webhooks"]
            or len(segments) != 4
            or not segments[2]
            or not segments[3]
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Discord webhook URL must be an HTTPS Discord webhook URL")
        self._url = webhook_url
        self._client = client
        self._owns_client = client is None
        self.retries = max(0, min(retries, 4))
        self.timeout = timeout

    async def _http(self):
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise RuntimeError("Discord notifications require the optional 'discord' extra (httpx)") from exc
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def __call__(self, event: Event) -> None:
        message = format_event(event)
        if message is not None:
            await self.send(message)

    async def send_message(self, message: str) -> None:
        await self.send(DiscordMessage(content=message))

    async def send_file(self, path: Path, caption: str | None = None) -> None:
        path = Path(path)
        if path.suffix.lower() != ".png":
            raise ValueError("Discord notification attachments currently support PNG files only")
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = DiscordMessage(content=caption).payload()
        import json
        client = await self._http()
        with path.open("rb") as handle:
            await self._request(client, files={"files[0]": (path.name, handle, "image/png")}, data={"payload_json": json.dumps(payload)})

    async def send(self, message: DiscordMessage) -> None:
        client = await self._http()
        await self._request(client, json=message.payload())

    async def _request(self, client, **kwargs) -> None:
        for attempt in range(self.retries + 1):
            try:
                response = await asyncio.wait_for(client.post(self._url, **kwargs), timeout=self.timeout)
                status = response.status_code
                if 200 <= status < 300:
                    return
                if status == 429:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after is not None else float(response.json().get("retry_after", 1))
                    except (ValueError, TypeError, AttributeError):
                        delay = 1.0
                    if attempt < self.retries:
                        await asyncio.sleep(min(max(delay, 0), 5.0))
                        continue
                    logger.warning("Discord webhook rate limited; notification was not delivered")
                    return
                if status >= 500 and attempt < self.retries:
                    await asyncio.sleep(min(0.25 * math.pow(2, attempt), 2.0))
                    continue
                if status in (401, 403, 404):
                    logger.error("Discord webhook rejected notification (HTTP %s); check webhook configuration", status)
                    return
                logger.error("Discord webhook notification failed with HTTP %s", status)
                return
            except Exception as exc:
                # Catch client/network exceptions without allowing any transport fault into supervision.
                if attempt < self.retries:
                    await asyncio.sleep(min(0.25 * math.pow(2, attempt), 2.0))
                    continue
                logger.warning("Discord webhook delivery failed after retries (%s)", type(exc).__name__)
                return

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
