# Telegram frontend

## Setup

1. Install `trainctl` with the `telegram` extra.
2. Create a bot using Telegram's official BotFather flow.
3. Put its token in `TRAINCTL_TELEGRAM_TOKEN`.
4. Put your numeric user ID in `TRAINCTL_TELEGRAM_ALLOWED_USERS`.
5. Start one `trainctl daemon` per server/user.

Do not paste a real token into TOML, documentation, logs, or source control. An
empty allowlist denies everyone, and unauthorized requests receive no data.

## Commands

- `/status` — latest run status, progress, loss, elapsed time, PID, and exit code
- `/gpu` — all detected NVIDIA GPUs plus system RAM; safe on CPU-only hosts
- `/tail [N]` — latest 1–200 lines from the latest run log
- `/disk` — unique filesystems behind configured paths and warning markers

The registered `/plot`, `/eval`, `/best`, `/compare`, `/pause`, and `/resume`
commands currently return a planned-feature message.

`trainctl run` publishes lifecycle and checkpoint events. When it has a token,
allowlist, and the optional dependency, Telegram subscribers send notifications.
The initial in-process bus does not yet relay events from arbitrary external
processes to an already-running daemon; a persistent outbox is planned.

Telegram never provides `/shell`, `/exec`, or a free-form remote `/run` command.
