# Configuration

Configuration priority is: CLI runtime options, environment variables, TOML,
then defaults. Paths expand `~` and become absolute.

The default file is `~/.config/trainctl/config.toml`:

```toml
[telegram]
enabled = true
allowed_users = [123456789]

[storage]
database = "~/.local/share/trainctl/trainctl.db"

[logging]
directory = "~/.local/share/trainctl/logs"
level = "INFO"

[monitoring]
disk_warning_percent = 90
disk_paths = ["~/datasets", "~/checkpoints"]

[checkpoints]
poll_interval_seconds = 5
pattern = '(?:checkpoint|ckpt)[-_]?(?P<step>\d+)'
```

Environment variables:

| Variable | Purpose |
| --- | --- |
| `TRAINCTL_TELEGRAM_TOKEN` | Bot token; intentionally not read from TOML |
| `TRAINCTL_TELEGRAM_ALLOWED_USERS` | Comma-separated numeric user IDs |
| `TRAINCTL_CONFIG` | Alternate TOML path |
| `TRAINCTL_DATABASE` | SQLite path override |
| `TRAINCTL_LOG_DIR` | training log directory override |
| `TRAINCTL_HOME` | base directory used while calculating defaults |

`trainctl config show` prints effective non-secret configuration and only says
whether a token is configured. It never prints the token.
