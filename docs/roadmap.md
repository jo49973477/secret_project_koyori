# Roadmap

## v0.1 — supervision foundation

- argv-based `trainctl run`
- SQLite run state and metrics
- bounded logs and generic metric parsing
- typed events, checkpoint detection, and lifecycle notifications
- Telegram `/status`, `/gpu`, `/tail`, `/disk`

## v0.2 — observability

- persistent event outbox consumed by the daemon
- metric plot rendering and `/plot`
- periodic disk-space and GPU-health policies
- checkpoint metric association and `/best`
- more robust process identity/restart reconciliation

## v0.3 — evaluation

- execute trusted local evaluation templates
- evaluation concurrency and GPU assignment limits
- `/eval latest` and `/compare 1000 1500 2000`
- evaluation result attachment to checkpoints

## v0.4 — broader control

- cooperative pause/resume hooks safe at step boundaries
- multiple active-run selection
- multi-server aggregation without heavy infrastructure
- Discord, Slack, and Web frontends

Potential framework adapters include Hugging Face Trainer, PyTorch Lightning,
and GR00T. They will remain optional and cannot become core dependencies.
