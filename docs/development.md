# Development

Python 3.10+ and `uv` are recommended:

```bash
uv sync
uv run pytest
uv run trainctl --help
uv build
```

For all optional adapters:

```bash
uv sync --all-extras
```

Editable installation is useful when invoking the command outside `uv run`:

```bash
uv pip install -e .
```

Run the end-to-end smoke example with:

```bash
uv run trainctl run --name demo -- python examples/dummy_train.py
uv run trainctl status
```

Tests use temporary SQLite databases, a tiny Python child process, and mocked or
fallback-safe boundaries. They require neither Telegram, a GPU, nor torchrun.

New framework support should implement `TrainingAdapter`; it must not leak
framework types into the core. New frontends should consume command services
and domain models rather than duplicate process or resource logic.
