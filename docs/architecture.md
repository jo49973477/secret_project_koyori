# Architecture

## Design boundary

`trainctl` separates experiment orchestration from presentation. The core knows
about runs, events, metrics, checkpoints, evaluation records, and state. It does
not know about Telegram messages. Telegram handlers authorize a caller, invoke
`CommandService`, and format returned domain objects.

```text
Telegram / CLI / future adapter
              │
              ▼
       typed command/query layer
              │
   ┌──────────┴────────────────────┐
   │ RunManager   MetricStore      │
   │ CheckpointManager  EventBus   │
   │ ResourceMonitor   EvalManager │
   └──────────┬────────────────────┘
              │
        Supervisor / watcher
              │
     python / torchrun / deepspeed
```

## Components

### Run

`Run` is the central experiment record: stable ID, display name, argv command,
PID/PGID, lifecycle status, progress, timestamps, log/checkpoint paths, exit
code, and a cache of latest metric values. `RunManager` enforces explicit state
transitions and persists every mutation.

### Supervisor

`Supervisor` uses `asyncio.create_subprocess_exec` with an explicit argument
list. On POSIX it creates a new process session, captures combined stdout and
stderr in a run-specific file, retains a bounded tail, feeds each line to the
selected training adapter, and maps exit code zero/nonzero to completion/failure.

### EventBus

The first bus is in-process async pub/sub. Typed events keep publishers
independent from notifier, registry, and future auto-evaluation subscribers. A
failed handler is logged and isolated. Persistent cross-process event delivery
is intentionally deferred.

### MetricStore

The generic adapter recognizes conservative key/value patterns. `MetricStore`
writes time-series points through the state-store interface and publishes
`MetricUpdated`. SQLite details are not exposed to callers.

```text
Training process → stdout → Supervisor → GenericTrainingAdapter
                                      → MetricStore → SQLite
                                                   → MetricUpdated
```

### CheckpointManager

The manager provides `latest`, `by_step`, `list`, and `best`. The polling watcher
accepts a configurable regex with an optional named `step` group, so it does not
hardcode a framework convention.

```text
CheckpointWatcher → CheckpointSaved ─┬→ SQLite registry
                                     ├→ Telegram notifier
                                     └→ future AutoEval
```

### ResourceMonitor

GPU collection prefers optional NVML and falls back to a bounded `nvidia-smi`
subprocess. Missing drivers and CPU-only machines return a useful unavailable
result rather than crashing. RAM and disk data use `psutil`.

### TelegramAdapter

The adapter owns long polling, the user allowlist, command routing, formatting,
and event notification. `/status`, `/gpu`, `/tail`, and `/disk` delegate to
services. No Telegram string is passed to a shell. Planned commands are explicit
routes rather than a generic execution endpoint.

### EvalManager and ProcessController

`EvalManager` currently creates persisted job models only from trusted local
command templates. Execution is planned. `ProcessController` contains local
POSIX signal primitives, but they are not remotely exposed: `SIGSTOP` can break
or deadlock collective distributed workloads. A cooperative pause protocol at
training-step boundaries is the intended production design.

### Storage and restart behavior

SQLite stores runs, metric points, checkpoints, and evaluation records. WAL mode
allows status readers while a run writes. Startup reconciliation probes a saved
PID before treating an active record as alive; missing PIDs become failed. PID
reuse and reattaching log pipes are not solved in this alpha.
