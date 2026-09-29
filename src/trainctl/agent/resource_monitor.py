from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass(frozen=True, slots=True)
class GpuInfo:
    index: int
    name: str
    utilization_percent: float | None
    memory_used_mb: float | None
    memory_total_mb: float | None
    temperature_c: float | None
    power_w: float | None


@dataclass(frozen=True, slots=True)
class ResourceSummary:
    gpus: tuple[GpuInfo, ...]
    ram_used_percent: float
    gpu_backend: str | None
    gpu_error: str | None = None


@dataclass(frozen=True, slots=True)
class DiskInfo:
    path: Path
    total_bytes: int
    used_bytes: int
    free_bytes: int
    percent_used: float
    warning: bool


class ResourceMonitor:
    def __init__(self, disk_warning_percent: float = 90.0) -> None:
        self.disk_warning_percent = disk_warning_percent

    async def get_gpu_summary(self) -> ResourceSummary:
        # The bounded nvidia-smi fallback can take a few seconds, but keeping
        # collection inline avoids creating persistent executor threads in a
        # tiny, otherwise-idle daemon. This can move to a dedicated executor
        # when periodic sampling is introduced.
        return self._collect_resources()

    def _collect_resources(self) -> ResourceSummary:
        ram = psutil.virtual_memory().percent
        try:
            gpus = self._from_nvml()
            return ResourceSummary(tuple(gpus), ram, "nvml")
        except Exception:
            # pynvml uses its own exception hierarchy for missing drivers,
            # permissions, and unsupported devices. All should fall back.
            pass
        try:
            gpus = self._from_nvidia_smi()
            return ResourceSummary(tuple(gpus), ram, "nvidia-smi")
        except (FileNotFoundError, subprocess.SubprocessError, ValueError) as exc:
            return ResourceSummary((), ram, None, f"GPU telemetry unavailable: {exc}")

    def _from_nvml(self) -> list[GpuInfo]:
        try:
            import pynvml
        except ImportError:
            raise
        try:
            pynvml.nvmlInit()
            output: list[GpuInfo] = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode(errors="replace")
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
                except pynvml.NVMLError:
                    power = None
                output.append(GpuInfo(
                    index=index, name=str(name), utilization_percent=float(utilization.gpu),
                    memory_used_mb=memory.used / 1024**2, memory_total_mb=memory.total / 1024**2,
                    temperature_c=float(pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )), power_w=power,
                ))
            return output
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _from_nvidia_smi(self) -> list[GpuInfo]:
        executable = shutil.which("nvidia-smi")
        if not executable:
            raise FileNotFoundError("neither NVML nor nvidia-smi is available")
        fields = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        result = subprocess.run(
            [executable, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        output: list[GpuInfo] = []
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            if len(row) != 7:
                continue
            number = lambda value: None if value.strip().lower() in {"n/a", "[not supported]"} else float(value)
            output.append(GpuInfo(
                index=int(row[0]), name=row[1], utilization_percent=number(row[2]),
                memory_used_mb=number(row[3]), memory_total_mb=number(row[4]),
                temperature_c=number(row[5]), power_w=number(row[6]),
            ))
        return output

    async def get_disk_summary(self, paths: list[Path]) -> list[DiskInfo]:
        return self._collect_disks(paths)

    def _collect_disks(self, paths: list[Path]) -> list[DiskInfo]:
        output: list[DiskInfo] = []
        seen_devices: set[int] = set()
        for configured in paths:
            path = configured.expanduser()
            existing = path
            while not existing.exists() and existing != existing.parent:
                existing = existing.parent
            try:
                device = existing.stat().st_dev
                if device in seen_devices:
                    continue
                seen_devices.add(device)
                usage = psutil.disk_usage(existing)
            except OSError:
                continue
            output.append(DiskInfo(
                path=path, total_bytes=usage.total, used_bytes=usage.used,
                free_bytes=usage.free, percent_used=usage.percent,
                warning=usage.percent >= self.disk_warning_percent,
            ))
        return output
