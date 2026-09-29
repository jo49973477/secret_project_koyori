from pathlib import Path

from trainctl.agent.resource_monitor import ResourceMonitor


async def test_disk_summary_deduplicates_filesystems(tmp_path) -> None:
    monitor = ResourceMonitor()
    summaries = await monitor.get_disk_summary([tmp_path, Path(tmp_path) / "missing"])
    assert len(summaries) == 1
    assert summaries[0].total_bytes > 0

