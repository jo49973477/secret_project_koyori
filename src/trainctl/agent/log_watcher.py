from __future__ import annotations

from collections import deque


class LogBuffer:
    def __init__(self, max_lines: int = 200) -> None:
        self._lines: deque[str] = deque(maxlen=max_lines)

    def append(self, line: str) -> None:
        self._lines.append(line.rstrip("\r\n"))

    def tail(self, count: int = 20) -> list[str]:
        if count <= 0:
            return []
        return list(self._lines)[-count:]

