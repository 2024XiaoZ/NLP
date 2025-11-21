"""计时与时间工具。"""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Optional


def now_ms() -> int:
    """返回当前毫秒时间戳。"""

    return int(time.perf_counter() * 1000)


@dataclass
class Timer(AbstractContextManager["Timer"]):
    """简单的上下文计时器，用于延迟统计。"""

    start_ms: Optional[int] = field(default=None)
    end_ms: Optional[int] = field(default=None)

    def __enter__(self) -> "Timer":
        self.start_ms = now_ms()
        self.end_ms = None
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:  # type: ignore[override]
        self.stop()

    def stop(self) -> int:
        """结束计时并返回耗时。"""

        if self.start_ms is None:
            raise RuntimeError("Timer 尚未启动。")
        self.end_ms = now_ms()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> int:
        """以毫秒返回耗时；若尚未停止，则以当前时间计算。"""

        if self.start_ms is None:
            return 0
        end = self.end_ms or now_ms()
        return max(0, end - self.start_ms)


