import time
from collections import Counter
from typing import Dict


class Metrics:
    def __init__(self) -> None:
        self.started_monotonic = time.monotonic()
        self.counters: Counter[str] = Counter()
        self.active_workers = 0

    def inc(self, key: str, amount: int = 1) -> None:
        self.counters[key] += amount

    def set_active_workers(self, value: int) -> None:
        self.active_workers = max(0, value)

    def uptime_seconds(self) -> int:
        return int(time.monotonic() - self.started_monotonic)

    def snapshot(self, queue_size: int = 0) -> Dict[str, int]:
        data = dict(self.counters)
        data["uptime_seconds"] = self.uptime_seconds()
        data["queue_size"] = queue_size
        data["active_workers"] = self.active_workers
        return data


metrics = Metrics()
