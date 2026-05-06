import time
from collections import defaultdict, deque
from typing import Deque, Dict, Hashable


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: Dict[Hashable, Deque[float]] = defaultdict(deque)
        self._last_cleanup = time.monotonic()

    def allow(self, key: Hashable) -> bool:
        if self.limit <= 0:
            return True

        now = time.monotonic()
        cutoff = now - self.window_seconds
        events = self._events[key]
        while events and events[0] <= cutoff:
            events.popleft()

        if len(events) >= self.limit:
            self._cleanup(now, cutoff)
            return False

        events.append(now)
        self._cleanup(now, cutoff)
        return True

    def _cleanup(self, now: float, cutoff: float) -> None:
        if now - self._last_cleanup < self.window_seconds:
            return
        self._last_cleanup = now
        empty_keys = []
        for key, events in self._events.items():
            while events and events[0] <= cutoff:
                events.popleft()
            if not events:
                empty_keys.append(key)
        for key in empty_keys:
            self._events.pop(key, None)
