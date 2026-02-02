'''import time
from collections import deque

class RollingEventCounter:
    def __init__(self, window_seconds: float):
        self.window = window_seconds
        self.events = deque()

    def add_event(self, t: float):
        self.events.append(t)
        self._trim(t)

    def count(self, t: float) -> int:
        self._trim(t)
        return len(self.events)

    def _trim(self, t: float):
        while self.events and (t - self.events[0]) > self.window:
            self.events.popleft()

class TimerState:
    def __init__(self):
        self.last_t = None
        self.accum = 0.0

    def update(self, condition: bool, t: float):
        if self.last_t is None:
            self.last_t = t
            self.accum = 0.0
            return self.accum

        dt = max(0.0, t - self.last_t)
        self.last_t = t

        if condition:
            self.accum += dt
        else:
            self.accum = 0.0

        return self.accum

def now():
    return time.time()'''
import time

def now() -> float:
    return time.time()

class TimerState:
    """
    Tracks how long a boolean condition has been continuously True.
    If condition is False, it resets to 0.
    """
    def __init__(self):
        self.start_t = None

    def update(self, condition: bool, t: float) -> float:
        if condition:
            if self.start_t is None:
                self.start_t = t
            return t - self.start_t
        else:
            self.start_t = None
            return 0.0

