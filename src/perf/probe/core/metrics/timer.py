
import time
from contextlib import contextmanager
from typing import Optional


class Timer:

    def __init__(self):
        self.start: float = 0.0
        self.end: float = 0.0
        self.duration: float = 0.0

    def __enter__(self) -> 'Timer':
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end = time.perf_counter()
        self.duration = self.end - self.start


@contextmanager
def timer():
    timer_obj = Timer()
    timer_obj.__enter__()
    try:
        yield timer_obj
    finally:
        timer_obj.__exit__()


def measure_time(func):
    def wrapper(*args, **kwargs):
        with Timer() as t:
            result = func(*args, **kwargs)
        print(f"{func.__name__} elapsed time: {t.duration:.4f} seconds")
        return result
    return wrapper
