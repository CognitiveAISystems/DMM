"""No-op hook for the observation generator's optional timing scopes."""

from contextlib import contextmanager


class NullProfiler:
    @contextmanager
    def time(self, name: str):
        yield
