import threading
from contextlib import contextmanager


class MRSWLock:
    """Multiple reader, single writer lock. Use `read` and `write` as context managers."""

    __slots__ = ["reader_count", "read_lock", "write_lock"]

    def __init__(self) -> None:
        self.reader_count = 0
        self.read_lock = threading.Lock()
        self.write_lock = threading.Lock()

    @contextmanager
    def read(self):
        with self.read_lock:
            if self.reader_count == 0:
                # first reader acquires write lock, so cannot join if writing
                self.write_lock.acquire()
            self.reader_count += 1

        try:
            yield
        finally:
            with self.read_lock:
                self.reader_count -= 1
                if self.reader_count == 0:
                    self.write_lock.release()

    @contextmanager
    def write(self):
        self.write_lock.acquire()
        try:
            yield
        finally:
            self.write_lock.release()
