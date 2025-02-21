import threading
from contextlib import contextmanager


class MRSWLock:
    """Fair multiple-reader, single-writer lock.
    Ensures that if readers are waiting, the writer cannot re-lock until the readers have left."""

    __slots__ = [
        "reader_count",
        "read_lock",
        "reader_waiting",
        "reader_wait_lock",
        "write_lock",
    ]

    def __init__(self) -> None:
        self.reader_count = 0
        self.read_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.reader_waiting = 0  # track number of waiting readers
        self.reader_wait_lock = threading.Lock()

    @contextmanager
    def read(self):
        with self.reader_wait_lock:
            self.reader_waiting += 1  # indicate a reader is waiting

        with self.read_lock:
            if self.reader_count == 0:
                self.write_lock.acquire()  # first reader acquires write lock
            self.reader_count += 1

        with self.reader_wait_lock:
            self.reader_waiting -= 1  # reader has acquired lock, no longer waiting

        try:
            yield
        finally:
            with self.read_lock:
                self.reader_count -= 1
                if self.reader_count == 0:
                    self.write_lock.release()

    @contextmanager
    def write(self):
        while True:
            with self.reader_wait_lock:
                if self.reader_waiting == 0:
                    # if no readers are waiting, can safely acquire the write lock
                    break
                # otherwise, spin until no readers are waiting

        self.write_lock.acquire()
        try:
            yield
        finally:
            self.write_lock.release()
