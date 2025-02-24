import threading
import time

import pytest

from utils import MRSWLock


class Dummy:
    lock: MRSWLock
    shared_resource: int

    def __init__(self) -> None:
        self.shared_resource = 0
        self.lock = MRSWLock()


@pytest.fixture
def dummy_resource():
    dummy = Dummy()
    yield dummy
    del dummy


def reader(d: Dummy, obs: list[int]):
    with d.lock.read():
        time.sleep(0.1)  # simulate reading work
        obs.append(d.shared_resource)


def writer(d: Dummy) -> None:
    with d.lock.write():
        time.sleep(0.1)  # simulate writing work
        d.shared_resource = 1


def test_multiple_readers_block_writer(dummy_resource):
    obs = []
    threads = [
        threading.Thread(target=reader, args=(dummy_resource, obs)) for _ in range(5)
    ]
    for t in threads:
        t.start()
    time.sleep(0.1)
    tw = threading.Thread(target=writer, args=(dummy_resource,))
    tw.start()
    for t in threads:
        t.join()
    tw.join()

    assert obs == [0] * 5


def test_writer_blocks_readers(dummy_resource):
    obs = []
    tw = threading.Thread(target=writer, args=(dummy_resource,))
    tw.start()
    threads = [
        threading.Thread(target=reader, args=(dummy_resource, obs)) for _ in range(5)
    ]
    time.sleep(0.1)
    for t in threads:
        t.start()
    tw.join()
    for t in threads:
        t.join()

    assert obs == [1] * 5


def test_writer_cannot_rewrite_if_reader_waiting():
    lock = MRSWLock()
    writer_attempts = []
    reads = []

    def writer():
        with lock.write():
            writer_attempts.append("first_write")
            time.sleep(0.1)  # simulate training time
        with lock.write():
            writer_attempts.append("second_write")
            time.sleep(0.1)  # simulate training time

    def reader():
        with lock.read():
            reads.append(writer_attempts[-1])

    writer_thread = threading.Thread(target=writer)
    num_threads = 5
    reader_threads = [threading.Thread(target=reader) for _ in range(num_threads)]

    writer_thread.start()
    for t in reader_threads:
        t.start()

    writer_thread.join()
    for t in reader_threads:
        t.join()

    assert reads == ["first_write"] * num_threads, (
        "Writer was able to rewrite before reader finished!"
    )
