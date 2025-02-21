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
