import torch

from .cloneable import CloneableNetwork
from .mrsw import MRSWLock


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


__all__ = ["CloneableNetwork", "get_device", "MRSWLock"]
