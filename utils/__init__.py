import torch

from .data import SelfPlayConfig, load_config
from .mrsw import MRSWLock


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


__all__ = ["get_device", "load_config", "MRSWLock", "SelfPlayConfig"]
