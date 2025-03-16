import torch

from .actions import mask_action_probs, random_argmax
from .config import SelfPlayConfig, load_self_play_config
from .mrsw import MRSWLock


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


__all__ = [
    "get_device",
    "load_self_play_config",
    "mask_action_probs",
    "MRSWLock",
    "random_argmax",
    "SelfPlayConfig",
]
