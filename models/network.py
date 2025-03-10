from abc import abstractmethod
from typing import Any, Generic, TypeVar

from torch import nn

T = TypeVar("T", bound="Network")


class Network(nn.Module, Generic[T]):
    """Abstract class that defines a clone interface for deep copying neural network models."""

    @abstractmethod
    def clone(self) -> T:
        """Creates a deep copy of the model with the same architecture and parameters.

        Returns:
            T: A deep copy of the model
        """
        pass

    @abstractmethod
    def get_init_config(self) -> dict[str, Any]:
        """Gets the config needed to initialise the class"""
        pass
