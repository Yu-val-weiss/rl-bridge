from abc import abstractmethod
from typing import Generic, TypeVar

from torch import nn

T = TypeVar("T", bound="CloneableNetwork")


class CloneableNetwork(nn.Module, Generic[T]):
    """Abstract class that defines a clone interface for deep copying neural network models."""

    @abstractmethod
    def clone(self) -> T:
        """Creates a deep copy of the model with the same architecture and parameters.

        Returns:
            T: A deep copy of the model
        """
        pass
