from collections import OrderedDict
from typing import Callable

import torch
from torch import nn

ModuleFactory = Callable[[], nn.Module]


class MLP(nn.Module):
    """Multilayer perceptron"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        *,
        activation_factory: ModuleFactory = nn.GELU,
    ):
        super().__init__()
        assert num_layers >= 2
        layers = OrderedDict()
        layers["fc_in"] = nn.Linear(input_size, hidden_size)
        layers["activ_in"] = nn.GELU()
        for i in range(2, num_layers):
            layers[f"fc_{i}"] = nn.Linear(hidden_size, hidden_size)
            layers[f"activ_{i}"] = activation_factory()
        layers["fc_out"] = nn.Linear(hidden_size, output_size)

        self.seq_model = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq_model(x)
