import copy
from dataclasses import asdict

import torch
from torch import nn

from utils import get_device
from utils.config import PolicyNetConfig

from .mlp import MLP
from .network import Network


class PolicyNetwork(Network["PolicyNetwork"]):
    def __init__(self, output_size: int, input_size=480, hidden_size=2048):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 4 layer multilayer perceptron, with gelu activation and softmax
        self.mlp = MLP(input_size, hidden_size, output_size, 4)

        self.softmax = nn.Softmax(dim=-1)

        self.to(get_device())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return self.softmax(x)

    def clone(self) -> "PolicyNetwork":
        new_model = PolicyNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()))
        return new_model

    def get_init_config(self) -> dict[str, int]:
        return dict(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
        )

    @classmethod
    def from_dataclass(cls, config: PolicyNetConfig) -> "PolicyNetwork":
        """Creates policy network from config dataclass"""
        return cls(**asdict(config))
