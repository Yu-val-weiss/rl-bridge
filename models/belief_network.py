import torch
from torch import nn

from utils import get_device

from .mlp import MLP


class BeliefNetwork(nn.Module):
    def __init__(
        self, output_size=24, input_size=84, hidden_size=248, dropout_rate=0.1
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 4 layer multilayer perceptron, with gelu activation and dropout
        self.mlp = MLP(
            input_size,
            hidden_size,
            output_size,
            4,
            use_dropout=True,
            dropout_rate=dropout_rate,
        )

        self.sigmoid = nn.Sigmoid()

        self.to(get_device())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)

        return self.sigmoid(x)
