import torch
from torch import nn

from utils import get_device

from .mlp import MLP


class BeliefNetwork(nn.Module):
    def __init__(self, output_size=32, input_size=480, hidden_size=2048):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 4 layer multilayer perceptron, with gelu activation and softmax
        self.mlp = MLP(input_size, hidden_size, output_size, 4)

        self.softmax = nn.Softmax(dim=2)

        self.to(get_device())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)

        # Specific to Tiny Bridge, make sure that distribution over cards of 1 player sum to 1.
        x = x.view(x.size(0), 3, 8)

        return self.softmax(x)
