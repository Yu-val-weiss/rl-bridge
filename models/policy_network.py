import torch
from torch import nn

from models.mlp import MLP


class PolicyNetwork(nn.Module):
    def __init__(self, output_size, input_size=480, hidden_size=2048):
        super().__init__()
        # 4 layer multilayer perceptron, with gelu activation and softmax
        self.mlp = MLP(input_size, hidden_size, output_size, 4)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return self.softmax(x)
