import torch
from torch import nn

from models.mlp import MLP


class ValueNetwork(nn.Module):
    def __init__(self, input_size=636, hidden_size=2048):
        super().__init__()
        # 4 layer multilayer perceptron, with gelu activation and softmax
        self.mlp = MLP(input_size, hidden_size, 1, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)
