import torch

from models.mlp import MLP
from utils import CloneableNetwork


class ValueNetwork(CloneableNetwork["ValueNetwork"]):
    def __init__(self, input_size=636, hidden_size=2048):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # 4 layer multilayer perceptron, output size of 1
        self.mlp = MLP(self.input_size, self.hidden_size, 1, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)

    def clone(self) -> "ValueNetwork":
        new_model = ValueNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        )
        new_model.load_state_dict(self.state_dict())
        return new_model
