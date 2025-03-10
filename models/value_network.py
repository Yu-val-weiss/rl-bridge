import torch

from utils import get_device

from .mlp import MLP
from .network import Network


class ValueNetwork(Network["ValueNetwork"]):
    def __init__(self, input_size=636, hidden_size=2048):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # 4 layer multilayer perceptron, output size of 1
        self.mlp = MLP(self.input_size, self.hidden_size, 1, 4)

        self.to(get_device())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)

    def clone(self) -> "ValueNetwork":
        new_model = ValueNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        )
        new_model.load_state_dict(self.state_dict())
        return new_model

    def get_init_config(self) -> dict[str, int]:
        return dict(input_size=self.input_size, hidden_size=self.hidden_size)
