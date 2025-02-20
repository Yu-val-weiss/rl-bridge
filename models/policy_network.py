import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):

    def __init__(self, output_size, input_size=480, hidden_size=2048):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.gelu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
