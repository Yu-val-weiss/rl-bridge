import copy
import functools
from typing import Any, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchrl.data import PrioritizedReplayBuffer

from models.policy_network import PolicyNetwork
from models.value_network import ValueNetwork
from utils import get_device
from utils.mrsw import MRSWLock


class Learner:
    def __init__(
        self,
        policy_net: nn.Module,
        value_net: nn.Module,
        replay_buffer: PrioritizedReplayBuffer,
        lr_pol: float,
        lr_val: float,
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.replay_buffer = replay_buffer
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_pol)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_val)
        self.device = get_device()

    def train_step(self, batch_size: int, beta: float):
        # Sample a mini-batch of transitions from the replay buffer
        batch = self.replay_buffer.sample(batch_size, return_info=True)
        states, actions, rewards, old_policies, old_values = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_policies = torch.tensor(old_policies, dtype=torch.float32)
        old_values = torch.tensor(old_values, dtype=torch.float32)

        # Compute action probabilities and log probabilities
        action_probs = self.policy_net(states)
        action_log_probs = torch.log(
            action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Value estimates
        values = self.value_net(states).squeeze()
        advantages = rewards - values.detach()

        # Importance sampling ratio (πθ / πθ') - using stored policies from replay buffer
        importance_sampling_ratio = action_probs.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1) / old_policies.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Policy network update
        self.policy_optimizer.zero_grad()
        # it should be the gradient of action_log_probs here, will debug later.
        policy_loss = -torch.mean(
            importance_sampling_ratio * action_log_probs * advantages
        )
        entropy_loss = -torch.mean(
            torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1)
        )  # Entropy term
        total_policy_loss = policy_loss + beta * entropy_loss
        total_policy_loss.backward()
        self.policy_optimizer.step()

        # Value network update
        self.value_optimizer.zero_grad()
        value_loss = F.mse_loss(values, rewards)
        value_loss.backward()
        self.value_optimizer.step()

        # Update priorities in the replay buffer based on advantage
        priorities = torch.abs(advantages).detach()
        for idx, priority in enumerate(priorities):
            self.replay_buffer.update_priority(idx, priority)


class SharedLearner:
    __slots__ = ("lock", "learner")

    def __init__(self, learner: Learner) -> None:
        self.lock = MRSWLock()
        self.learner = learner

    def get_state_dicts(
        self,
    ) -> dict[Union[Literal["policy_net"], Literal["value_net"]], dict[str, Any]]:
        with self.lock.read():
            return {
                "policy_net": copy.deepcopy(self.learner.policy_net.state_dict()),
                "value_net": copy.deepcopy(self.learner.policy_net.state_dict()),
            }

    def train_step(self, batch_size: int, beta: float):
        with self.lock.write():
            return self.learner.train_step(batch_size, beta)


functools.update_wrapper(SharedLearner.train_step, Learner.train_step)

# Usage:
if __name__ == "__main__":
    state_dim = 4  # set right values
    action_dim = 2  # set right values
    replay_buffer = PrioritizedReplayBuffer(
        alpha=0.7, beta=0.9
    )  # dummy values for alpha and beta
    policy_net = PolicyNetwork(input_size=state_dim, output_size=action_dim)
    value_net = ValueNetwork(input_size=state_dim, hidden_size=2048)
    learner = Learner(policy_net, value_net, replay_buffer, lr_pol=0.001, lr_val=0.001)

    batch_size = 32
    beta = 0.01

    for i in range(1000):
        learner.train_step(batch_size, beta)
