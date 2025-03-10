from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.data import PrioritizedReplayBuffer

from training.learner import Learner


class SimplePolicyNetwork(nn.Module):
    def __init__(self):
        super(SimplePolicyNetwork, self).__init__()
        self.fc = nn.Linear(10, 2)  # Simple 1 layer network

    def forward(self, x):
        return self.fc(x)


class SimpleValueNetwork(nn.Module):
    def __init__(self):
        super(SimpleValueNetwork, self).__init__()
        self.fc = nn.Linear(10, 1)  # Simple 1 layer network

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def setup_learner():
    # Use real networks
    policy_net = SimplePolicyNetwork()  # Use actual model
    value_net = SimpleValueNetwork()  # Use actual model
    replay_buffer = MagicMock(spec=PrioritizedReplayBuffer)

    lr_pol = 1e-3
    lr_val = 1e-3

    # Initialize the Learner with the actual models
    learner = Learner(
        policy_net=policy_net,
        value_net=value_net,
        replay_buffer=replay_buffer,
        lr_pol=lr_pol,
        lr_val=lr_val,
    )

    return learner, policy_net, value_net, replay_buffer


def test_initialization(setup_learner):
    learner, policy_net, value_net, replay_buffer = setup_learner

    # Test if the learner is initialized correctly
    assert learner.policy_net == policy_net
    assert learner.value_net == value_net
    assert learner.replay_buffer == replay_buffer
    assert isinstance(learner.policy_optimizer, optim.Adam)
    assert isinstance(learner.value_optimizer, optim.Adam)


def test_train_step_updates_policy_and_value_network(setup_learner):
    learner, policy_net, value_net, replay_buffer = setup_learner

    batch_size = 5
    beta = 0.01

    # Mock the data returned from the replay buffer
    mock_batch = [
        (
            np.random.rand(10).tolist(),
            0,
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
        )
        for _ in range(batch_size)
    ]
    replay_buffer.sample.return_value = mock_batch

    # Mock the policy and value networks' outputs
    policy_net.forward = MagicMock(return_value=torch.randn(batch_size, 2))
    value_net.forward = MagicMock(return_value=torch.randn(batch_size, 1))

    # Call train_step
    learner.train_step(batch_size=batch_size, beta=beta)

    # Ensure the policy network and value network are being called
    policy_net.forward.assert_called()
    value_net.forward.assert_called()

    # Ensure the optimizers' step methods are called
    learner.policy_optimizer.step.assert_called_once()
    learner.value_optimizer.step.assert_called_once()


def test_replay_buffer_priority_update(setup_learner):
    learner, policy_net, value_net, replay_buffer = setup_learner

    batch_size = 5
    beta = 0.01

    # Mock the data returned from the replay buffer
    mock_batch = [
        (
            np.random.rand(10).tolist(),
            0,
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
        )
        for _ in range(batch_size)
    ]
    replay_buffer.sample.return_value = mock_batch

    # Mock the policy and value networks' outputs
    policy_net.forward = MagicMock(return_value=torch.randn(batch_size, 2))
    value_net.forward = MagicMock(return_value=torch.randn(batch_size, 1))

    # Call train_step
    learner.train_step(batch_size=batch_size, beta=beta)

    # Ensure priorities were updated in the replay buffer
    replay_buffer.update_priority.assert_called()
    assert replay_buffer.update_priority.call_count == batch_size


def test_action_probs_and_value_loss_calculation(setup_learner):
    learner, policy_net, value_net, replay_buffer = setup_learner

    batch_size = 5
    beta = 0.01

    # Prepare mock batch
    mock_batch = [
        (
            np.random.rand(10).tolist(),
            0,
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
        )
        for _ in range(batch_size)
    ]
    replay_buffer.sample.return_value = mock_batch

    # Mock the outputs
    policy_net.forward = MagicMock(return_value=torch.randn(batch_size, 2))
    value_net.forward = MagicMock(return_value=torch.randn(batch_size, 1))

    # Call train_step
    learner.train_step(batch_size=batch_size, beta=beta)

    # Check the action probabilities and log probabilities
    action_probs = policy_net.forward.return_value
    actions = torch.tensor([item[1] for item in mock_batch], dtype=torch.int64)
    action_log_probs = torch.log(
        action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    )

    # Check the values and advantages
    values = value_net.forward.return_value.squeeze()
    rewards = torch.tensor([item[2] for item in mock_batch], dtype=torch.float32)
    advantages = rewards - values.detach()

    # Compute the importance sampling ratio and loss
    importance_sampling_ratio = action_probs.gather(1, actions.unsqueeze(1)).squeeze(
        1
    ) / torch.tensor([item[3] for item in mock_batch], dtype=torch.float32)
    policy_loss = -torch.mean(importance_sampling_ratio * action_log_probs * advantages)
    entropy_loss = -torch.mean(
        torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1)
    )
    total_policy_loss = policy_loss + beta * entropy_loss

    # Check that policy loss computation matches
    assert torch.isclose(
        total_policy_loss, learner.policy_optimizer.zero_grad.call_args[0][0]
    )


def test_optimization_step(setup_learner):
    learner, policy_net, value_net, replay_buffer = setup_learner

    batch_size = 5
    beta = 0.01

    # Mock the data returned from the replay buffer
    mock_batch = [
        (
            np.random.rand(10).tolist(),
            0,
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
        )
        for _ in range(batch_size)
    ]
    replay_buffer.sample.return_value = mock_batch

    # Mock the policy and value networks' outputs
    policy_net.return_value = torch.randn(batch_size, 3)  # Assuming 3 possible actions
    value_net.return_value = torch.randn(batch_size, 1)

    # Call train_step
    learner.train_step(batch_size=batch_size, beta=beta)

    # Ensure the optimizers are stepping
    learner.policy_optimizer.step.assert_called_once()
    learner.value_optimizer.step.assert_called_once()
