from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from open_spiel.python.rl_environment import Environment, TimeStep

from eval.policy_agent import PolicyAgent
from utils import get_device

device = get_device()

NUM_ACTIONS = 9


@pytest.fixture
def policy_net():
    mock = MagicMock()
    mock.return_value = torch.full((9,), 1 / 9, device=device)
    return mock


@pytest.fixture
def policy_agent(policy_net):
    return PolicyAgent(
        0,
        NUM_ACTIONS,
        policy_net,
    )


@pytest.fixture
def env():
    return Environment("tiny_bridge_4p")


@pytest.fixture
def initial_step(env):
    return env.reset()


def test_initial_step(policy_agent: PolicyAgent, initial_step: TimeStep):
    action, probs = policy_agent.step(initial_step)
    legal_actions = initial_step.observations["legal_actions"][0]
    assert action in legal_actions
    expected = np.zeros(9)
    expected[legal_actions] = 1
    expected = expected / expected.sum()
    np.testing.assert_almost_equal(probs, expected)
