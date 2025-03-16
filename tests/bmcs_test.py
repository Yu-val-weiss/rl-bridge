import copy
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from open_spiel.python.rl_environment import Environment, TimeStep

from eval import BMCS
from models import BeliefNetwork, PolicyNetwork


@pytest.fixture
def card_mapping():
    return {
        "HJ": 0,  # Hearts, Jack
        "HQ": 1,  # Hearts, Queen
        "HK": 2,  # Hearts, King
        "HA": 3,  # Hearts, Ace
        "SJ": 4,  # Spades, Jack
        "SQ": 5,  # Spades, Queen
        "SK": 6,  # Spades, King
        "SA": 7,  # Spades, Ace
    }


@pytest.fixture
def policy_net():
    # Setup the environment for the Policy Network
    env = Environment("tiny_bridge_4p")
    input_dim = env.observation_spec()["info_state"][0]
    return PolicyNetwork(
        output_size=9,
        input_size=input_dim,
        hidden_size=input_dim,
    )


@pytest.fixture
def belief_net():
    # Setup the belief network
    env = Environment("tiny_bridge_4p")
    input_dim = env.observation_spec()["info_state"][0]
    return BeliefNetwork(input_size=input_dim, output_size=24)


@pytest.fixture
def bmcs(belief_net, policy_net):
    # Setup BMCS with real networks and environment
    return BMCS(
        belief_net=belief_net,
        policy_net=policy_net,
        action_history=[],  # For simplicity
        r_min=5,
        r_max=100,
        p_max=100,
        p_min=0.1,
        k=8,
    )


@pytest.fixture
def mock_environment():
    """Create a mock environment for testing."""
    env = MagicMock()
    # Setup a mock state
    env.get_state.return_value = "W:SJHJ N:SKHK E:SASQ S:HAHQ"
    return env


@pytest.fixture
def mock_policy_net():
    """Create a mock policy network for testing."""
    policy_net = MagicMock()
    # Return predictable action probabilities
    policy_net.return_value = torch.tensor([0.1, 0.5, 0.2, 0.9, 0.3])
    return policy_net


@pytest.fixture
def mock_belief_net():
    """Create a mock belief network for testing."""
    belief_net = MagicMock()
    # Mock output shape should match your actual belief net
    belief_net.return_value = torch.tensor(
        [
            [
                [0.2, 0.3, 0.1, 0.5, 0.1, 0.4, 0.3, 0.2],
                [0.3, 0.1, 0.4, 0.2, 0.3, 0.2, 0.1, 0.4],
                [0.1, 0.2, 0.5, 0.3, 0.4, 0.1, 0.2, 0.3],
            ]
        ]
    )
    return belief_net


@pytest.fixture
def mock_time_step():
    """Create a mock time step for testing."""
    time_step = MagicMock()
    time_step.observations = {
        "current_player": 0,
        "info_state": {
            0: np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            1: np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
            2: np.array([0.3, 0.4, 0.5, 0.6, 0.7]),
            3: np.array([0.4, 0.5, 0.6, 0.7, 0.8]),
        },
        "legal_actions": {0: [0, 1, 3, 4], 1: [1, 2, 3], 2: [0, 2, 4], 3: [1, 3, 4]},
    }
    return time_step


def parse_state_str(state_str: str):
    return [int(x) for x in state_str.split()]


def hand_to_num(hand: list[int]):
    return BMCS.cards_to_chance_outcome(*sorted(hand, reverse=True))


def test_sample_deal_complies_with_h(bmcs):
    env = Environment("tiny_bridge_4p")
    h = env.reset()

    expected_first_player_hand = [
        i for i, value in enumerate(h.observations["info_state"][0][:8]) if value == 1.0
    ]

    expected_first_player_hand_idx = hand_to_num(expected_first_player_hand)

    state_str = bmcs.sample_deal(h)
    state_list = parse_state_str(state_str)

    assert expected_first_player_hand_idx == state_list[0]


def test_sample_deal_assigns_unique_cards(bmcs):
    env = Environment("tiny_bridge_4p")
    h = env.reset()

    state_str = bmcs.sample_deal(h)
    deals = parse_state_str(state_str)

    assert len(set(deals)) == 4


def test_sample_deal_player_agnostic(bmcs):
    env = Environment("tiny_bridge_4p")
    h = env.reset()
    h = env.step([0])  # dummy action

    current_player = h.observations["current_player"]
    expected_player_hand = [
        i
        for i, value in enumerate(h.observations["info_state"][current_player][:8])
        if value == 1.0
    ]

    expected_player_hand_num = hand_to_num(expected_player_hand)

    state_str = bmcs.sample_deal(h)
    deals_list = parse_state_str(state_str)

    # north N is second player

    assert expected_player_hand_num == deals_list[1]


def test_rollout_restores_state(bmcs):
    bmcs.env.reset()
    original_state = copy.deepcopy(bmcs.env.get_state)

    dummy_action = 1
    bmcs.rollout(dummy_action)

    state_after_rollout = bmcs.env.get_state

    assert str(original_state) == str(state_after_rollout)


def test_rollout_functionality(mocker):
    """Test the rollout function's core functionality."""
    # Create mock environment and policy network
    mock_env = mocker.MagicMock(spec=Environment)
    mock_policy_net = mocker.MagicMock(spec=PolicyNetwork)

    # Create BMCS instance with mocks
    bmcs = BMCS(
        policy_net=mock_policy_net, belief_net=mock_policy_net, action_history=[]
    )  # Using same mock for policy and belief nets
    bmcs.env = mock_env

    # Set up the original state
    original_state = "W:SJHJ N:SKHK E:SASQ S:HAHQ"
    type(mock_env).get_state = property(lambda self: original_state)

    # Mock the first step response
    first_time_step = mocker.MagicMock(spec=TimeStep)
    first_time_step.last.return_value = False
    first_time_step.observations = {
        "current_player": 1,
        "info_state": {1: [0.1, 0.2, 0.3]},  # Simplified info state
        "legal_actions": {1: [0, 2, 4]},  # Legal actions for player 1
    }
    mock_env.step.return_value = first_time_step

    # Mock policy network output
    mock_policy_net.return_value = torch.tensor(
        [0.1, 0.5, 0.2, 0.1, 0.8]
    )  # Action 4 has highest probability

    # Mock second step to be terminal
    second_time_step = mocker.MagicMock(spec=TimeStep)
    second_time_step.last.return_value = True
    second_time_step.rewards = {0: 0, 1: 1, 2: -1, 3: 0}

    # Set up step to return second_time_step on second call
    mock_env.step.side_effect = [first_time_step, second_time_step]

    # Call rollout
    reward = bmcs.rollout(1)

    # Assert we got the right reward (player 0's reward)
    assert reward == 0

    # Assert we made the expected number of steps (two in this case)
    assert mock_env.step.call_count == 2

    # Check the action selection logic worked
    second_step_args = mock_env.step.call_args_list[1][0][0]
    assert second_step_args == [
        4
    ]  # Should have selected action 4 which had highest probability


def test_search_returns_best_action(
    mock_environment, mock_policy_net, mock_belief_net, mock_time_step
):
    """Test that search returns the best action based on rollout results."""
    # Configure the mock environment
    mock_environment.get_time_step.return_value = mock_time_step

    # Create BMCS instance with test parameters
    bmcs = BMCS(
        policy_net=mock_policy_net,
        belief_net=mock_belief_net,
        k=3,  # Consider top 3 actions
        p_min=0.2,  # Minimum action probability
        p_max=5,  # Maximum number of passes
        r_min=1,  # Minimum number of rollouts
        r_max=10,  # Maximum number of rollouts
        action_history=[],  # Empty action history for simplicity
    )

    bmcs.env = mock_environment

    # Mock the rollout method to return predetermined values
    bmcs.rollout = MagicMock()
    bmcs.rollout.side_effect = lambda action: {
        3: 0.9,  # Action 3 has highest reward
        4: 0.5,
        1: 0.3,
    }.get(action, 0.0)

    # Mock sample_deal to return a fixed state
    bmcs.sample_deal = MagicMock()
    bmcs.sample_deal.return_value = "W:SJHJ N:SKHK E:SASQ S:HAHQ"

    # Call search
    best_action = bmcs.search(mock_time_step)

    # The best action should be 3 based on our mocked rollout rewards
    assert best_action == 3


def test_search_fallback_to_policy(
    mock_environment, mock_policy_net, mock_belief_net, mock_time_step
):
    """Test that search falls back to policy network when not enough rollouts are completed."""
    # Configure the mock environment
    mock_environment.get_time_step.return_value = mock_time_step

    # Create BMCS instance with test parameters
    bmcs = BMCS(
        policy_net=mock_policy_net,
        belief_net=mock_belief_net,
        k=3,  # Consider top 3 actions
        p_min=0.2,  # Minimum action probability
        p_max=0,  # Set to 0 to force fallback (no passes)
        r_min=5,  # Minimum number of rollouts (won't be reached)
        r_max=10,  # Maximum number of rollouts
        action_history=[],  # Empty action history for simplicity
    )

    # Mocks
    bmcs.env = mock_environment
    bmcs.rollout = MagicMock()
    bmcs.rollout.return_value = 0.5
    bmcs.sample_deal = MagicMock()
    bmcs.sample_deal.return_value = "W:SJHJ N:SKHK E:SASQ S:HAHQ"

    # Call search
    best_action = bmcs.search(mock_time_step)

    # The best action should be 3 based on policy network (highest probability is 0.9)
    assert best_action == 3
    # Verify rollout was not called (p_max is 0)
    bmcs.rollout.assert_not_called()


def test_search_with_action_history(
    mock_environment, mock_policy_net, mock_belief_net, mock_time_step
):
    """Test search handling of action history."""
    # Configure the mock environment
    mock_environment.get_time_step.return_value = mock_time_step

    # Create step response for past actions
    past_action_step = MagicMock()
    past_action_step.observations = mock_time_step.observations
    mock_environment.step.return_value = past_action_step

    # Create BMCS instance
    bmcs = BMCS(
        policy_net=mock_policy_net,
        belief_net=mock_belief_net,
        k=3,
        p_min=0.2,
        p_max=5,
        r_min=1,
        r_max=10,
        action_history=[1, 2],
    )

    # Mocks
    bmcs.env = mock_environment
    bmcs.rollout = MagicMock()
    bmcs.rollout.return_value = 0.5
    bmcs.sample_deal = MagicMock()
    bmcs.sample_deal.return_value = "W:SJHJ N:SKHK E:SASQ S:HAHQ"

    # Patch random to control action filter behavior
    with patch("numpy.random.rand", return_value=1.0):  # Always include past actions
        _ = bmcs.search(mock_time_step)

    # Check environment step was called with past actions
    assert mock_environment.step.call_count >= 2


def test_search_action_filtering(mock_environment, mock_belief_net, mock_time_step):
    """Test that search correctly filters actions based on probabilities."""
    # Configure environment
    mock_environment.get_time_step.return_value = mock_time_step

    # Create custom policy net that returns specific probabilities
    custom_policy = MagicMock()
    custom_policy.return_value = torch.tensor([0.1, 0.3, 0.15, 0.9, 0.2])

    # Create BMCS with higher p_min to test filtering
    bmcs = BMCS(
        policy_net=custom_policy,
        belief_net=mock_belief_net,
        k=4,  # Consider top 4 actions
        p_min=0.25,  # Only actions with prob > 0.25 are valid
        p_max=5,
        r_min=1,
        r_max=10,
        action_history=[],
    )

    # Mocks
    bmcs.env = mock_environment
    bmcs.rollout = MagicMock()
    bmcs.rollout.side_effect = lambda action: {1: 0.3, 3: 0.8}.get(action, 0.1)

    bmcs.sample_deal = MagicMock()
    bmcs.sample_deal.return_value = "W:SJHJ N:SKHK E:SASQ S:HAHQ"

    # Call search
    best_action = bmcs.search(mock_time_step)

    # Only actions 1 and 3 should be considered valid (prob > 0.25)
    # And action 3 has highest rollout value
    assert best_action == 3

    # Check which actions were rolled out
    rollout_calls = [call.args[0] for call in bmcs.rollout.call_args_list]
    # Only actions 1 and 3 should be rolled out (probs > 0.25)
    for action in rollout_calls:
        assert action in [1, 3]
