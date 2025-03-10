import pytest
import torch
import torch.nn as nn
from open_spiel.python.rl_environment import Environment
from torchrl.data import ListStorage, PrioritizedReplayBuffer

from models import Network, PolicyNetwork
from training.actor import Actor
from utils import MRSWLock


class DummyNet(Network):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.ones(self.output_size)  # Dummy uniform output

    def clone(self) -> "DummyNet":
        model = DummyNet(self.input_size, self.output_size)
        model.load_state_dict(self.state_dict())
        return model


@pytest.fixture
def replay_buffer():
    return PrioritizedReplayBuffer(
        alpha=0.1, beta=0, batch_size=32, storage=ListStorage(max_size=1000)
    )


@pytest.fixture
def policy_net():
    env = Environment("tiny_bridge_4p")
    input_dim = env.observation_spec()["info_state"][0]
    return PolicyNetwork(
        output_size=9,
        input_size=input_dim,
        hidden_size=input_dim,
    )


@pytest.fixture
# TODO: replace with real value net as soon as it's implemented
def mock_value_net():
    env = Environment("tiny_bridge_4p")
    input_dim = env.observation_spec()["info_state"][0]
    return DummyNet(input_size=input_dim, output_size=1)


@pytest.fixture
def actor(replay_buffer, policy_net, mock_value_net):
    actor = Actor(
        actor_id=0,
        policy_net=policy_net,
        value_net=mock_value_net,
        replay_buffer=replay_buffer,
        sync_freq=10,
        net_lock=MRSWLock(),
    )
    return actor


def test_sample_deal(actor):
    deal = actor.sample_deal()
    assert deal.first()
    assert deal.current_player() == 0


def test_play_bidding_step(actor):
    first_time_step = actor.env.reset()
    second_time_step = actor.play_bidding_step(first_time_step)

    assert len(actor.local_buffer) == 1
    assert not second_time_step.first()
    assert second_time_step.observations["current_player"] == 1


def test_store_transitions(actor):
    test_data = [
        (torch.tensor([1.0]), 0, 1.0, [0.1, 0.2, 0.3, 0.4]),
        (torch.tensor([2.0]), 1, 2.0, [0.2, 0.3, 0.4, 0.5]),
    ]

    for transition in test_data:
        actor.local_buffer.append(transition)

    rewards = [1.0, 0.5, 0.3, 0.2]
    actor.store_transitions(rewards)

    assert len(actor.replay_buffer) == 8  # 2 transitions * 4 rewards

    stored_transitions = {
        (
            obs,
            action,
            reward,
            tuple(policy_probs),
            value,
        )  # Convert list to tuple for hashability
        for obs, action, reward, policy_probs, value in actor.replay_buffer._storage
    }

    expected_transitions = {
        (
            obs,
            action,
            reward,
            tuple(policy_probs),
            value,
        )  # Convert list to tuple for hashability
        for obs, action, value, policy_probs in test_data
        for reward in rewards
    }

    assert stored_transitions == expected_transitions


def test_play_bidding_phase(actor):
    first_time_step = actor.env.reset()
    final_time_step = actor.play_bidding_phase(first_time_step)

    assert len(actor.local_buffer) > 0
    assert final_time_step.last()


def test_run(actor):
    actor.run()

    assert actor.step_count == actor.max_rounds
    assert len(actor.replay_buffer) > 0
