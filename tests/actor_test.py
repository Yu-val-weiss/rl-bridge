import pytest
import torch
import torch.nn as nn
from open_spiel.python.rl_environment import Environment
from tensordict import MemoryMappedTensor, TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler

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
    S = 1_000_000
    sampler = PrioritizedSampler(S, 1.1, 1.0)
    storage = LazyMemmapStorage(S)
    return TensorDictReplayBuffer(storage=storage, sampler=sampler)


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

    stored_transitions = actor.replay_buffer[:]

    expected_transitions = TensorDict(
        {
            "state": MemoryMappedTensor(
                torch.tensor([[1.0], [1.0], [1.0], [1.0], [2.0], [2.0], [2.0], [2.0]])
            ),
            "action": MemoryMappedTensor(torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])),
            "reward": MemoryMappedTensor(
                torch.tensor([1.0, 0.5, 0.3, 0.2, 1.0, 0.5, 0.3, 0.2])
            ),
            "old_policy": MemoryMappedTensor(
                torch.tensor(
                    [
                        [0.1, 0.2, 0.3, 0.4],
                        [0.1, 0.2, 0.3, 0.4],
                        [0.1, 0.2, 0.3, 0.4],
                        [0.1, 0.2, 0.3, 0.4],
                        [0.2, 0.3, 0.4, 0.5],
                        [0.2, 0.3, 0.4, 0.5],
                        [0.2, 0.3, 0.4, 0.5],
                        [0.2, 0.3, 0.4, 0.5],
                    ]
                )
            ),
            "old_value": MemoryMappedTensor(
                torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
            ),
        },
        batch_size=[8],
    )

    def compare_tensordicts(td1, td2, rtol=1e-5, atol=1e-8):
        if td1.keys() != td2.keys():
            print("Mismatch in keys:", td1.keys(), td2.keys())
            return False

        for key in td1.keys():
            tensor1 = td1[key]
            tensor2 = td2[key]

            if tensor1.dtype.is_floating_point:
                if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
                    print(f"Mismatch in field: {key}")
                    print("Tensor 1:", tensor1)
                    print("Tensor 2:", tensor2)
                    return False
            else:
                if not torch.equal(tensor1, tensor2):
                    print(f"Mismatch in field: {key}")
                    print("Tensor 1:", tensor1)
                    print("Tensor 2:", tensor2)
                    return False

        return True

    assert compare_tensordicts(stored_transitions, expected_transitions)


def test_play_bidding_phase(actor):
    first_time_step = actor.env.reset()
    final_time_step = actor.play_bidding_phase(first_time_step)

    assert len(actor.local_buffer) > 0
    assert final_time_step.last()


def test_run(actor):
    actor.run()

    assert actor.step_count == actor.max_rounds
    assert len(actor.replay_buffer) > 0
