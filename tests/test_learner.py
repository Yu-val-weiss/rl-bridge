import tempfile
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler

from models import PolicyNetwork, ValueNetwork
from training.learner import Learner
from utils import MRSWLock


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def replay_buffer():
    S = 1_000_000
    sampler = PrioritizedSampler(S, 1.1, 1.0)
    storage = LazyMemmapStorage(S)
    return TensorDictReplayBuffer(storage=storage, sampler=sampler)


POLICY_ARGS = dict(input_size=2, hidden_size=6, output_size=4)


@pytest.fixture
def policy_net():
    return PolicyNetwork(**POLICY_ARGS)


VALUE_ARGS = dict(input_size=4, hidden_size=6)


@pytest.fixture
def value_net():
    return ValueNetwork(**VALUE_ARGS)


@pytest.fixture
def learner(replay_buffer, policy_net, value_net):
    return Learner(
        policy_net=policy_net,
        value_net=value_net,
        replay_buffer=replay_buffer,
        batch_size=32,
        beta=0.01,
        lr_pol=0.001,
        lr_val=0.001,
        net_lock=MRSWLock(),
    )


def test_save_checkpoint(learner, temp_dir):
    checkpoint_path = temp_dir / "test_checkpoint"
    learner.save(checkpoint_path)

    assert checkpoint_path.with_suffix(".pt").exists()

    checkpoint = torch.load(checkpoint_path.with_suffix(".pt"))
    assert "policy_net" in checkpoint
    assert "value_net" in checkpoint
    assert "beta" in checkpoint
    assert "batch_size" in checkpoint


def test_load_checkpoint(learner, temp_dir):
    # Save initial state
    checkpoint_path = temp_dir / "test_checkpoint"
    learner.save(checkpoint_path)

    # Create new learner with different parameters
    new_learner = Learner(
        policy_net=PolicyNetwork(**POLICY_ARGS),
        value_net=ValueNetwork(**VALUE_ARGS),
        replay_buffer=learner.replay_buffer,
        batch_size=64,  # Different from original
        beta=0.02,  # Different from original
        lr_pol=0.002,
        lr_val=0.002,
        net_lock=MRSWLock(),
    )

    # Load state
    new_learner.load(str(checkpoint_path.with_suffix(".pt")))

    # Verify loaded state matches original
    assert new_learner.batch_size == learner.batch_size
    assert new_learner.beta == learner.beta


def test_from_checkpoint(learner, temp_dir, replay_buffer):
    # Save initial state
    checkpoint_path = temp_dir / "test_checkpoint"
    learner.save(checkpoint_path)

    # Create new learner from checkpoint
    new_learner = Learner.from_checkpoint(
        str(checkpoint_path.with_suffix(".pt")),
        replay_buffer=replay_buffer,
        net_lock=MRSWLock(),
    )

    # Verify loaded state matches original
    assert new_learner.batch_size == learner.batch_size
    assert new_learner.beta == learner.beta

    # Test network architectures
    assert type(new_learner.policy_net) is type(learner.policy_net)
    assert type(new_learner.value_net) is type(learner.value_net)


def test_train_step(learner, replay_buffer):
    """
    Test that the train_step method of the Learner class can run without errors.
    Adds dummy data to the replay buffer for the test.
    """
    # Create a batch of dummy transitions
    dummy_state = torch.randn(2)
    dummy_action = torch.tensor(0)
    dummy_reward = torch.tensor(1.0)
    dummy_old_policy = torch.tensor(0.5)
    dummy_old_value = torch.tensor(0.3)

    tensor_dict = TensorDict(
        {
            "state": torch.stack([dummy_state for _ in range(32)]),
            "action": torch.stack([dummy_action for _ in range(32)]),
            "reward": torch.stack([dummy_reward for _ in range(32)]),
            "old_policy": torch.stack([dummy_old_policy for _ in range(32)]),
            "old_value": torch.stack([dummy_old_value for _ in range(32)]),
        },
        batch_size=[32],
    )

    replay_buffer.extend(tensor_dict)

    learner.train_step()

    assert True
