import logging
import pathlib
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchrl.data import TensorDictReplayBuffer

from models import Network, PolicyNetwork, ValueNetwork
from utils import MRSWLock, get_device
from utils.data import LearnerConfig


class Learner:
    def __init__(
        self,
        policy_net: Network,
        value_net: Network,
        replay_buffer: TensorDictReplayBuffer,
        batch_size: int,
        beta: float,
        lr_pol: float,
        lr_val: float,
        net_lock: MRSWLock,
        max_steps: int = 10,
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.replay_buffer = replay_buffer

        self.lr_pol = lr_pol
        self.lr_val = lr_val

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_pol)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_val)

        self.net_lock = net_lock

        self.batch_size = batch_size
        self.beta = beta

        self.max_steps = max_steps

        self.device = get_device()

        self.logger = logging.getLogger("learner")

    def spin_sample(self):
        while len(self.replay_buffer) < self.batch_size:
            continue  # spin until can sample
        return self.replay_buffer.sample(self.batch_size).to(self.device)

    def train_step(self):
        # Sample a mini-batch of transitions from the replay buffer
        batch = self.spin_sample()
        states = batch["state"]
        actions = batch["action"]
        rewards = batch["reward"]
        old_policies = batch["old_policy"]
        old_values = batch["old_value"]

        # Ensure that old_policies and old_values require gradients
        old_policies.requires_grad_()
        old_values.requires_grad_()

        # Compute action probabilities and log probabilities
        action_probs = self.policy_net(states)
        self.logger.debug(f"ap shape: {action_probs.shape}")

        action_log_probs = torch.log(
            action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        )
        self.logger.debug(f"alp shape: {action_log_probs.shape}")

        self.logger.debug(f"rew shape: {rewards.shape}")
        self.logger.debug(f"ov shape: {old_values.shape}")

        advantages = rewards - old_values

        self.logger.debug(f"adv shape: {advantages.shape}")

        # Importance sampling ratio (πθ / πθ') - using stored policies from replay buffer
        quo = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        div = old_policies.gather(1, actions.unsqueeze(1)).squeeze(1)

        self.logger.debug(f"quo shape: {quo.shape}, div: {div.shape}")

        importance_sampling_ratio = quo / div

        self.logger.debug(f"isr shape: {importance_sampling_ratio.shape}")

        # Policy network update
        self.policy_optimizer.zero_grad()
        # it should be the gradient of action_log_probs here, will debug later.
        policy_loss = -torch.mean(
            importance_sampling_ratio * action_log_probs * advantages
        )
        entropy_loss = -torch.mean(
            torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1)
        )  # Entropy term
        total_policy_loss = policy_loss + self.beta * entropy_loss
        total_policy_loss.backward()

        # Value network update
        self.value_optimizer.zero_grad()
        value_loss = F.mse_loss(old_values, rewards)
        value_loss.backward()

        with self.net_lock.write():
            self.policy_optimizer.step()
            self.value_optimizer.step()

        # Update priorities in the replay buffer based on advantage
        priorities = torch.abs(advantages).detach()
        for idx, priority in enumerate(priorities):
            self.replay_buffer.update_priority(idx, priority)

    def train_loop(self, checkpoint_path: str, checkpoint_every: int):
        ckp = pathlib.Path(checkpoint_path)
        ckp.mkdir(parents=True, exist_ok=True)
        assert ckp.is_dir(), "checkpoint path must be a directory"
        for i in range(self.max_steps):
            self.logger.debug(f"step {i}")
            self.train_step()
            if i % checkpoint_every == 0:
                self.logger.info(f"saving at step {i}")
                self.save(ckp / f"step_{i}.pt")

        self.save(ckp / f"step_{self.max_steps}.pt")
        self.logger.info(f"iteration complete at step {self.max_steps}")

    def save(self, path: pathlib.Path):
        checkpoint = {
            "policy_net": {
                "config": self.policy_net.get_init_config(),
                "state_dict": self.policy_net.state_dict(),
                "optimizer": self.policy_optimizer.state_dict(),
                "lr": self.lr_pol,
            },
            "value_net": {
                "config": self.value_net.get_init_config(),
                "state_dict": self.value_net.state_dict(),
                "optimizer": self.value_optimizer.state_dict(),
                "lr": self.lr_val,
            },
            "beta": self.beta,
            "batch_size": self.batch_size,
        }
        torch.save(checkpoint, path)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        replay_buffer: TensorDictReplayBuffer,
        net_lock: MRSWLock,
    ):
        """
        Create a new Learner instance from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file
            replay_buffer (TensorDictReplayBuffer): Replay buffer to use
            net_lock (MRSWLock, optional): Lock for thread safety

        Returns:
            Learner: A new instance initialized with the checkpoint's weights and config
        """
        checkpoint = torch.load(checkpoint_path, map_location=get_device())

        # Create networks with correct architecture
        policy_net = PolicyNetwork(**checkpoint["policy_net"]["config"])
        value_net = ValueNetwork(**checkpoint["value_net"]["config"])

        # Create learner instance
        learner = cls(
            policy_net=policy_net,
            value_net=value_net,
            replay_buffer=replay_buffer,
            batch_size=checkpoint["batch_size"],
            beta=checkpoint["beta"],
            lr_pol=checkpoint["policy_net"]["lr"],
            lr_val=checkpoint["value_net"]["lr"],
            net_lock=net_lock,
        )

        # Load states
        learner.load("", checkpoint)
        return learner

    def load(self, checkpoint_path: str, checkpoint_dict: Optional[dict] = None):
        """
        Load the learner's state from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        checkpoint = checkpoint_dict or torch.load(
            checkpoint_path, map_location=self.device
        )

        self.policy_net.load_state_dict(checkpoint["policy_net"]["state_dict"])
        self.value_net.load_state_dict(checkpoint["value_net"]["state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_net"]["optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_net"]["optimizer"])
        self.beta = checkpoint["beta"]
        self.batch_size = checkpoint["batch_size"]

    @classmethod
    def from_config(
        cls,
        policy_net: Network,
        value_net: Network,
        replay_buffer: TensorDictReplayBuffer,
        lock: MRSWLock,
        config: LearnerConfig,
    ):
        return cls(
            policy_net=policy_net,
            value_net=value_net,
            replay_buffer=replay_buffer,
            net_lock=lock,
            **asdict(config),
        )


# Usage:
if __name__ == "__main__":
    state_dim = 4  # set right values
    action_dim = 2  # set right values
    replay_buffer = TensorDictReplayBuffer(
        alpha=0.7, beta=0.9
    )  # dummy values for alpha and beta
    policy_net = PolicyNetwork(input_size=state_dim, output_size=action_dim)
    value_net = ValueNetwork(input_size=state_dim, hidden_size=2048)
    learner = Learner(
        policy_net,
        value_net,
        replay_buffer,
        lr_pol=0.001,
        lr_val=0.001,
        net_lock=MRSWLock(),
        batch_size=32,
        beta=0.01,
    )

    for i in range(1000):
        learner.train_step()
