import logging
import pathlib
import threading
from dataclasses import asdict
from typing import Optional, Tuple

import torch
import torch.optim as optim
from torchrl.data import TensorDictReplayBuffer

import wandb
from models import Network, PolicyNetwork, ValueNetwork
from utils import MRSWLock, get_device
from utils.config import LearnerConfig, WBConfig


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
        clip_eps: float,
        max_steps: int = 10,
        wandb_conf: Optional[WBConfig] = None,
        value_weight: float = 1.0,
        max_grad_norm: float = 0.1,  # also experimented with 1.0
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

        self.wandb = False

        self.step_num = 0

        if wandb_conf:
            wandb.init(
                project=wandb_conf.project,
                name=wandb_conf.run_name,
                entity=wandb_conf.entity,
            )
            self.wandb = True

        self.logger = logging.getLogger("learner")

        self.clip_eps = clip_eps
        self.value_weight = value_weight
        self.max_grad_norm = max_grad_norm

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
        _ = batch[
            "old_value"
        ]  # old values don't seem to be needed w/ author's implementation but seems odd...

        p_loss, v_loss, priorities = self.compute_loss_and_priority(
            states, actions, rewards, old_policies
        )

        p_loss = p_loss.mean()
        v_loss = v_loss.mean()

        # Policy network update
        self.policy_optimizer.zero_grad()

        if self.wandb:
            wandb.log({"policy loss": p_loss}, step=self.step_num)

        p_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)

        # Value network update
        self.value_optimizer.zero_grad()

        if self.wandb:
            wandb.log({"value loss": v_loss}, step=self.step_num)

        v_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)

        with self.net_lock.write():
            self.policy_optimizer.step()
            self.value_optimizer.step()

        # Update priorities in the replay buffer based on advantage
        for idx, priority in enumerate(priorities):
            self.replay_buffer.update_priority(idx, priority)

    def compute_loss_and_priority(
        self, states, actions, rewards, old_policies
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = self.value_net(states)
        pi = self.policy_net(states)
        current_log_probs = torch.log(pi + 1e-16)
        current_action_log_probs = current_log_probs.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        old_log_probs = torch.log(old_policies + 1e-16)
        old_action_log_probs = old_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        ratio = torch.exp(current_action_log_probs - old_action_log_probs)

        adv = rewards - v.squeeze()

        surr1 = ratio * (adv.detach())
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * (
            adv.detach()
        )

        entropy = -torch.sum(pi * current_log_probs, -1)

        p_loss = -torch.min(surr1, surr2) - self.beta * entropy
        v_loss = torch.pow(adv, 2) * self.value_weight

        priority = torch.abs(adv).detach().cpu()

        if self.wandb:
            wandb.log(
                {
                    "adv_min": torch.min(adv),
                    "adv_max": torch.max(adv),
                    "adv_mean": torch.mean(adv),
                    "adv_norm": torch.norm(adv),
                    "ratio_mean": torch.mean(ratio),
                    "entropy_mean": torch.mean(entropy),
                    "action probs min": torch.min(torch.exp(current_action_log_probs)),
                    "action probs max": torch.max(torch.exp(current_action_log_probs)),
                    "action probs mean": torch.mean(
                        torch.exp(current_action_log_probs)
                    ),
                },
                step=self.step_num,
            )

        return p_loss, v_loss, priority

    def train_loop(
        self,
        checkpoint_path: str,
        checkpoint_every: int,
        sync_events: list[threading.Event],
        sync_freq: int,
    ):
        ckp = pathlib.Path(checkpoint_path)
        ckp.mkdir(parents=True, exist_ok=True)
        assert ckp.is_dir(), "checkpoint path must be a directory"
        for i in range(self.max_steps):
            self.logger.debug(f"step {i}")
            self.step_num = i
            self.train_step()
            if i % sync_freq == 0:
                for x in sync_events:
                    x.set()
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
            "clip_eps": self.clip_eps,
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
            clip_eps=checkpoint["clip_eps"],
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
        wandb: Optional[WBConfig],
    ):
        return cls(
            policy_net=policy_net,
            value_net=value_net,
            replay_buffer=replay_buffer,
            net_lock=lock,
            wandb_conf=wandb,
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
        clip_eps=0.2,
    )

    for i in range(1000):
        learner.train_step()
