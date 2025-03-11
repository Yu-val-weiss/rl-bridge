import contextlib
import logging
import threading
from collections import deque
from typing import List, Optional, TypeVar

import numpy as np
import torch
from open_spiel.python.rl_environment import Environment, TimeStep
from tensordict import TensorDict
from torch import nn
from torchrl.data import TensorDictReplayBuffer

from models import Network
from utils import MRSWLock, get_device
from utils.data import ActorConfig

PolicyNet = TypeVar("PolicyNet", bound=Network)
ValueNet = TypeVar("ValueNet", bound=Network)


class Actor:
    # these are the local versions, no need to synchronise them
    policy_net: nn.Module
    value_net: nn.Module

    def __init__(
        self,
        actor_id: int,
        policy_net: Network[PolicyNet],
        value_net: Network[ValueNet],
        replay_buffer: TensorDictReplayBuffer,
        sync_freq: int,
        net_lock: MRSWLock,
        max_steps: int = 10,
    ) -> None:
        self.actor_id = actor_id

        self._policy_net = policy_net  # points to shared one, should not be used
        self._value_net = value_net  # points to shared one, should not be used

        self.logger = logging.getLogger(f"actor_{actor_id}")

        self.synchronize(at_init=True)  # assigns self.policy_net and self.value_net

        self.replay_buffer = replay_buffer
        self.sync_freq = sync_freq

        self.env = Environment("tiny_bridge_4p")
        self.step_count = 0
        self.local_buffer = deque()

        self.net_lock = net_lock

        self.max_steps = max_steps

        self.device = get_device()

    def synchronize(self, *, at_init=False):
        if not at_init:
            self.logger.info(f"synchronising at time step {self.step_count}")

        with self.net_lock.read() if not at_init else contextlib.nullcontext():
            self.policy_net = self._policy_net.clone()
            self.value_net = self._value_net.clone()

    def sample_deal(self) -> TimeStep:
        """Sample a bridge deal (initial state) from the environment."""
        return self.env.reset()

    def play_bidding_step(self, time_step: TimeStep) -> TimeStep:
        """Plays through one bidding step (one action of one player)."""
        current_player = time_step.observations["current_player"]
        obs = torch.tensor(
            time_step.observations["info_state"][current_player],
            dtype=torch.float32,
            device=self.device,
        )
        legal_actions = time_step.observations["legal_actions"][current_player]

        action_probs_on_device = self.policy_net(obs)  # move to CPU before numpy() call
        action_probs = action_probs_on_device.cpu().detach().numpy()

        # Only sample an action from legal actions
        # Could replace this implementation by masking idea. This approach was straightforward now, hence I implemented it.
        legal_action_probs = action_probs[legal_actions]
        legal_action_probs /= legal_action_probs.sum()  # Normalize probabilities
        action = np.random.choice(legal_actions, p=legal_action_probs)

        value_estimate = self.value_net(obs).item()

        self.local_buffer.append((obs, action, value_estimate, action_probs))

        time_step = self.env.step([action])

        return time_step

    def play_bidding_phase(self, time_step: TimeStep) -> TimeStep:
        """Plays through the bidding phase using policy network."""
        while not time_step.last():
            time_step = self.play_bidding_step(time_step)

        return time_step

    def store_transitions(self, rewards: List[float]) -> None:
        """Store transitions in shared replay buffer for all players.

        Args:
            rewards: List of reward values to associate with each transition
        """
        transitions = {
            "state": [],
            "action": [],
            "reward": [],
            "old_policy": [],
            "old_value": [],
        }

        while self.local_buffer:
            obs, action, value, policy_probs = self.local_buffer.popleft()

            obs = obs.cpu().detach()

            for reward in rewards:
                # Add each part of the transition to the respective list
                transitions["state"].append(obs)
                transitions["action"].append(torch.tensor(action, dtype=torch.int64))
                transitions["reward"].append(torch.tensor(reward, dtype=torch.float32))
                transitions["old_policy"].append(
                    torch.tensor(policy_probs, dtype=torch.float32)
                )
                transitions["old_value"].append(
                    torch.tensor(value, dtype=torch.float32)
                )

        # Once all transitions are collected, create a TensorDict
        if transitions["state"]:
            # Stack the lists of tensors into batch tensors
            batch = TensorDict(
                {k: torch.stack(v) for k, v in transitions.items()},
                batch_size=[len(transitions["state"])],
            )
            # add batch to replay buffer (thread safe)
            self.replay_buffer.extend(batch)

    def run(self, stop_event: Optional[threading.Event] = None) -> None:
        """Main loop of the Actor thread."""
        while (
            stop_event is not None and not stop_event.is_set()
        ) or self.step_count < self.max_steps:
            self.logger.debug(f"step {self.step_count}")
            self.step_count += 1
            if self.step_count % self.sync_freq == 0:
                self.synchronize()

            deal = self.sample_deal()
            final_time_step = self.play_bidding_phase(deal)
            self.store_transitions(final_time_step.rewards)

    @classmethod
    def from_config(
        cls,
        policy_net: Network[PolicyNet],
        value_net: Network[ValueNet],
        replay_buffer: TensorDictReplayBuffer,
        net_lock: MRSWLock,
        config: ActorConfig,
    ) -> list["Actor"]:
        return [
            cls(
                id,
                policy_net,
                value_net,
                replay_buffer,
                config.sync_frequency,
                net_lock,
                config.max_steps,
            )
            for id in range(config.num_actors)
        ]
