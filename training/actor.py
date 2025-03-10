import contextlib
from collections import deque
from typing import List, TypeVar

import numpy as np
import torch
from open_spiel.python.rl_environment import Environment, TimeStep
from torch import nn
from torchrl.data import PrioritizedReplayBuffer

from models import Network
from utils import MRSWLock

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
        replay_buffer: PrioritizedReplayBuffer,
        sync_freq: int,
        net_lock: MRSWLock,
    ) -> None:
        self.actor_id = actor_id

        self._policy_net = policy_net  # points to shared one, should not be used
        self._value_net = value_net  # points to shared one, should not be used

        self.synchronize(at_init=True)

        self.replay_buffer = replay_buffer
        self.sync_freq = sync_freq

        self.env = Environment("tiny_bridge_4p")
        self.step_count = 0
        self.local_buffer = deque()

        self.net_lock = net_lock

        self.max_rounds = 10

    # TODO: implement synchronization
    def synchronize(self, *, at_init=False):
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
            time_step.observations["info_state"][current_player], dtype=torch.float32
        )
        legal_actions = time_step.observations["legal_actions"][current_player]

        action_probs = self.policy_net(obs).detach().numpy()

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
        """Store transitions in shared replay buffer for all players."""
        transitions = []  # Collect transitions before extending the buffer

        while self.local_buffer:
            obs, action, value, policy_probs = self.local_buffer.popleft()

            for reward in rewards:
                transitions.append((obs, action, reward, policy_probs, value))

        # no lock as replay buffer should be thread safe
        self.replay_buffer.extend(transitions)

    def run(self) -> None:
        """Main loop of the Actor thread."""
        while self.step_count < self.max_rounds:
            self.step_count += 1
            # if self.step_count % self.sync_freq == 0:
            # self.synchronize()

            deal = self.sample_deal()
            final_time_step = self.play_bidding_phase(deal)
            self.store_transitions(final_time_step.rewards)
