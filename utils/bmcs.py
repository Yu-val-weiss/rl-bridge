import threading
from typing import Dict

import numpy as np
import pyspiel
import torch
import torch.nn as nn
from open_spiel.python.rl_environment import Environment, TimeStep


class BMCS:
    def __init__(
        self,
        belief_net: nn.Module,
        policy_net: nn.Module,
        r_min: int,  # minimum number of rollouts
        r_max: int,  # maximum number of rollouts
        p_max: int,  # maximum number of deals to sample
        p_min: float,  # minimum action probability to consider
        k: int,  # maximum number of actions for search
    ) -> None:
        self.belief_net = belief_net
        self.policy_net = policy_net
        self.r_min = r_min
        self.r_max = r_max
        self.p_max = p_max
        self.p_min = p_min
        self.k = k
        self.env = Environment("tiny_bridge_4p")

        self.policy_net_lock = threading.Lock()

    # TODO: finish this!
    def sample_deal(self, h: TimeStep) -> TimeStep:
        """Samples a deal consistent with the belief distribution."""
        # return self.belief_net(h)
        return h

    def rollout(self, action: int) -> float:
        """Performs a rollout and restores the original state afterward."""
        original_state: pyspiel.State = self.env.get_state  # Save the current state
        original_player = original_state.current_player()

        time_step = self.env.step([action])
        while not time_step.last():
            current_player = time_step.observations["current_player"]
            obs = torch.tensor(
                time_step.observations["info_state"][current_player],
                dtype=torch.float32,
            )

            with self.policy_net_lock:
                action_probs = self.policy_net(obs).detach().numpy()

            # Select from legal actions
            legal_actions = time_step.observations["legal_actions"][current_player]
            legal_action_probs = action_probs[legal_actions]
            legal_action_probs /= legal_action_probs.sum()

            greedy_action = np.argmax(legal_action_probs)
            time_step = self.env.step([greedy_action])

        reward = time_step.rewards[original_player]

        self.env.set_state(original_state)  # Restore the environment state

        return reward

    def search(self, h: TimeStep) -> int:
        """Performs Belief Monte Carlo Search and returns the best action."""
        history = h.observations["action_history"]
        action_probs = (
            self.policy_net(
                torch.tensor(
                    h.observations["info_state"][h.observations["current_player"]],
                    dtype=torch.float32,
                )
            )
            .detach()
            .numpy()
        )
        top_actions = np.argsort(action_probs)[-self.k :]
        valid_actions = [a for a in top_actions if action_probs[a] > self.p_min]

        V: Dict[int, float] = {a: 0 for a in valid_actions}
        R, P = 0, 0

        while P < self.p_max and R < self.r_max:
            sampled_deal = self.sample_deal(h)
            P += 1

            for past_action in history:
                if (
                    np.random.rand()
                    < 1
                    - self.policy_net(
                        torch.tensor(
                            sampled_deal.observations["info_state"][
                                sampled_deal.observations["current_player"]
                            ],
                            dtype=torch.float32,
                        )
                    )
                    .detach()
                    .numpy()[past_action]
                ):
                    continue

            for action in valid_actions:
                V[action] += self.rollout(h, action)
            R += 1

        if R > self.r_min:
            return max(V, key=V.get)

        return np.argmax(action_probs)
