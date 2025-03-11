from typing import Dict, List, TypeVar

import numpy as np
import torch
from open_spiel.python.rl_environment import Environment, TimeStep

from models import BeliefNetwork, Network

PolicyNet = TypeVar("PolicyNet", bound=Network)


class BMCS:
    def __init__(
        self,
        belief_net: BeliefNetwork,
        policy_net: Network[PolicyNet],
        action_history: List[int],
        r_min: int = 5,  # minimum number of rollouts
        r_max: int = 100,  # maximum number of rollouts
        p_max: int = 100,  # maximum number of deals to sample
        p_min: float = 0.1,  # minimum action probability to consider
        k: int = 8,  # maximum number of actions for search
    ) -> None:
        self.belief_net = belief_net
        self.policy_net = policy_net
        self.r_min = r_min
        self.r_max = r_max
        self.p_max = p_max
        self.p_min = p_min
        self.k = k
        self.action_history = action_history
        self.env = Environment("tiny_bridge_4p")

    def sample_deal(self, h: TimeStep) -> str:
        """Samples a deal consistent with the belief distribution and returns the state string."""
        current_player = h.observations["current_player"]
        known_hand_vector = h.observations["info_state"][current_player][:8]
        known_hand = [i for i, value in enumerate(known_hand_vector) if value == 1.0]
        obs = torch.tensor(
            h.observations["info_state"][current_player], dtype=torch.float32
        )
        obs = obs.unsqueeze(0)  # Add batch dimension

        hands_probs = (
            self.belief_net(obs).detach().numpy()
        )  # shape [batch_size = 1, 3, 8]
        hands_probs = hands_probs.squeeze(0)
        num_players = hands_probs.shape[0]

        dealt_cards = set(known_hand)
        player_hands = {current_player: known_hand}

        for player_idx in range(num_players):
            card_probs = hands_probs[player_idx].copy()

            # Zero out already dealt cards
            for card in dealt_cards:
                card_probs[card] = 0.0

            sampled_hand = []
            while len(sampled_hand) < 2:  # every player should have 2 cards
                # Normalize probabilities
                if np.sum(card_probs) > 0:
                    card_probs = card_probs / np.sum(card_probs)
                else:
                    # Handle the case where all remaining probabilities are zero
                    remaining_cards = [
                        c for c in range(len(card_probs)) if c not in dealt_cards
                    ]
                    if remaining_cards:
                        card_probs = np.zeros_like(card_probs)
                        card_probs[remaining_cards] = 1.0 / len(remaining_cards)
                    else:
                        break

                sampled_card = np.random.choice(len(card_probs), p=card_probs)
                sampled_hand.append(sampled_card)
                dealt_cards.add(sampled_card)

                # Zero out the sampled card for next selection
                card_probs[sampled_card] = 0.0

            # TODO: look into whether this is the right logic. If player 2 is playing, what does the belief
            # network output represent? What order do the outputs of the belief network go in?
            player_hands[(current_player + player_idx + 1) % 4] = sampled_hand

        state_str = self.construct_state_str(player_hands)

        return state_str

    def rollout(self, action: int) -> float:
        """Performs a rollout and restores the original state afterward."""
        original_state = self.env.get_state  # Save the current state

        time_step = self.env.step([action])
        original_player = (time_step.observations["current_player"] - 1) % 4
        while not time_step.last():
            current_player = time_step.observations["current_player"]
            obs = torch.tensor(
                time_step.observations["info_state"][current_player],
                dtype=torch.float32,
            )

            action_probs = self.policy_net(obs).detach().numpy()

            # Select from legal actions
            legal_actions = time_step.observations["legal_actions"][current_player]
            legal_action_probs = action_probs[legal_actions]
            legal_action_probs /= legal_action_probs.sum()

            greedy_action_idx = np.argmax(legal_action_probs)
            time_step = self.env.step([legal_actions[greedy_action_idx]])

        reward = time_step.rewards[original_player]

        self.env.set_state(original_state)  # Restore the environment state

        return reward

    def card_to_str(self, card_idx: int) -> str:
        """
        Converts a card index to its string representation.
        All the logic is taken from the tiny_bridge.cc code.
        """

        suits = ["H", "S", "N"]
        ranks = ["J", "Q", "K", "A"]

        suit = suits[card_idx // len(ranks)]
        rank = ranks[card_idx % len(ranks)]

        return f"{suit}{rank}"

    def construct_state_str(self, player_hands: Dict[int, List[int]]) -> str:
        """Constructs a Tiny Bridge state string from sampled player hands."""
        hand_strs = {0: "W:", 1: "N:", 2: "E:", 3: "S:"}

        # Convert hands into formatted strings
        for player, cards in player_hands.items():
            hand_strs[player] += "".join(
                self.card_to_str(card) for card in sorted(cards)
            )

        # Combine hands into the state string
        state_str = " ".join(hand_strs.values())

        return state_str  # Example: "W:HQHJ N:SKHA E:SASJ S:SQHK"

    def search(self, h: TimeStep) -> int:
        """Performs Belief Monte Carlo Search and returns the best action."""
        current_player = h.observations["current_player"]

        action_probs = (
            self.policy_net(
                torch.tensor(
                    h.observations["info_state"][current_player],
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
            new_state = self.sample_deal(h)
            self.env.set_state(new_state)
            h_new = self.env.get_time_step()

            P += 1

            for past_action in self.action_history:
                action_prob = (
                    self.policy_net(
                        torch.tensor(
                            h_new.observations["info_state"][current_player],
                            dtype=torch.float32,
                        )
                    )
                    .detach()
                    .numpy()[past_action]
                )

                if np.random.rand() < (1 - action_prob):
                    continue

                h_new = self.env.step([past_action])

            for action in valid_actions:
                V[action] += self.rollout(action)
            R += 1

        if R > self.r_min:
            return max(V, key=lambda x: V[x])
        else:
            return int(np.argmax(action_probs))
