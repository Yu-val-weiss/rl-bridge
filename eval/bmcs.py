import copy
from typing import TypeVar

import numpy as np
import torch
from open_spiel.python.rl_environment import Environment, TimeStep

from models import BeliefNetwork, Network
from utils import get_device

from .sampler import HandDealer

PolicyNet = TypeVar("PolicyNet", bound=Network)


class BMCS:
    def __init__(
        self,
        belief_net: BeliefNetwork,
        policy_net: Network[PolicyNet],
        action_history: list[int],
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
        self.dealer = HandDealer()
        self.env = Environment("tiny_bridge_4p", chance_event_sampler=self.dealer)
        self.device = get_device()

    def sample_deal(self, h: TimeStep) -> list[int]:
        """Samples a deal consistent with the belief distribution and returns the deal actions necessary to construct it."""
        current_player = h.observations["current_player"]
        known_hand = self.get_known_hand(h, current_player)
        obs = torch.tensor(
            h.observations["info_state"][current_player],
            dtype=torch.float32,
            device=self.device,
        )
        obs = obs.unsqueeze(0)  # Add batch dimension

        hands_probs = (
            self.belief_net(obs).cpu().detach().numpy()
        )  # shape [batch_size = 1, 24]
        hands_probs = hands_probs.squeeze(0).reshape(3, 8)  # reshape to [3, 8]
        num_players = hands_probs.shape[0]

        dealt_cards = set(known_hand)
        player_hands = {current_player: known_hand}

        for player_idx in range(num_players):  # iterate through other players
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
            sample_player_idx = (
                player_idx if player_idx < current_player else player_idx + 1
            )
            player_hands[sample_player_idx] = sampled_hand

        deals = self.construct_deal_actions(player_hands)

        return deals

    def get_known_hand(self, ts: TimeStep, current_player: int):
        def extract_from_info_tensor(info_tensor):
            khv = info_tensor[:8]
            return [i for i, value in enumerate(khv) if value == 1.0]

        known_hand = extract_from_info_tensor(
            ts.observations["info_state"][current_player]
        )
        if len(known_hand) != 2:
            # fallback option, hacky but works
            state_to_deserialize = (
                ts.observations["serialized_state"]
                .lower()
                .split("[state]")[1]
                .strip()
                .split("\n")[:4]
            )

            state = self.env.game.deserialize_state("\n".join(state_to_deserialize))  # type: ignore
            known_hand = extract_from_info_tensor(
                state.information_state_tensor(current_player)
            )

        return known_hand

    def rollout(self, action: int) -> float:
        """Performs a rollout and restores the original state afterward."""

        # save the current state, need to copy otherwise state will get modified
        original_state = copy.deepcopy(self.env.get_state)

        time_step = self.env.step([action])
        original_player = (time_step.observations["current_player"] - 1) % 4
        while not time_step.last():
            current_player = time_step.observations["current_player"]
            obs = torch.tensor(
                time_step.observations["info_state"][current_player],
                dtype=torch.float32,
                device=self.device,
            )

            action_probs = self.policy_net(obs).cpu().detach().numpy()

            # Select from legal actions
            legal_actions = time_step.observations["legal_actions"][current_player]
            legal_action_probs = action_probs[legal_actions]
            legal_action_probs /= legal_action_probs.sum()

            greedy_action_idx = np.argmax(legal_action_probs)
            time_step = self.env.step([legal_actions[greedy_action_idx]])

        reward = time_step.rewards[original_player]

        self.env.set_state(original_state)  # Restore the environment state

        return reward

    @staticmethod
    def cards_to_chance_outcome(card0: int, card1: int):
        """Requires card0 > card1 in value.

        Converts the 2 cards in a hand (by index) to the action index in the deal phase.
        Taken from the c++ implementation of tiny bridge"""
        return int((card0 * (card0 - 1)) / 2 + card1)

    @staticmethod
    def construct_deal_actions(player_hands: dict[int, list[int]]) -> list[int]:
        """Constructs the sequence of deal actions to create sampled player hands."""
        deals = []
        # Convert hands into formatted strings
        sorted_hands = [player_hands[key] for key in sorted(player_hands.keys())]
        for cards in sorted_hands:
            assert len(cards) == 2, f"expected 2, got {len(cards)}"
            deals.append(BMCS.cards_to_chance_outcome(*sorted(cards, reverse=True)))

        return deals

    def real_deal_from_ts(self, ts: TimeStep) -> list[int]:
        player_hands = {}
        for i in range(4):
            cards = self.get_known_hand(ts, i)
            assert len(cards) == 2, f"got {len(cards)}"
            player_hands[i] = cards
        return BMCS.construct_deal_actions(player_hands)

    def search(self, h: TimeStep, *, use_ground_truth: bool = False) -> int:
        """Performs Belief Monte Carlo Search and returns the best action."""
        current_player = h.observations["current_player"]

        action_probs = (
            self.policy_net(
                torch.tensor(
                    h.observations["info_state"][current_player],
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            .cpu()
            .detach()
            .numpy()
        )
        top_actions = np.argsort(action_probs)[-self.k :]
        valid_actions = [a for a in top_actions if action_probs[a] > self.p_min]

        V: dict[int, float] = {a: 0 for a in valid_actions}
        R, P = 0, 0

        while P < self.p_max and R < self.r_max:
            deal = (
                self.sample_deal(h)
                if not use_ground_truth
                else self.real_deal_from_ts(h)
            )
            self.dealer.seed_deal(deal)
            h_new = self.env.reset()

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
