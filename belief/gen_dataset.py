import logging
import pathlib
from typing import List, Tuple

import click
import numpy as np
import torch
from open_spiel.python.rl_environment import Environment, TimeStep
from tqdm import tqdm

from models import Network, PolicyNetwork
from utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


class DataGenerator:
    def __init__(self, policy_net: Network, num_games: int = 10000) -> None:
        self.policy_net = policy_net
        self.num_games = num_games
        self.env = Environment("tiny_bridge_4p")
        self.device = get_device()
        self.logger = logging.getLogger("data_generator")

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
        legal_action_probs = action_probs[legal_actions]
        legal_action_probs /= legal_action_probs.sum()  # Normalize probabilities
        action = np.random.choice(legal_actions, p=legal_action_probs)

        time_step = self.env.step([action])

        return time_step

    def play_bidding_phase(
        self, time_step: TimeStep
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Plays through the bidding phase using policy network and stores observations."""
        observations = []
        hands = []
        while not time_step.last():
            current_player = time_step.observations["current_player"]
            obs = torch.tensor(
                time_step.observations["info_state"][current_player],
                dtype=torch.float32,
                device=self.device,
            )
            # Collect other players' hands
            player_hands = []
            for i in range(1, 4):
                hand = time_step.observations["info_state"][(current_player + i) % 4][
                    :8
                ]
                assert len(hand) == 8
                player_hands.append(torch.tensor(hand, dtype=torch.float32))

            assert len(player_hands) == 3
            assert torch.cat(player_hands).size(dim=0) == 24
            # Store observation and concatenated hands
            observations.append(obs)
            hands.append(torch.cat(player_hands))  # This creates a tensor of size [24]

            time_step = self.play_bidding_step(time_step)

        return observations, hands

    def generate_data(self):
        """Generate data from self-playing agents."""
        all_observations = []
        all_hands = []
        for game in tqdm(range(self.num_games), desc="Generating games"):
            deal = self.sample_deal()
            observations, hands = self.play_bidding_phase(deal)
            all_observations.extend(observations)
            all_hands.extend(hands)

        all_observations_tensor = torch.stack(all_observations)
        all_hands_tensor = torch.stack(all_hands)

        self.save_data(all_observations_tensor, all_hands_tensor)

    def save_data(
        self,
        observations: torch.Tensor,
        hands: torch.Tensor,
    ) -> None:
        """Save the generated data to a file"""
        torch.save(
            {"info_states": observations, "hands": hands},
            "belief/dataset.pt",
        )
        self.logger.info(f"Dataset saved. {observations.size(dim=0)} elements.")


@click.command()
@click.option(
    "-p",
    "--policy_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Path to trained policy network",
)
@click.option(
    "-n",
    "--num_games",
    default=10000,
    type=int,
    help="Number of games to generate",
)
@click.option(
    "-hs",
    "--hidden_size",
    default=2048,
    type=int,
    help="Number of neurons in hidden layers",
)
@click.option(
    "-os",
    "--output_size",
    default=9,
    type=int,
    help="Size of the output layer",
)
@click.option(
    "-n",
    "--input_size",
    default=84,
    type=int,
    help="Size of the input layer",
)
def self_play(
    policy_path: pathlib.Path,
    num_games: int,
    hidden_size: int,
    output_size: int,
    input_size: int,
):
    policy_net = PolicyNetwork(
        output_size=output_size, hidden_size=hidden_size, input_size=input_size
    )
    policy_net.load_state_dict(
        torch.load(policy_path, map_location=torch.device("cpu"))
    )
    data_generator = DataGenerator(policy_net, num_games=num_games)
    data_generator.generate_data()


if __name__ == "__main__":
    self_play()
