import logging
import pathlib

import click
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm, trange

from models import PolicyNetwork
from utils import get_device, load_self_play_config

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class BridgeDataset(Dataset):
    def __init__(self, info_states, actions):
        self.info_states = info_states
        self.actions = actions

    def __len__(self):
        return len(self.info_states)

    def __getitem__(self, idx):
        # Return a tuple of (info_state, action)
        info_state_tensor = self.info_states[idx]
        action_tensor = self.actions[idx]
        return info_state_tensor, action_tensor


def cross_entropy_loss(policy_net, states, actions):
    """
    Computes the cross-entropy loss for the given policy network.

    Args:
        policy_net: The policy network (assumed to output action probabilities).
        states: The input states (tensor) for which the loss is computed.
        actions: The ground truth actions (tensor) taken in those states.

    Returns:
        loss: The computed cross-entropy loss.
    """
    action_probs = policy_net(states)  # Forward pass to get action probabilities
    log_probs = torch.log(action_probs + 1e-10)  # Avoid log(0) issues
    loss = F.nll_loss(log_probs, actions)
    return loss


def load_dataset(file_path: pathlib.Path, batch_size: int, validation_split=0.1):
    """
    Loads the dataset from the given file path.

    Args:
        file_path: The path to the dataset file.

    Returns:
        Dataset
    """
    data = torch.load(file_path)
    info_states = data["info_states"]
    actions = data["actions"]

    dataset = BridgeDataset(info_states, actions)

    dataset_size = len(dataset)

    train_size = int((1 - validation_split) * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    return train_loader, val_loader


@click.command()
@click.option(
    "-c",
    "--config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Path to config yaml file",
)
@click.option(
    "-d",
    "--dataset_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Path to the dataset file",
)
@click.option(
    "-e",
    "--epochs",
    default=10,
    type=int,
    help="Number of training epochs",
)
@click.option(
    "-b",
    "--batch_size",
    default=128,
    type=int,
    help="Batch size for training",
)
@click.option(
    "-lr",
    "--learning_rate",
    default=0.001,
    type=float,
    help="Learning rate for training",
)
@click.option(
    "-vs",
    "--validation_split",
    default=0.1,
    type=float,
    help="Fraction of the dataset to reserve for validation (default 10%)",
)
def train_sl(
    config_path: pathlib.Path,
    dataset_path: pathlib.Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_split: float,
):
    conf = load_self_play_config(config_path)

    # Initialise networks
    policy_net = PolicyNetwork.from_dataclass(conf.policy_net)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    logging.info("loading dataset...")
    # Load dataset
    train_set, val_set = load_dataset(dataset_path, batch_size, validation_split)
    logging.info("dataset loaded...")

    best_val_loss = float("inf")

    num_batches = len(train_set) // batch_size

    # Training loop
    for epoch in (pbar := trange(epochs)):
        policy_net.train()
        total_loss = 0

        # Training phase
        for states, actions in tqdm(train_set, leave=False):
            # Zero the gradients
            optimizer.zero_grad()

            # Compute loss
            loss = cross_entropy_loss(
                policy_net, states.to(get_device()), actions.to(get_device())
            )

            # Backpropagate the loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Log the average loss for the epoch
        avg_loss = total_loss / num_batches

        val_total_loss = 0
        # Validation phase
        policy_net.eval()
        for val_states, val_actions in val_set:
            val_loss = cross_entropy_loss(
                policy_net, val_states.to(get_device()), val_actions.to(get_device())
            )
            val_total_loss += val_loss.item()

        # Log the validation loss
        pbar.write(
            f"epoch [{epoch + 1}/{epochs}], loss: {avg_loss:.4f}, val loss: {val_total_loss:.4f}"
        )

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save(policy_net.state_dict(), "policy_net.pt")
            pbar.write(f"new best model saved with val loss: {best_val_loss:.4f}")

    # Save the trained model
    if val_total_loss < best_val_loss:  # type: ignore
        best_val_loss = val_total_loss  # type: ignore
        torch.save(policy_net.state_dict(), "policy_net.pt")
        pbar.write(f"new best model saved with val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_sl()
