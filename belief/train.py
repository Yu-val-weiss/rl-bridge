import logging
import pathlib
import time
from typing import Optional

import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import wandb
from models import BeliefNetwork

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime%s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_data(file_path: pathlib.Path):
    data = torch.load(file_path)
    x = data["info_states"]
    y = data["hands"]

    dataset = TensorDataset(x, y)

    return dataset


def train_belief_network(
    data_path: pathlib.Path,
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.1,
    l2: float = 1e-4,
    patience: int = 5,
    model_path: Optional[str] = None,
    hidden_size: int = 248,
):
    dataset = load_data(data_path)

    torch.manual_seed(0)

    # Split the dataset into training, validation, and test sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BeliefNetwork(
        dropout_rate=dropout_rate, hidden_size=hidden_size, output_size=24
    )

    # Load model state if model_path is provided
    if model_path:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # weight_decay=l2

    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    wandb.init(project="RL", name=f"belief_network_{time_stamp}")
    wandb.watch(model, log="all")

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    loss = torch.tensor([0.0])
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(
            f"Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}"
        )

        # Log the losses to wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "training_loss": loss.item(),
                "validation_loss": val_loss,
            }
        )

        # Early stopping and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate on the test set and log predictions
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    # Log the test loss to wandb
    wandb.log({"test_loss": test_loss})
    wandb.finish()

    # Save the trained model
    model_save_path = f"belief/belief_net_lr_{learning_rate}_dropout_{dropout_rate}_hidden_{hidden_size}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

    return model


@click.command()
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
    default=100,
    type=int,
    help="Number of training epochs",
)
@click.option(
    "-b",
    "--batch_size",
    default=512,
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
    "-hs",
    "--hidden_size",
    default=20,
    type=int,
    help="Size of hidden layers. Default: 20",
)
@click.option(
    "-dr",
    "--dropout_rate",
    default=0.1,
    type=float,
    help="Dropout rate for training. Default: 10%",
)
def train_belief(
    dataset_path: pathlib.Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    dropout_rate: float,
):
    # Train the belief network
    train_belief_network(
        data_path=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        hidden_size=hidden_size,
    )


if __name__ == "__main__":
    train_belief()
