import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from models import BeliefNetwork
from utils import get_device

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = "\n".join(
    [
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{libertine}",
        r"\usepackage{inconsolata}",
    ]
)


def load_data(file_path: str):
    data = torch.load(file_path)
    x = data["info_states"]
    y = data["hands"]

    dataset = TensorDataset(x, y)

    return dataset


def evaluate_model(data_path: str, model_path: str):
    dataset = load_data(data_path)

    torch.manual_seed(0)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    _, _, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
    )

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = BeliefNetwork(dropout_rate=0.1, hidden_size=20, output_size=24)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_predictions = []

    with torch.no_grad():
        for batch_x, _ in test_loader:
            outputs = model(batch_x.to(get_device()))
            all_predictions.append(outputs)

    all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()

    # Prepare data for seaborn strip plot
    data = []
    for i in range(all_predictions.shape[1]):
        for prediction in all_predictions[:, i]:
            data.append((i, prediction))

    df = pd.DataFrame(data, columns=["Output Element", "Prediction"])

    plt.figure(figsize=(12, 8))
    sns.stripplot(x="Output Element", y="Prediction", data=df)
    plt.xlabel("Output Element", fontsize=14)
    plt.ylabel("Prediction", fontsize=14)
    plt.title("Predictions per Output Element on Test Set", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("belief/predictions_plot.pdf")
    plt.show()


if __name__ == "__main__":
    data_path = "belief/dataset.pt"
    model_path = "belief/belief_net_lr_0.001_dropout_0.1_hidden_20.pth"
    evaluate_model(data_path, model_path)
