from pathlib import Path

import torch
from tqdm import tqdm


def transform_checkpoints(checkpoint_path: Path):
    for path in tqdm(list(checkpoint_path.glob("*.pt"))):
        checkpoint: dict = torch.load(path)
        checkpoint["policy_net_opt"] = checkpoint["policy_net"].pop("optimizer")
        checkpoint["value_net_opt"] = checkpoint["value_net"].pop("optimizer")
        checkpoint["self"] = {
            "beta": checkpoint.pop("beta"),
            "batch_size": checkpoint.pop("batch_size"),
            "clip_eps": checkpoint.pop("clip_eps"),
        }
        new_path = path.with_suffix("")
        new_path.mkdir(parents=True, exist_ok=True)
        for key, val in checkpoint.items():
            torch.save(val, new_path / f"{key}.pt")


transform_checkpoints(Path("checkpoints/50k_test"))
