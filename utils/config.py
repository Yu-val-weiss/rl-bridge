import pathlib
from dataclasses import dataclass
from typing import Optional

import yaml
from dacite import Config, from_dict


@dataclass
class PolicyNetConfig:
    """Defines configuration for the policy network"""

    input_size: int
    hidden_size: int
    output_size: int


@dataclass
class ValueNetConfig:
    """Defines configuration for the value network"""

    input_size: int
    hidden_size: int


@dataclass
class ActorConfig:
    """Defines configuration for actor threads"""

    num_actors: int
    sync_frequency: int
    max_steps: int


@dataclass
class LearnerConfig:
    """Defines configuration for learner thread"""

    batch_size: int
    beta: float
    max_steps: int
    lr_pol: float
    lr_val: float
    clip_eps: float


@dataclass
class BufferConfig:
    """Defines configuration for TensorDictReplayBuffer"""

    alpha: float
    beta: float
    max_capacity: int


@dataclass
class WBConfig:
    project: str = "RL"
    entity: str = "biermann-carla-university-of-cambridge"
    run_name: str = ""


@dataclass
class SelfPlayConfig:
    """Defines configuration for self play training"""

    policy_net: PolicyNetConfig
    value_net: ValueNetConfig
    actor: ActorConfig
    learner: LearnerConfig
    replay_buffer: BufferConfig
    wandb: Optional[WBConfig]
    checkpoint_path: str
    checkpoint_every: int
    load_policy_net: Optional[str]


def load_self_play_config(config_path: pathlib.Path) -> SelfPlayConfig:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        SelfPlayConfig object
    """
    with config_path.open("r") as f:
        config_d = yaml.safe_load(f)

    return from_dict(
        data_class=SelfPlayConfig,
        data=config_d,
        config=Config(strict=True),
    )
