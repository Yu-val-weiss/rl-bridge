import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from utils.data import SelfPlayConfig, load_self_play_config


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def config_data():
    return {
        "policy_net": {
            "input_size": 100,
            "hidden_size": 101,
            "output_size": 102,
        },
        "value_net": {
            "input_size": 100,
            "hidden_size": 101,
        },
        "actor": {
            "num_actors": 1,
            "sync_frequency": 2,
            "max_steps": 3,
        },
        "learner": {
            "batch_size": 32,
            "beta": 0.99,
            "max_steps": 400,
            "lr_pol": 0.01,
            "lr_val": 0.01,
            "clip_eps": 0.2,
        },
        "replay_buffer": {"alpha": 0.01, "beta": 0.03, "max_capacity": 5},
        "checkpoint_path": "tests/safsa",
        "checkpoint_every": 5,
        "wandb": None,
    }


@pytest.fixture
def valid_config_file(temp_dir, config_data):
    config_path = temp_dir / "valid_config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def invalid_yaml_file(temp_dir):
    config_path = temp_dir / "invalid.yaml"
    with config_path.open("w") as f:
        f.write("invalid: yaml: content:\nwith improper formatting")
    return config_path


@pytest.fixture
def invalid_config_file(temp_dir):
    config_data = {"wrong_field": "wrong_value"}
    config_path = temp_dir / "invalid_config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_data, f)
    return config_path


def test_load_valid_config(valid_config_file, config_data):
    config = load_self_play_config(valid_config_file)
    assert isinstance(config, SelfPlayConfig)
    assert asdict(config) == config_data


def test_load_nonexistent_config():
    with pytest.raises(Exception):
        load_self_play_config(Path("nonexistent_config.yaml"))


def test_load_invalid_yaml(invalid_yaml_file):
    with pytest.raises(yaml.YAMLError):
        load_self_play_config(invalid_yaml_file)


def test_load_invalid_config(invalid_config_file):
    with pytest.raises(Exception):  # dacite will raise an exception for invalid config
        load_self_play_config(invalid_config_file)
