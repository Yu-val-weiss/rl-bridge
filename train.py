import logging
import pathlib
import threading
from dataclasses import asdict

import click
import torch
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler

import wandb
from models import PolicyNetwork, ValueNetwork
from training import Actor, Learner
from utils import load_self_play_config
from utils.mrsw import MRSWLock

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)-7s @ %(name)-7s] %(asctime)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option(
    "-c",
    "--config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Path to config yaml file",
)
@click.option("--resume/--no-resume", help="Whether to auto resume", default=False)
def self_play(config_path: pathlib.Path, resume: bool):
    conf = load_self_play_config(config_path)

    # initialise networks
    policy_net = PolicyNetwork.from_dataclass(conf.policy_net)

    if conf.load_policy_net:
        logging.info(f"Initialising policy network from {conf.load_policy_net}")
        policy_net.load_state_dict(torch.load(conf.load_policy_net))

    value_net = ValueNetwork.from_dataclass(conf.value_net)

    # initialise lock
    lock = MRSWLock()

    # initialise replay buffer
    S = conf.replay_buffer.max_capacity
    storage = LazyMemmapStorage(S)
    sampler = PrioritizedSampler(
        S,
        conf.replay_buffer.alpha,
        conf.replay_buffer.beta,
    )
    buffer = TensorDictReplayBuffer(sampler=sampler, storage=storage)

    sync_events = [threading.Event() for _ in range(conf.actor.num_actors)]

    ckp = pathlib.Path(conf.checkpoint_path)

    ckp.mkdir(parents=True, exist_ok=True)

    # initiliase wandb

    if conf.wandb:
        run_id = None
        ckp_run_id = ckp / "wandb_id"
        if resume and ckp_run_id.exists():
            run_id = ckp_run_id.open("r").read()
        run = wandb.init(
            entity=conf.wandb.entity,
            project=conf.wandb.project,
            name=conf.wandb.run_name,
            config=asdict(conf),
            id=run_id,
        )
        with ckp_run_id.open("w") as f:
            f.write(run.id)

    # initialise learner
    learner = Learner.from_config(
        policy_net, value_net, buffer, lock, conf.learner, conf.wandb is not None
    )

    latest = None
    if resume:
        latest = get_latest_checkpoint(ckp)
        if latest:
            logging.info(f"loading learner from {latest}")
            learner.load(latest)
        else:
            logging.warning("could not find a checkpoint to load from")

    # initialise actors
    actors = Actor.from_config(
        policy_net, value_net, buffer, lock, sync_events, conf.actor
    )

    threads: list[threading.Thread] = []

    stop_event = threading.Event()

    def learner_wrapper():
        try:
            learner.train_loop(
                conf.checkpoint_path,
                conf.checkpoint_every,
                sync_events,
                conf.actor.sync_frequency,
                offset=get_step(latest) if resume and latest is not None else 0,
            )
        finally:
            stop_event.set()  # Signal actors to stop when learner finishes

    threads.append(
        threading.Thread(
            target=learner_wrapper,
            name="learner",
            daemon=True,
        )
    )

    for actor in actors:
        threads.append(
            threading.Thread(
                target=actor.run,
                name=f"actor_{actor.actor_id}",
                daemon=True,
                kwargs={"stop_event": stop_event},
            )
        )

    logging.info("starting threads!")

    for thread in threads:
        logging.info(f"starting thread {thread.getName()}")
        thread.start()

    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logging.info("killed by interrupt!")
        return

    logging.info("iteration complete!")


def get_step(path: pathlib.Path):
    return int(path.with_suffix("").name.split("_")[1])


def get_latest_checkpoint(checkpoint_path: pathlib.Path):
    checkpoints = list(checkpoint_path.glob("*.pt"))
    if checkpoints:
        return max(checkpoints, key=get_step)
    return None


if __name__ == "__main__":
    self_play()
