import logging
import pathlib
import threading

import click
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler

from models import PolicyNetwork, ValueNetwork
from training import Actor, Learner
from utils import load_config
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
def self_play(config_path: pathlib.Path):
    conf = load_config(config_path)

    # initialise networks
    policy_net = PolicyNetwork.from_dataclass(conf.policy_net)
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

    # initialise learner
    learner = Learner.from_config(policy_net, value_net, buffer, lock, conf.learner)

    # initialise actors
    actors = Actor.from_config(policy_net, value_net, buffer, lock, conf.actor)

    threads: list[threading.Thread] = []

    stop_event = threading.Event()

    def learner_wrapper():
        try:
            learner.train_loop(conf.checkpoint_path, conf.checkpoint_every)
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
        thread.start()

    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logging.info("killed by interrupt!")
        return

    logging.info("iteration complete!")


if __name__ == "__main__":
    self_play()
