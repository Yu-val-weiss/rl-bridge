import threading

import torch.nn as nn
from actor import Actor
from learner import Learner
from torchrl.data import PrioritizedReplayBuffer


def main():
    num_actors = 4
    sync_freq = 10

    policy_net = nn.Sequential(
        nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5)
    )  # Example network
    value_net = nn.Sequential(
        nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1)
    )  # Example network
    replay_buffer = PrioritizedReplayBuffer(size=1000)

    actors = [
        Actor(
            actor_id=i,
            policy_net=policy_net,
            value_net=value_net,
            replay_buffer=replay_buffer,
            sync_freq=sync_freq,
        )
        for i in range(num_actors)
    ]

    learner = Learner(
        policy_net=policy_net, value_net=value_net, replay_buffer=replay_buffer
    )

    actor_threads = [threading.Thread(target=actor.run) for actor in actors]
    learner_thread = threading.Thread(target=learner.run)

    for thread in actor_threads:
        thread.start()

    learner_thread.start()

    for thread in actor_threads:
        thread.join()

    learner_thread.join()


if __name__ == "__main__":
    main()
