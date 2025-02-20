import numpy as np
import random
import torch
from torchrl.data import ReplayBuffer, TensorDictReplayBuffer

class Actor:
    def __init__(self, policy_net, value_net, replay_buffer, sync_freq):
        self.policy_net = policy_net
        self.value_net = value_net
        self.replay_buffer = replay_buffer
        self.local_buffer = []
        self.sync_freq = sync_freq
        self.steps = 0

    def synchronize(self, learner_policy_net, learner_value_net):
        #self.policy_net.load_state_dict(learner_policy_net.state_dict())
        #self.value_net.load_state_dict(learner_value_net.state_dict())
        pass

    def sample_action(self, state):
        with torch.no_grad():
            action_probs = self.policy_net(state)
        action = np.random.choice(len(action_probs), p=action_probs.numpy())
        return action

    def run(self, env, learner_policy_net, learner_value_net, T):
        for i in range(1, 1000):
            if i % self.sync_freq == 0:
                self.synchronize(learner_policy_net, learner_value_net)

            state = env.reset()
            for t in range(1, T + 1):
                action = self.sample_action(state)
                next_state, reward, done, _ = env.step(action)
                value_estimation = self.value_net(state)

                self.local_buffer.append((state, action, value_estimation, reward, next_state, done))
                state = next_state

                if done:
                    break

            for state, action, value_estimation, reward, next_state, done in self.local_buffer:
                self.replay_buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

            self.local_buffer = []

policy_net = 
value_net = 
replay_buffer = TensorDictReplayBuffer(storage=ReplayBuffer(storage_size=10000))
actor = Actor(policy_net, value_net, replay_buffer, sync_freq=100)
actor.run(env, learner_policy_net, learner_value_net, T=10)
