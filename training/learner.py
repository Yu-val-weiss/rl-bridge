import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torchrl.data import ReplayBuffer
from models.policy_network import PolicyNetwork
from models.value_network import ValueNetwork
import gym
import torch.nn.functional as F


def train(policy_net: PolicyNetwork, value_net: ValueNetwork, replay_buffer: ReplayBuffer, policy_optimizer: optim.Optimizer, value_optimizer: optim.Optimizer, batch_size: int, beta: float):
    for i in range(1, 1001):
        batch = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Update policy network
        policy_optimizer.zero_grad()
        action_probs = policy_net(states)
        action_log_probs = torch.log(torch.sum(action_probs * actions, dim=1))
        values = value_net(states).squeeze()
        advantages = rewards - values

        policy_loss = -torch.mean(action_log_probs * advantages)
        entropy_loss = -torch.mean(action_probs * torch.log(action_probs))
        total_policy_loss = policy_loss + beta * entropy_loss

        total_policy_loss.backward()
        policy_optimizer.step()

        # Update value network
        value_optimizer.zero_grad()
        values = value_net(states).squeeze()
        value_loss = torch.mean((rewards - values) ** 2)

        value_loss.backward()
        value_optimizer.step()

        # Update priorities
        for idx, (state, action, reward, next_state, done) in enumerate(batch):
            priority = abs(reward - value_net(state.unsqueeze(0)).item())
            replay_buffer.update_priority(idx, priority)

# Actor critic function from https://medium.com/@dixitaniket76/advantage-actor-critic-a2c-algorithm-explained-and-implemented-in-pytorch-dc3354b60b50
def actor_critic(actor, critic, episodes, max_steps=2000, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3):
    optimizer_actor = optim.AdamW(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.AdamW(critic.parameters(), lr=lr_critic)
    stats = {'Actor Loss': [], 'Critic Loss': [], 'Returns': []}

    env = gym.make('CartPole-v1')
    input_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    for episode in range(1, episodes + 1):
        state = env.reset()[0]
        ep_return = 0
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            state_tensor = torch.FloatTensor(state)
            
            # Actor selects action
            action_probs = actor(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            
            # Take action and observe next state and reward
            next_state, reward, done, _,_ = env.step(action.item())
            
            # Critic estimates value function
            value = critic(state_tensor)
            next_value = critic(torch.FloatTensor(next_state))
            
            # Calculate TD target and Advantage
            td_target = reward + gamma * next_value * (1 - done)
            advantage = td_target - value
            
            # Critic update with MSE loss
            critic_loss = F.mse_loss(value, td_target.detach())
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()
            
            # Actor update
            log_prob = dist.log_prob(action)
            actor_loss = -log_prob * advantage.detach()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()
            
            # Update state, episode return, and step count
            state = next_state
            ep_return += reward
            step_count += 1

        # Record statistics
        stats['Actor Loss'].append(actor_loss.item())
        stats['Critic Loss'].append(critic_loss.item())
        stats['Returns'].append(ep_return)

        # Print episode statistics
        print(f"Episode {episode}: Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Return: {ep_return}, Steps: {step_count}")

    env.close()
    return stats


if __name__ == "__main__":
    state_dim = 4
    action_dim = 2
    replay_buffer = ReplayBuffer(size=10000)
    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.001)
    batch_size = 32
    beta = 0.01

    # Train using the original train function
    train(policy_net, value_net, replay_buffer, policy_optimizer, value_optimizer, batch_size, beta)

    # Train using the actor_critic function
    actor_critic(policy_net, value_net, episodes=1000)
