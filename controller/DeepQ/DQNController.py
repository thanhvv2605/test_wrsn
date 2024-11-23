import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque
import numpy as np
from controller.DeepQ.DQN_model import DQN
from controller.DeepQ.ReplayBuffer import ReplayBuffer

class DQNController:
    def __init__(self, num_agents, state_dim, action_dim, config):
        # Load hyperparameters from config
        self.num_agents = num_agents  # Number of agents (mobile chargers)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 1e-3)
        self.epsilon_start = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 500)
        self.target_update_freq = config.get("target_update", 10)
        self.batch_size = config.get("batch_size", 64)

        # Initialize epsilon
        self.epsilon = self.epsilon_start

        # Replay buffers for all agents
        self.replay_buffers = [ReplayBuffer(capacity=10000) for _ in range(num_agents)]

        # Networks for all agents
        self.q_networks = [DQN(state_dim, action_dim) for _ in range(num_agents)]
        self.target_networks = [DQN(state_dim, action_dim) for _ in range(num_agents)]
        self.optimizers = [optim.Adam(self.q_networks[i].parameters(), lr=self.lr) for i in range(num_agents)]

        # Sync target networks with Q-networks
        for i in range(num_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
            self.target_networks[i].eval()

        self.steps_done = 0  # For epsilon decay

    def select_action(self, agent_id, state):
        """
        Select an action for a specific agent based on the current state using epsilon-greedy policy.
        """
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            # Exploitation: choose the best action from the agent's Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_networks[agent_id](state_tensor)
            action = int(torch.argmax(q_values).item())
        else:
            # Exploration: choose a random action
            action = random.randrange(self.action_dim)
        return action

    def store_transition(self, agent_id, state, action, reward, next_state, done):
        """
        Store experience in the replay buffer of a specific agent.
        """
        self.replay_buffers[agent_id].store(state, action, reward, next_state, done)

    def train_agent(self, agent_id):
        """
        Train the Q-network of a specific agent using its replay buffer.
        """
        if len(self.replay_buffers[agent_id]) < self.batch_size:
            return  # Not enough samples to train

        # Sample a mini-batch from the replay buffer
        batch = self.replay_buffers[agent_id].sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Q-values of current states
        q_values = self.q_networks[agent_id](states).gather(1, actions)

        # Q-values of next states from target network
        with torch.no_grad():
            max_next_q_values = self.target_networks[agent_id](next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute loss and update the agent's Q-network
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizers[agent_id].zero_grad()
        loss.backward()
        self.optimizers[agent_id].step()

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def sync_target_network(self, agent_id):
        """
        Update the target network of a specific agent to match its Q-network.
        """
        if self.steps_done % self.target_update_freq == 0:
            self.target_networks[agent_id].load_state_dict(self.q_networks[agent_id].state_dict())

    def save_model(self, agent_id, path):
        """
        Save the Q-network of a specific agent to a file.
        """
        torch.save(self.q_networks[agent_id].state_dict(), path)

    def load_model(self, agent_id, path):
        """
        Load the Q-network of a specific agent from a file.
        """
        self.q_networks[agent_id].load_state_dict(torch.load(path))
        self.q_networks[agent_id].eval()