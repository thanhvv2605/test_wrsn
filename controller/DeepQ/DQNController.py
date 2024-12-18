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
    def __init__(self, num_agents, state_dim, action_dim, config, device=torch.device('cpu')):
        # Load hyperparameters from config
        self.device = device
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
        self.replay_buffer_capacity = config.get("replay_buffer_capacity", 10000)

        # Initialize epsilon and steps_done for each agent
        self.epsilon = [self.epsilon_start for _ in range(num_agents)]
        self.steps_done = [0 for _ in range(num_agents)]

        # Replay buffers for all agents
        self.replay_buffers = [ReplayBuffer(capacity=self.replay_buffer_capacity) for _ in range(num_agents)]

        # Networks for all agents
        self.q_networks = [DQN(state_dim, action_dim).to(self.device) for _ in range(num_agents)]
        self.target_networks = [DQN(state_dim, action_dim).to(self.device) for _ in range(num_agents)]
        self.optimizers = [optim.Adam(self.q_networks[i].parameters(), lr=self.lr) for i in range(num_agents)]

        # Sync target networks with Q-networks
        for i in range(num_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
            self.target_networks[i].eval()

    def select_action(self, agent_id, state):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        math.exp(-1. * self.steps_done[agent_id] / self.epsilon_decay)
        self.steps_done[agent_id] += 1

        if sample > eps_threshold:
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                state_tensor = state.unsqueeze(0)  # state is already a tensor
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

        # Debugging: Ensure data is in the correct format
        if isinstance(states, tuple):
            states = list(states)  # Convert tuple to list

        # Ensure states and next_states are numpy arrays for tensor conversion
        try:
            states = np.array(states, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to convert states to numpy array: {e}")

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        print("State -----------------")
        print(states)
        print(type(states))
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

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

        # Update epsilon for the agent
        self.epsilon[agent_id] = max(self.epsilon_end, self.epsilon[agent_id] * self.epsilon_decay)

    def sync_target_network(self, agent_id):
        """
        Update the target network of a specific agent to match its Q-network.
        """
        if self.steps_done[agent_id] % self.target_update_freq == 0:
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